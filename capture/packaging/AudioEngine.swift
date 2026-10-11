// Native duplex audio I/O: a single AUHAL (kAudioUnitSubType_HALOutput) instance, opened on one Core Audio device for
// both input and output at once - the same device/channel-map model already proven today by the Python backend
// (capture/backends.py:CoreAudioBackend, via sounddevice/PortAudio). The channel-map conventions here are taken
// directly from PortAudio's own pa_mac_core.c (which implements exactly this AUHAL setup) so behaviour matches what
// we already tested live on the Fireface: input map is "AU channel -> device channel" (one entry per AU channel we
// want); output map is "device channel -> AU channel, or -1" (one entry per DEVICE output channel) - the very same
// shape as backends.py's CoreAudioBackend.output_map().
//
// The render callback pulls the input with AudioUnitRender, hands the block to RTEngine (native/engine/RTEngine.h,
// via the bridging header) and writes the result to the output buffer. No allocation happens on the audio thread:
// every buffer is sized once in open().

import AudioToolbox
import AVFoundation
import CoreAudio
import Foundation

enum AudioEngineError: Error, CustomStringConvertible {
    case deviceNotFound(String)
    case coreAudio(String, OSStatus)

    var description: String {
        switch self {
        case .deviceNotFound(let name): return "device not found: \(name)"
        case .coreAudio(let step, let status): return "\(step) failed (OSStatus \(status))"
        }
    }
}

final class AudioEngine {
    private var unit: AudioUnit?
    private var engine: OpaquePointer?
    private var inputBufferList: UnsafeMutableAudioBufferListPointer?
    private var scratchIn: UnsafeMutablePointer<Float>?
    private var scratchCapacity: Int = 0
    private(set) var isOpen = false
    private(set) var sampleRate: Double = 48_000
    private(set) var grantedBlocksize: Int = 0
    private(set) var xruns: Int = 0
    private var inPeak: Float = 0
    private let stateLock = NSLock()

    // MARK: - dual-chain mode ("Prova dal vivo" while driving a real pedal, see openDual)
    private var guitarEngine: OpaquePointer?          // gain-only, exactly the unity-gain monitor's own math
    private var dualMode = false
    /// Player mode ("Lettore NAM", openPlayer): a dual-mode layout whose chains are an RTPlayer's (RTEngine.h). When set,
    /// `engine` and `guitarEngine` are NOT owned here: they are the player's leg B and feed engines, the player owns them.
    private var player: OpaquePointer?
    private var inputBufferListDual: UnsafeMutableAudioBufferListPointer?
    private var scratchGuitar: UnsafeMutablePointer<Float>?
    private var scratchReturn: UnsafeMutablePointer<Float>?
    private var scratchPedalOut: UnsafeMutablePointer<Float>?
    private var scratchMonitorOut: UnsafeMutablePointer<Float>?
    private var monitorChannelCountDual = 1
    private var singleOutChannelCount = 1               // single-chain mode: how many AU output buffers render() must fill
    private var guitarInPeak: Float = 0

    // MARK: - device lookup

    private static func findDevice(named name: String) throws -> AudioDeviceID {
        var propertySize: UInt32 = 0
        var address = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDevices,
                                                  mScope: kAudioObjectPropertyScopeGlobal,
                                                  mElement: kAudioObjectPropertyElementMain)
        var status = AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &propertySize)
        guard status == noErr else { throw AudioEngineError.coreAudio("list devices (size)", status) }
        let count = Int(propertySize) / MemoryLayout<AudioDeviceID>.size
        var ids = [AudioDeviceID](repeating: 0, count: count)
        status = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &propertySize, &ids)
        guard status == noErr else { throw AudioEngineError.coreAudio("list devices", status) }

        for id in ids {
            var nameAddress = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyDeviceNameCFString,
                                                          mScope: kAudioObjectPropertyScopeGlobal,
                                                          mElement: kAudioObjectPropertyElementMain)
            var cfName: CFString = "" as CFString
            var size = UInt32(MemoryLayout<CFString>.size)
            let s = withUnsafeMutablePointer(to: &cfName) { ptr -> OSStatus in
                AudioObjectGetPropertyData(id, &nameAddress, 0, nil, &size, ptr)
            }
            if s == noErr, (cfName as String) == name {
                return id
            }
        }
        throw AudioEngineError.deviceNotFound(name)
    }

    // MARK: - open / close

    /// Opens `deviceName` for duplex I/O: `inChannel`/`outChannel` are 0-based physical channel indices (the same
    /// numbering the page already shows as "Canale N" = index N-1). `blocksize` of 0 leaves the device's current
    /// buffer size untouched (mirrors the Python side's "Automatico"); otherwise it is requested on the device.
    /// `extraOutChannels`: further device channels that get the same mono signal as `outChannel` ("Prova dal vivo" listens on a
    /// stereo pair of monitor outputs; the one AU channel is mapped onto every device channel listed).
    func open(deviceName: String, inChannel: Int, outChannel: Int, extraOutChannels: [Int] = [], blocksize: Int, gainDb: Double, ceilingDb: Double) throws {
        close()
        let deviceID = try Self.findDevice(named: deviceName)

        var desc = AudioComponentDescription(componentType: kAudioUnitType_Output,
                                             componentSubType: kAudioUnitSubType_HALOutput,
                                             componentManufacturer: kAudioUnitManufacturer_Apple,
                                             componentFlags: 0, componentFlagsMask: 0)
        guard let component = AudioComponentFindNext(nil, &desc) else {
            throw AudioEngineError.coreAudio("AudioComponentFindNext", -1)
        }
        var newUnit: AudioUnit?
        try check("AudioComponentInstanceNew", AudioComponentInstanceNew(component, &newUnit))
        guard let au = newUnit else { throw AudioEngineError.coreAudio("AudioComponentInstanceNew", -1) }

        var enable: UInt32 = 1
        try check("enable input IO", AudioUnitSetProperty(au, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, 1, &enable, UInt32(MemoryLayout<UInt32>.size)))
        try check("enable output IO", AudioUnitSetProperty(au, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0, &enable, UInt32(MemoryLayout<UInt32>.size)))

        var dev = deviceID
        try check("set current device", AudioUnitSetProperty(au, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &dev, UInt32(MemoryLayout<AudioDeviceID>.size)))

        if blocksize > 0 {
            var frames = UInt32(blocksize)
            var bufAddress = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyBufferFrameSize,
                                                         mScope: kAudioObjectPropertyScopeGlobal,
                                                         mElement: kAudioObjectPropertyElementMain)
            _ = AudioObjectSetPropertyData(deviceID, &bufAddress, 0, nil, UInt32(MemoryLayout<UInt32>.size), &frames)
            // Not fatal if refused: we read back whatever the device actually grants right after, same spirit as
            // capture/backends.py's probe ("what appears is always what was really granted, not what was asked").
        }

        let rate = try Self.currentSampleRate(deviceID)
        var format = AudioStreamBasicDescription(mSampleRate: rate, mFormatID: kAudioFormatLinearPCM,
                                                  mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
                                                  mBytesPerPacket: 4, mFramesPerPacket: 1, mBytesPerFrame: 4,
                                                  mChannelsPerFrame: 1, mBitsPerChannel: 32, mReserved: 0)
        try check("input stream format", AudioUnitSetProperty(au, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 1, &format, UInt32(MemoryLayout<AudioStreamBasicDescription>.size)))
        // One output channel keeps the plain mono format. Several (a stereo monitor pair): a non-interleaved N-channel format,
        // one AU channel per device channel, the same mono signal copied into each in render() - the same way openDual feeds
        // its monitor pair (mapping ONE AU channel onto two device channels did not make the interface play).
        let allOut = [outChannel] + extraOutChannels
        var outFormat = format
        if allOut.count > 1 {
            outFormat.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked | kAudioFormatFlagIsNonInterleaved
            outFormat.mChannelsPerFrame = UInt32(allOut.count)
        }
        try check("output stream format", AudioUnitSetProperty(au, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &outFormat, UInt32(MemoryLayout<AudioStreamBasicDescription>.size)))

        var inMap: [Int32] = [Int32(inChannel)]
        try check("input channel map", AudioUnitSetProperty(au, kAudioOutputUnitProperty_ChannelMap, kAudioUnitScope_Output, 1, &inMap, UInt32(inMap.count * MemoryLayout<Int32>.size)))

        let deviceOutChannels = try Self.channelCount(deviceID, input: false)
        var outMap = [Int32](repeating: -1, count: max(deviceOutChannels, (allOut.max() ?? 0) + 1))
        for (i, ch) in allOut.enumerated() { outMap[ch] = Int32(i) }
        singleOutChannelCount = allOut.count
        try check("output channel map", AudioUnitSetProperty(au, kAudioOutputUnitProperty_ChannelMap, kAudioUnitScope_Output, 0, &outMap, UInt32(outMap.count * MemoryLayout<Int32>.size)))

        guard let e = rt_create(rate) else { throw AudioEngineError.coreAudio("rt_create", -1) }
        rt_set_ceiling_db(e, ceilingDb)
        rt_set_gain_db(e, gainDb)
        engine = e

        let maxFrames = 8192
        scratchIn = UnsafeMutablePointer<Float>.allocate(capacity: maxFrames)
        scratchIn?.initialize(repeating: 0, count: maxFrames)
        scratchCapacity = maxFrames
        let abl = AudioBufferList.allocate(maximumBuffers: 1)
        abl[0] = AudioBuffer(mNumberChannels: 1, mDataByteSize: UInt32(maxFrames * 4), mData: UnsafeMutableRawPointer(scratchIn))
        inputBufferList = abl

        var callback = AURenderCallbackStruct(inputProc: audioEngineRenderCallback, inputProcRefCon: Unmanaged.passUnretained(self).toOpaque())
        try check("set render callback", AudioUnitSetProperty(au, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Global, 0, &callback, UInt32(MemoryLayout<AURenderCallbackStruct>.size)))

        try check("AudioUnitInitialize", AudioUnitInitialize(au))
        try check("AudioOutputUnitStart", AudioOutputUnitStart(au))

        unit = au
        sampleRate = rate
        grantedBlocksize = Int(try Self.currentBufferFrameSize(deviceID))
        xruns = 0
        isOpen = true
    }

    /// Opens the device for the "Pedale" live-test's dual chain: the guitar, continuously, at unity gain, out
    /// to the channel driving the real pedal - AT THE SAME TIME as the pedal's return through NAM+IR, out to a
    /// SEPARATE monitor output (Roman: needs to hear the pedal through NAM+IR while it is actually driving the
    /// pedal, not one or the other - stopping the guitar feed to listen would also silence the pedal's own
    /// return). One AUHAL instance, two independent RTEngine instances: `guitarEngine` (gain only, unloaded
    /// model/IR - literally the same math as the unity-gain monitor above) and `engine`, the same NAM/IR-
    /// capable one `open()` already uses, so loadModel/loadIR/setPreGainDb/setGainDb keep meaning what they
    /// already mean elsewhere in this file; only the guitar leg's own gain gets a new setter (setGuitarGainDb).
    /// `monitorOutChannels`: 1 (mono) or 2 (stereo - the same mono result duplicated onto both) channels.
    func openDual(deviceName: String, guitarChannel: Int, returnChannel: Int, pedalOutChannel: Int,
                  monitorOutChannels: [Int], blocksize: Int, guitarGainDb: Double, ceilingDb: Double) throws {
        try openDualCommon(deviceName: deviceName, guitarChannel: guitarChannel, returnChannel: returnChannel, pedalOutChannel: pedalOutChannel,
                           monitorOutChannels: monitorOutChannels, blocksize: blocksize, guitarGainDb: guitarGainDb, ceilingDb: ceilingDb, asPlayer: false)
    }

    /// The NAM player: the guitar goes through the model (leg B) and, at the same time, out to the reamp output and back through the
    /// real gear (leg A); the monitor hears one of them (selectLeg). Same channels and layout as openDual - all the routing is RTPlayer's.
    func openPlayer(deviceName: String, guitarChannel: Int, returnChannel: Int, reampOutChannel: Int,
                    monitorOutChannels: [Int], blocksize: Int, feedGainDb: Double, ceilingDb: Double) throws {
        try openDualCommon(deviceName: deviceName, guitarChannel: guitarChannel, returnChannel: returnChannel, pedalOutChannel: reampOutChannel,
                           monitorOutChannels: monitorOutChannels, blocksize: blocksize, guitarGainDb: feedGainDb, ceilingDb: ceilingDb, asPlayer: true)
    }

    private func openDualCommon(deviceName: String, guitarChannel: Int, returnChannel: Int, pedalOutChannel: Int,
                                monitorOutChannels: [Int], blocksize: Int, guitarGainDb: Double, ceilingDb: Double, asPlayer: Bool) throws {
        close()
        guard monitorOutChannels.count == 1 || monitorOutChannels.count == 2 else {
            throw AudioEngineError.coreAudio("openDual: monitorOutChannels must have 1 or 2 channels", -1)
        }
        let deviceID = try Self.findDevice(named: deviceName)

        var desc = AudioComponentDescription(componentType: kAudioUnitType_Output,
                                             componentSubType: kAudioUnitSubType_HALOutput,
                                             componentManufacturer: kAudioUnitManufacturer_Apple,
                                             componentFlags: 0, componentFlagsMask: 0)
        guard let component = AudioComponentFindNext(nil, &desc) else {
            throw AudioEngineError.coreAudio("AudioComponentFindNext", -1)
        }
        var newUnit: AudioUnit?
        try check("AudioComponentInstanceNew", AudioComponentInstanceNew(component, &newUnit))
        guard let au = newUnit else { throw AudioEngineError.coreAudio("AudioComponentInstanceNew", -1) }

        var enable: UInt32 = 1
        try check("enable input IO", AudioUnitSetProperty(au, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, 1, &enable, UInt32(MemoryLayout<UInt32>.size)))
        try check("enable output IO", AudioUnitSetProperty(au, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0, &enable, UInt32(MemoryLayout<UInt32>.size)))

        var dev = deviceID
        try check("set current device", AudioUnitSetProperty(au, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &dev, UInt32(MemoryLayout<AudioDeviceID>.size)))

        if blocksize > 0 {
            var frames = UInt32(blocksize)
            var bufAddress = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyBufferFrameSize,
                                                         mScope: kAudioObjectPropertyScopeGlobal,
                                                         mElement: kAudioObjectPropertyElementMain)
            _ = AudioObjectSetPropertyData(deviceID, &bufAddress, 0, nil, UInt32(MemoryLayout<UInt32>.size), &frames)
        }

        let rate = try Self.currentSampleRate(deviceID)

        // 2 input channels (guitar, return), non-interleaved: one mono buffer per channel. AUHAL's own
        // canonical format is already non-interleaved for a single channel (trivially); explicit here because
        // there is more than one now.
        var inFormat = AudioStreamBasicDescription(mSampleRate: rate, mFormatID: kAudioFormatLinearPCM,
                                                    mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked | kAudioFormatFlagIsNonInterleaved,
                                                    mBytesPerPacket: 4, mFramesPerPacket: 1, mBytesPerFrame: 4,
                                                    mChannelsPerFrame: 2, mBitsPerChannel: 32, mReserved: 0)
        try check("input stream format", AudioUnitSetProperty(au, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 1, &inFormat, UInt32(MemoryLayout<AudioStreamBasicDescription>.size)))

        let outChannelCountAU = 1 + monitorOutChannels.count   // AU channel 0 = pedal feed, 1(-2) = monitor
        var outFormat = AudioStreamBasicDescription(mSampleRate: rate, mFormatID: kAudioFormatLinearPCM,
                                                     mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked | kAudioFormatFlagIsNonInterleaved,
                                                     mBytesPerPacket: 4, mFramesPerPacket: 1, mBytesPerFrame: 4,
                                                     mChannelsPerFrame: UInt32(outChannelCountAU), mBitsPerChannel: 32, mReserved: 0)
        try check("output stream format", AudioUnitSetProperty(au, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &outFormat, UInt32(MemoryLayout<AudioStreamBasicDescription>.size)))

        var inMap: [Int32] = [Int32(guitarChannel), Int32(returnChannel)]
        try check("input channel map", AudioUnitSetProperty(au, kAudioOutputUnitProperty_ChannelMap, kAudioUnitScope_Output, 1, &inMap, UInt32(inMap.count * MemoryLayout<Int32>.size)))

        let deviceOutChannels = try Self.channelCount(deviceID, input: false)
        let touchedChannels = [pedalOutChannel] + monitorOutChannels
        var outMap = [Int32](repeating: -1, count: max(deviceOutChannels, (touchedChannels.max() ?? 0) + 1))
        outMap[pedalOutChannel] = 0
        for (i, ch) in monitorOutChannels.enumerated() { outMap[ch] = Int32(1 + i) }
        try check("output channel map", AudioUnitSetProperty(au, kAudioOutputUnitProperty_ChannelMap, kAudioUnitScope_Output, 0, &outMap, UInt32(outMap.count * MemoryLayout<Int32>.size)))

        if asPlayer {
            guard let p = rt_player_create(rate) else { throw AudioEngineError.coreAudio("rt_player_create", -1) }
            for part in [RT_LEG_FEED, RT_LEG_A, RT_LEG_B] { rt_set_ceiling_db(rt_player_engine(p, part), ceilingDb) }
            rt_set_gain_db(rt_player_engine(p, RT_LEG_FEED), guitarGainDb)
            player = p
            guitarEngine = rt_player_engine(p, RT_LEG_FEED)     // not owned (see `player`)
            engine = rt_player_engine(p, RT_LEG_B)
        } else {
            guard let ge = rt_create(rate) else { throw AudioEngineError.coreAudio("rt_create (guitar)", -1) }
            rt_set_ceiling_db(ge, ceilingDb)
            rt_set_gain_db(ge, guitarGainDb)
            guitarEngine = ge

            guard let le = rt_create(rate) else { throw AudioEngineError.coreAudio("rt_create (live)", -1) }
            rt_set_ceiling_db(le, ceilingDb)
            engine = le
        }

        let maxFrames = 8192
        scratchGuitar = UnsafeMutablePointer<Float>.allocate(capacity: maxFrames)
        scratchReturn = UnsafeMutablePointer<Float>.allocate(capacity: maxFrames)
        scratchPedalOut = UnsafeMutablePointer<Float>.allocate(capacity: maxFrames)
        scratchMonitorOut = UnsafeMutablePointer<Float>.allocate(capacity: maxFrames)
        for p in [scratchGuitar, scratchReturn, scratchPedalOut, scratchMonitorOut] { p?.initialize(repeating: 0, count: maxFrames) }
        scratchCapacity = maxFrames
        monitorChannelCountDual = monitorOutChannels.count

        let abl = AudioBufferList.allocate(maximumBuffers: 2)
        abl[0] = AudioBuffer(mNumberChannels: 1, mDataByteSize: UInt32(maxFrames * 4), mData: UnsafeMutableRawPointer(scratchGuitar))
        abl[1] = AudioBuffer(mNumberChannels: 1, mDataByteSize: UInt32(maxFrames * 4), mData: UnsafeMutableRawPointer(scratchReturn))
        inputBufferListDual = abl

        var callback = AURenderCallbackStruct(inputProc: audioEngineRenderCallback, inputProcRefCon: Unmanaged.passUnretained(self).toOpaque())
        try check("set render callback", AudioUnitSetProperty(au, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Global, 0, &callback, UInt32(MemoryLayout<AURenderCallbackStruct>.size)))

        try check("AudioUnitInitialize", AudioUnitInitialize(au))
        try check("AudioOutputUnitStart", AudioOutputUnitStart(au))

        unit = au
        dualMode = true
        sampleRate = rate
        grantedBlocksize = Int(try Self.currentBufferFrameSize(deviceID))
        xruns = 0
        isOpen = true
    }

    func close() {
        if let au = unit {
            if let p = player {
                rt_player_stop(p)       // all three parts fade together
                var tries = 0
                while rt_player_is_finished(p) == 0 && tries < 50 { Thread.sleep(forTimeInterval: 0.01); tries += 1 }
            } else if let e = engine {
                rt_stop(e)
                // give the fade a moment to finish so stopping never clicks; the render callback keeps running meanwhile
                var tries = 0
                while rt_is_finished(e) == 0 && tries < 50 { Thread.sleep(forTimeInterval: 0.01); tries += 1 }
            }
            if player == nil, dualMode, let ge = guitarEngine {
                rt_stop(ge)   // the guitar leg fades out together with the live one - same "never a click" contract
                var tries = 0
                while rt_is_finished(ge) == 0 && tries < 50 { Thread.sleep(forTimeInterval: 0.01); tries += 1 }
            }
            AudioOutputUnitStop(au)
            AudioUnitUninitialize(au)
            AudioComponentInstanceDispose(au)
        }
        if let p = player { rt_player_destroy(p) }          // owns the engines behind `engine` / `guitarEngine`
        else {
            if let e = engine { rt_destroy(e) }
            if let ge = guitarEngine { rt_destroy(ge) }
        }
        player = nil
        scratchIn?.deallocate()
        inputBufferList?.unsafeMutablePointer.deallocate()
        scratchGuitar?.deallocate(); scratchReturn?.deallocate(); scratchPedalOut?.deallocate(); scratchMonitorOut?.deallocate()
        inputBufferListDual?.unsafeMutablePointer.deallocate()
        unit = nil
        engine = nil
        guitarEngine = nil
        scratchIn = nil
        inputBufferList = nil
        scratchGuitar = nil; scratchReturn = nil; scratchPedalOut = nil; scratchMonitorOut = nil
        inputBufferListDual = nil
        dualMode = false
        isOpen = false
    }

    /// The guitar leg's own gain, dual mode only (unity gain, G = Ci_guitar - Co - the same formula as the
    /// plain guitar monitor). No-op outside dual mode.
    func setGuitarGainDb(_ db: Double) {
        if let ge = guitarEngine { rt_set_gain_db(ge, db) }
    }

    func setGainDb(_ db: Double) {
        if let e = engine { rt_set_gain_db(e, db) }
    }

    // MARK: player mode only (no-ops otherwise)

    /// 0 = A (the real gear, through the return), 1 = B (the model).
    func selectLeg(_ leg: Int) { if let p = player { rt_player_select(p, Int32(leg)) } }
    /// Leg A's gain: return -> monitor (Ci_return - Co).
    func setAmpGainDb(_ db: Double) { if let p = player { rt_set_gain_db(rt_player_engine(p, RT_LEG_A), db) } }
    func setBTrimDb(_ db: Double) { if let p = player { rt_player_set_b_trim_db(p, db) } }
    func setBDelay(_ samples: Int) { if let p = player { rt_player_set_b_delay(p, Int32(samples)) } }
    /// (leg, a_peak, b_peak, ir_a_loaded) of the player, nil outside player mode.
    func playerState() -> (leg: Int, aPeakDb: Double, bPeakDb: Double, irALoaded: Bool)? {
        guard let p = player else { return nil }
        return (Int(rt_player_selected(p)), rt_player_peak_a_db(p), rt_player_peak_b_db(p),
                rt_ir_loaded(rt_player_engine(p, RT_LEG_A)) == 1)
    }

    /// The gain applied BEFORE the NAM model, if any (see RTEngine.h) - unity by default, only meaningful
    /// once a model is loaded, to present it with the level it was calibrated at.
    func setPreGainDb(_ db: Double) {
        if let e = engine { rt_set_pre_gain_db(e, db) }
    }

    /// Noise gate threshold (dBFS, before the model) and 3-band tone stack (bass/mid/treble, after the model,
    /// before the IR) - see RTEngine.h's own comment. No-op if nothing is open, same as the other setters here.
    func setGateThresholdDb(_ db: Double) {
        if let e = engine { rt_set_gate_threshold_db(e, db) }
    }
    func setEqBassDb(_ db: Double) {
        if let e = engine { rt_set_eq_bass_db(e, db) }
    }
    func setEqMidDb(_ db: Double) {
        if let e = engine { rt_set_eq_mid_db(e, db) }
    }
    func setEqTrebleDb(_ db: Double) {
        if let e = engine { rt_set_eq_treble_db(e, db) }
    }
    func setIrInGainDb(_ db: Double) {
        if let e = engine { rt_set_ir_in_gain_db(e, db) }
    }
    func setIrOutGainDb(_ db: Double) {
        if let e = engine { rt_set_ir_out_gain_db(e, db) }
    }

    /// Loads a .nam profile into the currently open engine (see RTEngine.h's rt_load_model). Not real-time
    /// safe by itself, but RTEngine hot-swaps it in without glitching whatever is already playing.
    /// `legs`: bit 0 = leg B, the model's chain (the default); bit 1 = leg A, a second copy of the amp's model for the real pedal's return (player mode).
    func loadModel(path: String, legs: Int = 1) throws {
        guard let e = engine else { throw AudioEngineError.coreAudio("loadModel: interface not open", -1) }
        if legs & 2 != 0 && player == nil { throw AudioEngineError.coreAudio("il modello sul ritorno esiste solo nel lettore NAM", -1) }
        if legs & 1 != 0 {
            let ok = path.withCString { rt_load_model(e, $0) }
            guard ok == 1 else { throw AudioEngineError.coreAudio("model: \(String(cString: rt_last_error(e)))", -1) }
        }
        if legs & 2 != 0, let p = player {
            let a = rt_player_engine(p, RT_LEG_A)
            let ok = path.withCString { rt_load_model(a, $0) }
            guard ok == 1 else { throw AudioEngineError.coreAudio("model: \(String(cString: rt_last_error(a)))", -1) }
        }
    }

    func unloadModel(legs: Int = 1) {
        if legs & 1 != 0, let e = engine { rt_unload_model(e) }
        if legs & 2 != 0, let p = player { rt_unload_model(rt_player_engine(p, RT_LEG_A)) }
    }

    // "Test Pedali" (player mode only): a pedal's NAM in front of the amp's, see RTEngine.h
    private var pedalEngine: OpaquePointer? { player.map { rt_player_engine($0, RT_LEG_PEDAL) } }
    func loadPedal(path: String) throws {
        guard let p = pedalEngine else { throw AudioEngineError.coreAudio("il test pedali esiste solo nel lettore NAM", -1) }
        let ok = path.withCString { rt_load_model(p, $0) }
        guard ok == 1 else { throw AudioEngineError.coreAudio("model: \(String(cString: rt_last_error(p)))", -1) }
    }
    func unloadPedal() { if let p = pedalEngine { rt_unload_model(p) } }
    func pedalSizes() -> Int { pedalEngine.map { Int(rt_model_sizes($0)) } ?? 0 }
    @discardableResult
    func setPedalSize(_ index: Int) -> Int { pedalEngine.map { Int(rt_set_model_size($0, Int32(index))) } ?? -1 }
    func setPedalPreGainDb(_ db: Double) { if let p = pedalEngine { rt_set_pre_gain_db(p, db) } }
    func setPedalGainDb(_ db: Double) { if let p = pedalEngine { rt_set_gain_db(p, db) } }
    /// Leg A: return -> amp model input.
    func setAmpPreGainDb(_ db: Double) { if let p = player { rt_set_pre_gain_db(rt_player_engine(p, RT_LEG_A), db) } }
    /// The pedal model's own levels as the .nam file states them (nil when not loaded / not written).
    func pedalLevels() -> (loaded: Bool, inputDbu: Double?, outputDbu: Double?) {
        guard let p = pedalEngine, rt_model_loaded(p) != 0 else { return (false, nil, nil) }
        let i = rt_model_input_level_dbu(p), o = rt_model_output_level_dbu(p)
        return (true, i.isNaN ? nil : i, o.isNaN ? nil : o)
    }

    /// A2 models (official trainer 0.13): 2 sizes, 0 = Lite, 1 = full; 1 for an ordinary model, 0 with none loaded.
    func modelSizes() -> Int { engine.map { Int(rt_model_sizes($0)) } ?? 0 }

    @discardableResult
    func setModelSize(_ index: Int) -> Int {
        if let p = player { rt_set_model_size(rt_player_engine(p, RT_LEG_A), Int32(index)) }      // the amp's second copy (no-op when it has no model)
        return engine.map { Int(rt_set_model_size($0, Int32(index))) } ?? -1
    }

    /// Loads a cabinet IR (any format AVFoundation can decode - a plain WAV in practice) into the currently
    /// open engine. Decoding and downmixing to mono happen here, in Swift: RTEngine.h itself takes only a
    /// plain decoded sample array, on purpose (see its own comment - file/format I/O stays out of the native
    /// DSP core). The IR's sample rate must match the interface's own rate: this project never resamples
    /// behind the user's back (same rule as the 48 kHz check on the capture side, capture/backends.py) - a
    /// mismatched IR is refused with a clear reason instead of being silently resampled.
    /// `legs`: bit 0 = leg B, the model's chain (the default, and the only one outside player mode); bit 1 = leg A, the real gear's (player mode).
    func loadIR(path: String, legs: Int = 1) throws {
        guard let e = engine else { throw AudioEngineError.coreAudio("loadIR: interface not open", -1) }
        if legs & 2 != 0 && player == nil { throw AudioEngineError.coreAudio("la IR sul ritorno esiste solo nel lettore NAM", -1) }
        let file = try AVAudioFile(forReading: URL(fileURLWithPath: path))
        let format = file.processingFormat
        guard format.sampleRate == sampleRate else {
            throw AudioEngineError.coreAudio(
                "IR a \(Int(format.sampleRate)) Hz, la scheda gira a \(Int(sampleRate)) Hz: esporta la IR alla stessa frequenza, non viene ricampionata", -1)
        }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length)) else {
            throw AudioEngineError.coreAudio("loadIR: buffer allocation", -1)
        }
        try file.read(into: buffer)
        let frameCount = Int(buffer.frameLength)
        guard frameCount > 0, let channelData = buffer.floatChannelData else {
            throw AudioEngineError.coreAudio("loadIR: empty file", -1)
        }
        let channels = Int(format.channelCount)
        var mono = [Float](repeating: 0, count: frameCount)
        for i in 0..<frameCount {
            var sum: Float = 0
            for c in 0..<channels { sum += channelData[c][i] }
            mono[i] = sum / Float(channels)
        }
        var targets: [OpaquePointer] = []
        if legs & 1 != 0 { targets.append(e) }
        if legs & 2 != 0, let p = player, let a = rt_player_engine(p, RT_LEG_A) { targets.append(a) }
        for t in targets {
            let ok = mono.withUnsafeBufferPointer { ptr in rt_load_ir(t, ptr.baseAddress, Int32(frameCount)) }
            guard ok == 1 else { throw AudioEngineError.coreAudio("IR: \(String(cString: rt_last_error(t)))", -1) }
        }
    }

    func unloadIR(legs: Int = 1) {
        if legs & 1 != 0, let e = engine { rt_unload_ir(e) }
        if legs & 2 != 0, let p = player, let a = rt_player_engine(p, RT_LEG_A) { rt_unload_ir(a) }
    }

    func state() -> (inPeakDb: Double, outPeakDb: Double, xruns: Int, modelLoaded: Bool, irLoaded: Bool,
                      modelInputLevelDbu: Double?, modelOutputLevelDbu: Double?,
                      guitarInPeakDb: Double?, guitarOutPeakDb: Double?) {
        stateLock.lock(); defer { stateLock.unlock() }
        let outPeak = player.map { rt_player_peak_out_db($0) } ?? engine.map { rt_last_peak_db($0) } ?? -240.0
        let inDb: Double = inPeak > 0 ? 20.0 * log10(Double(inPeak)) : -240.0
        let modelLoaded = engine.map { rt_model_loaded($0) == 1 } ?? false
        let irLoaded = engine.map { rt_ir_loaded($0) == 1 } ?? false
        let modelInDbu = engine.flatMap { e -> Double? in let v = rt_model_input_level_dbu(e); return v.isNaN ? nil : v }
        let modelOutDbu = engine.flatMap { e -> Double? in let v = rt_model_output_level_dbu(e); return v.isNaN ? nil : v }
        let guitarOutPeak = guitarEngine.map { rt_last_peak_db($0) }
        let guitarInDb: Double? = dualMode ? (guitarInPeak > 0 ? 20.0 * log10(Double(guitarInPeak)) : -240.0) : nil
        return (inDb, outPeak, xruns, modelLoaded, irLoaded, modelInDbu, modelOutDbu, guitarInDb, guitarOutPeak)
    }

    /// The note the tuner hears on the guitar side: the guitar engine in the dual layouts and the player, the only engine otherwise.
    /// nil while nothing clear is playing. The analysis runs here, on the HTTP thread (see native/engine/Tuner.h).
    func tuner() -> (hz: Double, confidence: Double)? {
        stateLock.lock(); defer { stateLock.unlock() }
        guard let e = guitarEngine ?? engine else { return nil }
        var hz: Float = 0, confidence: Float = 0
        guard rt_tuner_read(e, &hz, &confidence) != 0 else { return nil }
        return (Double(hz), Double(confidence))
    }

    // MARK: - render (called on the real-time audio thread)

    fileprivate func render(ioActionFlags: UnsafeMutablePointer<AudioUnitRenderActionFlags>, inTimeStamp: UnsafePointer<AudioTimeStamp>,
                            inBusNumber: UInt32, inNumberFrames: UInt32, ioData: UnsafeMutablePointer<AudioBufferList>?) -> OSStatus {
        if dualMode {
            return renderDual(ioActionFlags: ioActionFlags, inTimeStamp: inTimeStamp, inBusNumber: inBusNumber, inNumberFrames: inNumberFrames, ioData: ioData)
        }
        guard let au = unit, let abl = inputBufferList, let scratch = scratchIn, let e = engine else { return noErr }
        let n = Int(inNumberFrames)
        if n > scratchCapacity {
            // a device asking for more frames than we pre-allocated: play silence this block rather than overrun
            if let out = ioData { silence(out, count: n) }
            return noErr
        }
        abl[0].mDataByteSize = UInt32(n * 4)
        let status = AudioUnitRender(au, ioActionFlags, inTimeStamp, 1, inNumberFrames, abl.unsafeMutablePointer)
        if status != noErr {
            xruns += 1
            if let out = ioData { silence(out, count: n) }
            return noErr
        }
        inPeak = 0
        for i in 0..<n { inPeak = max(inPeak, abs(scratch[i])) }
        guard let out = ioData, out.pointee.mNumberBuffers > 0 else { return noErr }
        let outPtr = out.pointee.mBuffers.mData?.assumingMemoryBound(to: Float.self)
        rt_process(e, scratch, outPtr, Int32(n))
        if singleOutChannelCount > 1, let src = outPtr {
            let buffers = UnsafeMutableAudioBufferListPointer(out)
            for c in 1..<min(singleOutChannelCount, buffers.count) {
                if let p = buffers[c].mData?.assumingMemoryBound(to: Float.self) { p.update(from: src, count: n) }
            }
        }
        return noErr
    }

    /// Two inputs (guitar, return) in one AudioUnitRender call, two independent RTEngine chains, up to three
    /// outputs (pedal feed + 1-2 monitor channels) - see openDual's own comment for why.
    private func renderDual(ioActionFlags: UnsafeMutablePointer<AudioUnitRenderActionFlags>, inTimeStamp: UnsafePointer<AudioTimeStamp>,
                            inBusNumber: UInt32, inNumberFrames: UInt32, ioData: UnsafeMutablePointer<AudioBufferList>?) -> OSStatus {
        guard let au = unit, let abl = inputBufferListDual, let sg = scratchGuitar, let sr = scratchReturn,
              let sp = scratchPedalOut, let sm = scratchMonitorOut else { return noErr }
        let n = Int(inNumberFrames)
        if n > scratchCapacity {
            if let out = ioData { silence(out, count: n) }
            return noErr
        }
        abl[0].mDataByteSize = UInt32(n * 4)
        abl[1].mDataByteSize = UInt32(n * 4)
        let status = AudioUnitRender(au, ioActionFlags, inTimeStamp, 1, inNumberFrames, abl.unsafeMutablePointer)
        if status != noErr {
            xruns += 1
            if let out = ioData { silence(out, count: n) }
            return noErr
        }
        var gp: Float = 0, rp: Float = 0
        for i in 0..<n { gp = max(gp, abs(sg[i])); rp = max(rp, abs(sr[i])) }
        inPeak = rp             // /state's in_peak_dbfs keeps meaning "the return", same as single-engine live mode
        guitarInPeak = gp

        if let p = player {
            rt_player_process(p, sg, sr, sp, sm, Int32(n))   // the whole A/B routing, see RTEngine.h
        } else if let ge = guitarEngine, let le = engine {
            rt_process(ge, sg, sp, Int32(n))   // guitar -> pedal feed, unity gain, no NAM/IR
            rt_process(le, sr, sm, Int32(n))   // return -> [NAM] -> [IR] -> monitor
        } else {
            return noErr
        }

        guard let out = ioData, out.pointee.mNumberBuffers >= 1 + monitorChannelCountDual else { return noErr }
        let buffers = UnsafeMutableAudioBufferListPointer(out)
        if let p = buffers[0].mData?.assumingMemoryBound(to: Float.self) { p.update(from: sp, count: n) }
        for c in 0..<monitorChannelCountDual {
            if let p = buffers[1 + c].mData?.assumingMemoryBound(to: Float.self) { p.update(from: sm, count: n) }
        }
        return noErr
    }

    private func silence(_ list: UnsafeMutablePointer<AudioBufferList>, count: Int) {
        let buffers = UnsafeMutableAudioBufferListPointer(list)
        for i in 0..<buffers.count {
            if let p = buffers[i].mData { memset(p, 0, Int(buffers[i].mDataByteSize)) }
        }
    }

    private func check(_ step: String, _ status: OSStatus) throws {
        guard status == noErr else { throw AudioEngineError.coreAudio(step, status) }
    }

    private static func currentSampleRate(_ device: AudioDeviceID) throws -> Double {
        var rate: Float64 = 48_000
        var size = UInt32(MemoryLayout<Float64>.size)
        var address = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyNominalSampleRate, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        let status = AudioObjectGetPropertyData(device, &address, 0, nil, &size, &rate)
        guard status == noErr else { throw AudioEngineError.coreAudio("read sample rate", status) }
        return rate
    }

    private static func currentBufferFrameSize(_ device: AudioDeviceID) throws -> UInt32 {
        var frames: UInt32 = 0
        var size = UInt32(MemoryLayout<UInt32>.size)
        var address = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyBufferFrameSize, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        let status = AudioObjectGetPropertyData(device, &address, 0, nil, &size, &frames)
        guard status == noErr else { throw AudioEngineError.coreAudio("read buffer size", status) }
        return frames
    }

    private static func channelCount(_ device: AudioDeviceID, input: Bool) throws -> Int {
        var address = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyStreamConfiguration,
                                                  mScope: input ? kAudioDevicePropertyScopeInput : kAudioDevicePropertyScopeOutput,
                                                  mElement: kAudioObjectPropertyElementMain)
        var size: UInt32 = 0
        var status = AudioObjectGetPropertyDataSize(device, &address, 0, nil, &size)
        guard status == noErr, size > 0 else { throw AudioEngineError.coreAudio("channel count (size)", status) }
        let listPtr = UnsafeMutableRawPointer.allocate(byteCount: Int(size), alignment: MemoryLayout<AudioBufferList>.alignment)
        defer { listPtr.deallocate() }
        status = AudioObjectGetPropertyData(device, &address, 0, nil, &size, listPtr)
        guard status == noErr else { throw AudioEngineError.coreAudio("channel count", status) }
        let abl = listPtr.assumingMemoryBound(to: AudioBufferList.self)
        let buffers = UnsafeMutableAudioBufferListPointer(abl)
        return buffers.reduce(0) { $0 + Int($1.mNumberChannels) }
    }
}

private func audioEngineRenderCallback(inRefCon: UnsafeMutableRawPointer, ioActionFlags: UnsafeMutablePointer<AudioUnitRenderActionFlags>,
                                       inTimeStamp: UnsafePointer<AudioTimeStamp>, inBusNumber: UInt32, inNumberFrames: UInt32,
                                       ioData: UnsafeMutablePointer<AudioBufferList>?) -> OSStatus {
    let engine = Unmanaged<AudioEngine>.fromOpaque(inRefCon).takeUnretainedValue()
    return engine.render(ioActionFlags: ioActionFlags, inTimeStamp: inTimeStamp, inBusNumber: inBusNumber, inNumberFrames: inNumberFrames, ioData: ioData)
}
