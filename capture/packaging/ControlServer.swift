// A minimal local HTTP control server for the native AudioEngine, listening on 127.0.0.1 only. No framework: a plain
// BSD socket accept loop parsing just enough HTTP/1.1 (request line + Content-Length body) to carry small JSON
// requests - the same "only what's needed" choice as native/build.sh skipping CMake. The Python server proxies
// /api/capture/native/* here (capture_api.py), so the page never talks to this port directly; it is a second,
// internal seam, not a second thing the user has to think about.
//
// Routes (JSON in, JSON out):
//   POST /open          {device, in_channel, out_channel, extra_out_channels?: [Int], blocksize, gain_db, ceiling_db}
//   POST /open_dual     {device, guitar_channel, return_channel, pedal_out_channel, monitor_out_channels: [Int],
//                         blocksize, guitar_gain_db, ceiling_db}   - see AudioEngine.openDual's own comment
//   POST /open_player   {device, guitar_channel, return_channel, reamp_out_channel, monitor_out_channels: [Int],
//                         blocksize, feed_gain_db, ceiling_db}   - the NAM player (A/B), see AudioEngine.openPlayer
//   POST /leg           {leg: "A"|"B"}     - player: what the monitor hears (A = the real gear via the return, B = the model)
//   POST /amp_gain      {gain_db}          - player, leg A: return -> monitor (Ci_return - Co)
//   POST /b_trim        {trim_db}          - player: fine level trim of leg B against leg A
//   POST /b_delay       {samples}          - player: delays leg B to line it up with the real gear's round trip
//   POST /gain          {gain_db}          - the OUTPUT-side gain (unity monitor: the whole story; NAM+IR: the final trim)
//   POST /pregain       {pre_gain_db}      - the INPUT-side gain, before the NAM model (no-op without one)
//   POST /guitar_gain   {gain_db}          - dual mode only: the guitar leg's own unity gain (G = Ci_guitar - Co)
//   POST /gate          {threshold_db}     - noise gate threshold, before the NAM model (-100 dB default = effectively off)
//   POST /ir_gain       {in_db?, out_db?}  - the cabinet's own level knobs, right before / right after the IR (0 dB default)
//   POST /eq            {bass_db?, mid_db?, treble_db?}  - 3-band tone stack after the model, before the IR (any subset)
//   POST /model         {path, legs?}      - loads a .nam profile; returns its own input/output_level_dbu if it has them; legs default ["B"], ["A"] = the second copy for the real pedal's return
//   POST /model/unload  {}
//   POST /model_size    {index}            - an A2 model's size: 0 = Lite ... sizes-1 = full (the /model reply says how many sizes); the amp: both copies
//   POST /pedal_model   {path}             - player ("Test Pedali"): loads a pedal's .nam in front of the amp's; returns its input/output_level_dbu and sizes
//   POST /pedal_model/unload {}
//   POST /pedal_controls {pre_db?, post_db?, size?}  - the pedal model's input level, output level and A2 size (0 = Lite)
//   POST /amp_pregain   {gain_db}          - player, leg A: return -> amp model input (Ci_return - the amp's input level) when a pedal is tested
//   POST /ir            {path, legs?: ["A","B"]}  - loads a cabinet IR (must be at the interface's own sample rate); legs default ["B"]
//   POST /ir/unload     {legs?}
//   POST /stop          {}
//   GET  /state  -> {opened, in_peak_dbfs, out_peak_dbfs, xruns, blocksize, sample_rate, model_loaded, ir_loaded,
//                    model_input_level_dbu, model_output_level_dbu, guitar_in_peak_dbfs, guitar_out_peak_dbfs, tuner_hz, tuner_confidence,
//                    player, [leg, ir_a_loaded, a_peak_dbfs, b_peak_dbfs when player]}

import Foundation

#if canImport(Glibc)
import Glibc
#else
import Darwin
#endif

final class ControlServer {
    private let engine = AudioEngine()
    private let port: UInt16
    private var listenSocket: Int32 = -1

    init(port: UInt16 = 8767) {
        self.port = port
    }

    func start() {
        // The accept loop blocks forever in accept(): it needs its own thread, not a GCD queue (a serial queue
        // would never free up to run the per-connection handlers; a queue this blocks doesn't cooperate with GCD).
        let thread = Thread { [weak self] in self?.runLoop() }
        thread.name = "namcapture.controlserver.accept"
        thread.start()
    }

    /// Releases the audio device (fading out, never a click) - called on app quit so nothing is left running.
    func stop() {
        engine.close()
    }

    private func runLoop() {
        listenSocket = socket(AF_INET, SOCK_STREAM, 0)
        guard listenSocket >= 0 else { return }
        var reuse: Int32 = 1
        setsockopt(listenSocket, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = inet_addr("127.0.0.1") // loopback only: never reachable from the network
        let bindResult = withUnsafePointer(to: &addr) { ptr -> Int32 in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(listenSocket, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
        }
        guard bindResult == 0, listen(listenSocket, 8) == 0 else {
            NSLog("NAM Capture native control server: could not bind 127.0.0.1:\(port)")
            return
        }
        while true {
            let client = accept(listenSocket, nil, nil)
            if client < 0 { continue }
            // Requests arrive one at a time in practice (the Python relay is the only caller), but a dedicated
            // concurrent queue - not the accept loop's own thread - keeps a slow handler from ever blocking accept().
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in self?.handle(client: client) }
        }
    }

    private func handle(client: Int32) {
        defer { close(client) }
        guard let request = readRequest(client) else { return }
        let (status, body) = route(method: request.method, path: request.path, jsonBody: request.json)
        let payload = (try? JSONSerialization.data(withJSONObject: body)) ?? Data("{}".utf8)
        let response = "HTTP/1.1 \(status)\r\nContent-Type: application/json\r\nContent-Length: \(payload.count)\r\nConnection: close\r\n\r\n"
        let headerBytes = Array(response.utf8)
        headerBytes.withUnsafeBytes { _ = send(client, $0.baseAddress, headerBytes.count, 0) }
        payload.withUnsafeBytes { _ = send(client, $0.baseAddress, payload.count, 0) }
    }

    private struct Request { let method: String; let path: String; let json: [String: Any] }

    private func readRequest(_ fd: Int32) -> Request? {
        var raw = Data()
        var buf = [UInt8](repeating: 0, count: 4096)
        var headerEnd: Range<Data.Index>?
        while headerEnd == nil {
            let n = read(fd, &buf, buf.count)
            if n <= 0 { return nil }
            raw.append(contentsOf: buf[0..<n])
            headerEnd = raw.range(of: Data("\r\n\r\n".utf8))
            if raw.count > 1 << 20 { return nil } // a header this large is not one of ours
        }
        guard let end = headerEnd else { return nil }
        let headerText = String(data: raw[..<end.lowerBound], encoding: .utf8) ?? ""
        let lines = headerText.components(separatedBy: "\r\n")
        guard let first = lines.first else { return nil }
        let parts = first.split(separator: " ")
        guard parts.count >= 2 else { return nil }
        let method = String(parts[0]), path = String(parts[1])
        var contentLength = 0
        for line in lines.dropFirst() {
            if let colon = line.firstIndex(of: ":") {
                let key = line[line.startIndex..<colon].trimmingCharacters(in: .whitespaces).lowercased()
                if key == "content-length" {
                    contentLength = Int(line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)) ?? 0
                }
            }
        }
        var body = raw[end.upperBound...]
        while body.count < contentLength {
            let n = read(fd, &buf, min(buf.count, contentLength - body.count))
            if n <= 0 { break }
            body.append(contentsOf: buf[0..<n])
        }
        let json = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any] ?? [:]
        return Request(method: method, path: path, json: json)
    }

    /// {"legs": ["A","B"]} -> bit 0 = B (the model's chain: the default, and the only one outside player mode), bit 1 = A (the real gear's)
    private func legsMask(_ json: [String: Any]) -> Int {
        guard let legs = json["legs"] as? [String] else { return 1 }
        var mask = 0
        if legs.contains("B") { mask |= 1 }
        if legs.contains("A") { mask |= 2 }
        return mask
    }

    private func route(method: String, path: String, jsonBody: [String: Any]) -> (String, [String: Any]) {
        switch (method, path) {
        case ("POST", "/open"):
            guard let device = jsonBody["device"] as? String,
                  let inChannel = jsonBody["in_channel"] as? Int,
                  let outChannel = jsonBody["out_channel"] as? Int else {
                return ("400 Bad Request", ["error": "missing device/in_channel/out_channel"])
            }
            let blocksize = jsonBody["blocksize"] as? Int ?? 0
            let gainDb = jsonBody["gain_db"] as? Double ?? 0.0
            let ceilingDb = jsonBody["ceiling_db"] as? Double ?? -1.0
            let extraOut = jsonBody["extra_out_channels"] as? [Int] ?? []
            do {
                try engine.open(deviceName: device, inChannel: inChannel, outChannel: outChannel, extraOutChannels: extraOut, blocksize: blocksize, gainDb: gainDb, ceilingDb: ceilingDb)
                return ("200 OK", ["ok": true, "blocksize": engine.grantedBlocksize, "sample_rate": engine.sampleRate])
            } catch {
                return ("400 Bad Request", ["ok": false, "error": "\(error)"])
            }
        case ("POST", "/open_dual"):
            guard let device = jsonBody["device"] as? String,
                  let guitarChannel = jsonBody["guitar_channel"] as? Int,
                  let returnChannel = jsonBody["return_channel"] as? Int,
                  let pedalOutChannel = jsonBody["pedal_out_channel"] as? Int,
                  let monitorOutChannels = jsonBody["monitor_out_channels"] as? [Int] else {
                return ("400 Bad Request", ["error": "missing device/guitar_channel/return_channel/pedal_out_channel/monitor_out_channels"])
            }
            let blocksize = jsonBody["blocksize"] as? Int ?? 0
            let guitarGainDb = jsonBody["guitar_gain_db"] as? Double ?? 0.0
            let ceilingDb = jsonBody["ceiling_db"] as? Double ?? -1.0
            do {
                try engine.openDual(deviceName: device, guitarChannel: guitarChannel, returnChannel: returnChannel,
                                     pedalOutChannel: pedalOutChannel, monitorOutChannels: monitorOutChannels,
                                     blocksize: blocksize, guitarGainDb: guitarGainDb, ceilingDb: ceilingDb)
                return ("200 OK", ["ok": true, "blocksize": engine.grantedBlocksize, "sample_rate": engine.sampleRate])
            } catch {
                return ("400 Bad Request", ["ok": false, "error": "\(error)"])
            }
        case ("POST", "/open_player"):
            guard let device = jsonBody["device"] as? String,
                  let guitarChannel = jsonBody["guitar_channel"] as? Int,
                  let returnChannel = jsonBody["return_channel"] as? Int,
                  let reampOutChannel = jsonBody["reamp_out_channel"] as? Int,
                  let monitorOutChannels = jsonBody["monitor_out_channels"] as? [Int] else {
                return ("400 Bad Request", ["error": "missing device/guitar_channel/return_channel/reamp_out_channel/monitor_out_channels"])
            }
            let blocksize = jsonBody["blocksize"] as? Int ?? 0
            let feedGainDb = jsonBody["feed_gain_db"] as? Double ?? 0.0
            let ceilingDb = jsonBody["ceiling_db"] as? Double ?? -1.0
            do {
                try engine.openPlayer(deviceName: device, guitarChannel: guitarChannel, returnChannel: returnChannel,
                                       reampOutChannel: reampOutChannel, monitorOutChannels: monitorOutChannels,
                                       blocksize: blocksize, feedGainDb: feedGainDb, ceilingDb: ceilingDb)
                return ("200 OK", ["ok": true, "blocksize": engine.grantedBlocksize, "sample_rate": engine.sampleRate])
            } catch {
                return ("400 Bad Request", ["ok": false, "error": "\(error)"])
            }
        case ("POST", "/leg"):
            guard let leg = jsonBody["leg"] as? String else { return ("400 Bad Request", ["error": "missing leg (A or B)"]) }
            guard leg == "A" || leg == "B" else { return ("400 Bad Request", ["error": "leg must be A or B"]) }
            engine.selectLeg(leg == "B" ? 1 : 0)
            return ("200 OK", ["ok": true, "leg": leg])
        case ("POST", "/amp_gain"):
            guard let gainDb = jsonBody["gain_db"] as? Double else { return ("400 Bad Request", ["error": "missing gain_db"]) }
            engine.setAmpGainDb(gainDb)
            return ("200 OK", ["ok": true, "gain_db": gainDb])
        case ("POST", "/b_trim"):
            guard let trimDb = jsonBody["trim_db"] as? Double else { return ("400 Bad Request", ["error": "missing trim_db"]) }
            engine.setBTrimDb(trimDb)
            return ("200 OK", ["ok": true, "trim_db": trimDb])
        case ("POST", "/b_delay"):
            guard let samples = jsonBody["samples"] as? Int else { return ("400 Bad Request", ["error": "missing samples"]) }
            engine.setBDelay(samples)
            return ("200 OK", ["ok": true, "samples": samples])
        case ("POST", "/gain"):
            guard let gainDb = jsonBody["gain_db"] as? Double else { return ("400 Bad Request", ["error": "missing gain_db"]) }
            engine.setGainDb(gainDb)
            return ("200 OK", ["ok": true, "gain_db": gainDb])
        case ("POST", "/pregain"):
            guard let gainDb = jsonBody["pre_gain_db"] as? Double else { return ("400 Bad Request", ["error": "missing pre_gain_db"]) }
            engine.setPreGainDb(gainDb)
            return ("200 OK", ["ok": true, "pre_gain_db": gainDb])
        case ("POST", "/guitar_gain"):
            guard let gainDb = jsonBody["gain_db"] as? Double else { return ("400 Bad Request", ["error": "missing gain_db"]) }
            engine.setGuitarGainDb(gainDb)
            return ("200 OK", ["ok": true, "gain_db": gainDb])
        case ("POST", "/gate"):
            guard let thresholdDb = jsonBody["threshold_db"] as? Double else { return ("400 Bad Request", ["error": "missing threshold_db"]) }
            engine.setGateThresholdDb(thresholdDb)
            return ("200 OK", ["ok": true, "threshold_db": thresholdDb])
        case ("POST", "/ir_gain"):
            if let v = jsonBody["in_db"] as? Double { engine.setIrInGainDb(v) }
            if let v = jsonBody["out_db"] as? Double { engine.setIrOutGainDb(v) }
            return ("200 OK", ["ok": true])
        case ("POST", "/eq"):
            if let v = jsonBody["bass_db"] as? Double { engine.setEqBassDb(v) }
            if let v = jsonBody["mid_db"] as? Double { engine.setEqMidDb(v) }
            if let v = jsonBody["treble_db"] as? Double { engine.setEqTrebleDb(v) }
            return ("200 OK", ["ok": true])
        case ("POST", "/model"):
            guard let path = jsonBody["path"] as? String else { return ("400 Bad Request", ["error": "missing path"]) }
            do {
                try engine.loadModel(path: path, legs: legsMask(jsonBody))
                let s = engine.state()
                return ("200 OK", ["ok": true, "input_level_dbu": s.modelInputLevelDbu as Any, "output_level_dbu": s.modelOutputLevelDbu as Any,
                                   "sizes": engine.modelSizes()])
            } catch {
                return ("400 Bad Request", ["ok": false, "error": "\(error)"])
            }
        case ("POST", "/model/unload"):
            engine.unloadModel(legs: legsMask(jsonBody))
            return ("200 OK", ["ok": true])
        case ("POST", "/pedal_model"):
            guard let path = jsonBody["path"] as? String else { return ("400 Bad Request", ["error": "missing path"]) }
            do {
                try engine.loadPedal(path: path)
                let l = engine.pedalLevels()
                return ("200 OK", ["ok": true, "input_level_dbu": l.inputDbu as Any, "output_level_dbu": l.outputDbu as Any, "sizes": engine.pedalSizes()])
            } catch {
                return ("400 Bad Request", ["ok": false, "error": "\(error)"])
            }
        case ("POST", "/pedal_model/unload"):
            engine.unloadPedal()
            return ("200 OK", ["ok": true])
        case ("POST", "/pedal_controls"):
            if let pre = jsonBody["pre_db"] as? Double { engine.setPedalPreGainDb(pre) }
            if let post = jsonBody["post_db"] as? Double { engine.setPedalGainDb(post) }
            var size = -1
            if let index = jsonBody["size"] as? Int { size = engine.setPedalSize(index) }
            return ("200 OK", ["ok": true, "size": size, "sizes": engine.pedalSizes()])
        case ("POST", "/amp_pregain"):
            guard let gainDb = jsonBody["gain_db"] as? Double else { return ("400 Bad Request", ["error": "missing gain_db"]) }
            engine.setAmpPreGainDb(gainDb)
            return ("200 OK", ["ok": true, "gain_db": gainDb])
        case ("POST", "/model_size"):
            guard let index = jsonBody["index"] as? Int else { return ("400 Bad Request", ["error": "missing index"]) }
            let inUse = engine.setModelSize(index)
            return ("200 OK", ["ok": true, "index": inUse, "sizes": engine.modelSizes()])
        case ("POST", "/ir"):
            guard let path = jsonBody["path"] as? String else { return ("400 Bad Request", ["error": "missing path"]) }
            do {
                try engine.loadIR(path: path, legs: legsMask(jsonBody))
                return ("200 OK", ["ok": true])
            } catch {
                return ("400 Bad Request", ["ok": false, "error": "\(error)"])
            }
        case ("POST", "/ir/unload"):
            engine.unloadIR(legs: legsMask(jsonBody))
            return ("200 OK", ["ok": true])
        case ("POST", "/stop"):
            engine.close()
            return ("200 OK", ["ok": true])
        case ("GET", "/state"):
            let s = engine.state()
            var reply: [String: Any] = ["opened": engine.isOpen, "in_peak_dbfs": s.inPeakDb, "out_peak_dbfs": s.outPeakDb,
                               "xruns": s.xruns, "blocksize": engine.grantedBlocksize, "sample_rate": engine.sampleRate,
                               "model_loaded": s.modelLoaded, "ir_loaded": s.irLoaded,
                               "model_input_level_dbu": s.modelInputLevelDbu as Any, "model_output_level_dbu": s.modelOutputLevelDbu as Any,
                               "guitar_in_peak_dbfs": s.guitarInPeakDb as Any, "guitar_out_peak_dbfs": s.guitarOutPeakDb as Any]
            let note = engine.tuner()
            reply["tuner_hz"] = note?.hz as Any
            reply["tuner_confidence"] = note?.confidence as Any
            if let p = engine.playerState() {
                reply["player"] = true
                reply["leg"] = p.leg == 0 ? "A" : "B"
                reply["ir_a_loaded"] = p.irALoaded
                reply["a_peak_dbfs"] = p.aPeakDb
                reply["b_peak_dbfs"] = p.bPeakDb
                reply["pedal_loaded"] = engine.pedalLevels().loaded
            } else {
                reply["player"] = false
            }
            return ("200 OK", reply)
        default:
            return ("404 Not Found", ["error": "unknown route"])
        }
    }
}
