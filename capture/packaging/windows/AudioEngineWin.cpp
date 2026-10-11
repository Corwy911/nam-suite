// AudioEngineWin: ASIO-backed implementation of AudioEngineWin.h (see that file for the public contract, mirrored
// from packaging/AudioEngine.swift). Uses the Steinberg ASIO SDK directly, not a wrapper library (PortAudio etc.),
// so the buffer-switch callback can call straight into RTEngine (native/engine/RTEngine.h) with the least possible
// latency, the same reasoning as AudioEngine.swift talking to CoreAudio's AUHAL directly.
//
// *** BUILD STATUS: verified against real hardware (Focusrite Scarlett 18i20, 3rd gen) via
// packaging/windows/build_controlserver_asio.bat - open/state/stop all confirmed working over a real ASIO duplex
// stream. Three real bugs were found and fixed this way (not by inspection): ASIOCallbacks must be a persistent
// member, not a stack-local passed to ASIOCreateBuffers (some drivers keep the pointer, not a copy); ASIOExit()
// reaches for the SDK's own global `asioDrivers` pointer, so loading a driver through anything else leaves it null
// and ASIOExit() null-derefs (see Impl's own comment); every ASIO/COM call must happen on the SAME thread for the
// object's whole lifetime (see ControlServerWin.cpp's run_loop comment) - a fresh thread per HTTP connection, fine
// on Mac's CoreAudio, crashes here on a cross-apartment COM call. NAM_HAVE_ASIO_SDK undefined (SDK not present)
// still compiles as a harmless stub that fails open()/openDual() with a clear error instead of touching hardware.
//
// ASIO quirk that shapes this whole file: the C API (ASIOInit/ASIOCreateBuffers/...) and the bufferSwitch callback
// have NO per-instance user-data parameter - a process can only ever have one ASIO driver open at a time, and the
// callback is a bare function pointer. g_active is the one instance whose buffers are live; exactly the same
// constraint the manual already documents for the Python side ("while the native engine is listening, the
// interface is no longer open for Python").
#include "AudioEngineWin.h"

#include "WavRead.h"
#include "../../native/engine/RTEngine.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace {
// A GUI-subsystem app (NamCaptureApp.exe) has no console: fprintf(stderr, ...) goes nowhere, which is why earlier
// diagnostics here only ever worked in the standalone console test harness. This appends to the same per-user data
// folder capture_tool/app.py and PythonSidecar.cpp already log to, so real-app issues (like the buffer-size
// investigation this was added for) can be diagnosed without rebuilding a separate harness each time.
void log_line(const char* fmt, ...) {
    wchar_t local_app_data[MAX_PATH]{};
    DWORD len = GetEnvironmentVariableW(L"LOCALAPPDATA", local_app_data, MAX_PATH);
    if (len == 0) return;
    std::wstring dir = std::wstring(local_app_data) + L"\\NAM Capture";
    CreateDirectoryW(dir.c_str(), nullptr);
    std::wstring path = dir + L"\\native-engine.log";
    FILE* f = nullptr;
    if (_wfopen_s(&f, path.c_str(), L"a") != 0 || !f) return;
    va_list args;
    va_start(args, fmt);
    vfprintf(f, fmt, args);
    va_end(args);
    fclose(f);
}
}  // namespace

#ifdef NAM_HAVE_ASIO_SDK
// Paths match how native/third_party/ASIO_SDK_SETUP.md asks the SDK to be dropped in:
//   native/third_party/asio_sdk/common/{asio.h,asiosys.h,asiodrivers.h,iasiodrv.h,...}
//   native/third_party/asio_sdk/host/{asiodrivers.cpp,...}
//   native/third_party/asio_sdk/host/pc/{asiolist.cpp,...}
// asiosys.h MUST come before asio.h (asio.h relies on it having already set WINDOWS/IEEE754_64FLOAT/etc - it
// doesn't include it itself, exactly like the SDK's own common/asio.cpp does it).
#include <asiosys.h>
#include <asio.h>
#include <asiodrivers.h>
// Declared (not in any SDK header - only forward-declared locally inside host/asiodrivers.cpp itself, right above
// its own definition) as the free function that owns the SDK's global `asioDrivers` pointer: ASIOExit() reaches
// for that same global to release the driver, so loading through anything else leaves it null and ASIOExit()
// null-derefs - see AudioEngineWin::Impl's own comment on why this class has no AsioDrivers member of its own.
bool loadAsioDriver(char* name);
#endif

namespace {

#ifdef NAM_HAVE_ASIO_SDK

// The single live instance (see file header) - the trampolines below dispatch through it.
AudioEngineWin::Impl* g_active = nullptr;

// One buffer slot per physical channel this engine touches, in a fixed order: for open() it is
// [input] [output_0] [output_1] ...; for open_dual() it is [guitar_in] [return_in] [pedal_out] [monitor_0] [monitor_1?].
// ASIOCreateBuffers wants them all in one array regardless of direction.
constexpr int kMaxBufferInfos = 5;

float asio_sample_to_float(const void* buf, long frame, ASIOSampleType type) {
    switch (type) {
        case ASIOSTInt16LSB: return reinterpret_cast<const int16_t*>(buf)[frame] / 32768.0f;
        case ASIOSTInt32LSB: return reinterpret_cast<const int32_t*>(buf)[frame] / 2147483648.0f;
        case ASIOSTFloat32LSB: return reinterpret_cast<const float*>(buf)[frame];
        case ASIOSTFloat64LSB: return static_cast<float>(reinterpret_cast<const double*>(buf)[frame]);
        case ASIOSTInt24LSB: {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(buf) + frame * 3;
            int32_t v = (p[0]) | (p[1] << 8) | (p[2] << 16);
            if (v & 0x800000) v |= static_cast<int32_t>(0xFF000000);
            return v / 8388608.0f;
        }
        // ASIOSTInt32LSB16/18/20/24: 32-bit container, fewer significant bits - still read as a plain int32 sample,
        // just with the low bits unused; treating it as ASIOSTInt32LSB is the documented-safe fallback.
        default: return reinterpret_cast<const int32_t*>(buf)[frame] / 2147483648.0f;
    }
}

void float_to_asio_sample(float v, void* buf, long frame, ASIOSampleType type) {
    switch (type) {
        case ASIOSTInt16LSB: reinterpret_cast<int16_t*>(buf)[frame] = static_cast<int16_t>(std::clamp(v, -1.0f, 1.0f) * 32767.0f); return;
        case ASIOSTFloat32LSB: reinterpret_cast<float*>(buf)[frame] = v; return;
        case ASIOSTFloat64LSB: reinterpret_cast<double*>(buf)[frame] = v; return;
        case ASIOSTInt24LSB: {
            int32_t iv = static_cast<int32_t>(std::clamp(v, -1.0f, 1.0f) * 8388607.0f);
            uint8_t* p = reinterpret_cast<uint8_t*>(buf) + frame * 3;
            p[0] = iv & 0xFF; p[1] = (iv >> 8) & 0xFF; p[2] = (iv >> 16) & 0xFF;
            return;
        }
        case ASIOSTInt32LSB:
        default: reinterpret_cast<int32_t*>(buf)[frame] = static_cast<int32_t>(std::clamp(v, -1.0f, 1.0f) * 2147483647.0f); return;
    }
}

#endif  // NAM_HAVE_ASIO_SDK

}  // namespace

struct AudioEngineWin::Impl {
#ifdef NAM_HAVE_ASIO_SDK
    // No AsioDrivers member here on purpose: ASIOExit() (common/asio.cpp) reaches for the SDK's own GLOBAL
    // `asioDrivers` pointer (host/asiodrivers.cpp) to release the driver, not whatever instance loaded it - found
    // by a real crash (null-pointer call inside ASIOExit) when this used to keep a separate instance here instead
    // of going through the free loadAsioDriver()/asioDrivers global the rest of the SDK's glue code expects.
    ASIOBufferInfo buffer_infos[kMaxBufferInfos]{};
    ASIOChannelInfo channel_infos[kMaxBufferInfos]{};
    // Persistent, NOT a local in open()/open_dual(): some drivers keep the pointer passed to ASIOCreateBuffers
    // instead of copying it, so a stack-local ASIOCallbacks (as this used to be) becomes a dangling pointer the
    // instant open() returns - the driver's real-time callback thread then calls through garbage. Found by an
    // actual crash (address 0x2) on real hardware, not by inspection.
    ASIOCallbacks callbacks{};
    long n_buffer_infos = 0;
    long buffer_size = 0;

    bool dual_mode = false;
    RTEngine* engine = nullptr;         // single mode: the only chain. dual mode: return -> [NAM] -> [IR] -> monitor
    RTEngine* guitar_engine = nullptr;  // dual mode only: guitar -> pedal feed, unity gain, no NAM/IR
    RTPlayer* player = nullptr;         // player mode (a dual-mode layout whose chains are RTPlayer's): when set, `engine` and
                                        // `guitar_engine` are NOT owned, they are the player's leg B and feed engines

    int single_out_count = 1;           // single mode: how many of buffer_infos[1..] carry the same output (extra_out_channels)
    int monitor_channel_count = 1;      // dual mode: 1 or 2

    std::vector<float> scratch_in, scratch_out;             // single mode
    std::vector<float> scratch_guitar, scratch_return, scratch_pedal_out, scratch_monitor_out;  // dual mode

    std::atomic<float> in_peak{0.0f};
    std::atomic<float> guitar_in_peak{0.0f};
    std::atomic<int> xrun_count{0};

    void buffer_switch(long double_buffer_index) {
        const long n = buffer_size;
        if (!dual_mode) {
            if (static_cast<long>(scratch_in.size()) < n) { scratch_in.resize(n); scratch_out.resize(n); }
            for (long i = 0; i < n; ++i)
                scratch_in[i] = asio_sample_to_float(buffer_infos[0].buffers[double_buffer_index], i, channel_infos[0].type);
            float peak = 0.0f;
            for (long i = 0; i < n; ++i) peak = std::max(peak, std::fabs(scratch_in[i]));
            in_peak.store(peak, std::memory_order_relaxed);

            rt_process(engine, scratch_in.data(), scratch_out.data(), static_cast<int>(n));

            for (int ch = 0; ch < single_out_count; ++ch) {
                ASIOBufferInfo& bi = buffer_infos[1 + ch];
                ASIOSampleType type = channel_infos[1 + ch].type;
                for (long i = 0; i < n; ++i) float_to_asio_sample(scratch_out[i], bi.buffers[double_buffer_index], i, type);
            }
        } else {
            if (static_cast<long>(scratch_guitar.size()) < n) {
                scratch_guitar.resize(n); scratch_return.resize(n); scratch_pedal_out.resize(n); scratch_monitor_out.resize(n);
            }
            for (long i = 0; i < n; ++i) {
                scratch_guitar[i] = asio_sample_to_float(buffer_infos[0].buffers[double_buffer_index], i, channel_infos[0].type);
                scratch_return[i] = asio_sample_to_float(buffer_infos[1].buffers[double_buffer_index], i, channel_infos[1].type);
            }
            float gp = 0.0f, rp = 0.0f;
            for (long i = 0; i < n; ++i) { gp = std::max(gp, std::fabs(scratch_guitar[i])); rp = std::max(rp, std::fabs(scratch_return[i])); }
            guitar_in_peak.store(gp, std::memory_order_relaxed);
            in_peak.store(rp, std::memory_order_relaxed);   // /state's in_peak_dbfs keeps meaning "the return", same as AudioEngine.swift

            if (player) {
                rt_player_process(player, scratch_guitar.data(), scratch_return.data(), scratch_pedal_out.data(),
                                  scratch_monitor_out.data(), static_cast<int>(n));
            } else {
                rt_process(guitar_engine, scratch_guitar.data(), scratch_pedal_out.data(), static_cast<int>(n));
                rt_process(engine, scratch_return.data(), scratch_monitor_out.data(), static_cast<int>(n));
            }

            {
                ASIOBufferInfo& bi = buffer_infos[2];
                for (long i = 0; i < n; ++i) float_to_asio_sample(scratch_pedal_out[i], bi.buffers[double_buffer_index], i, channel_infos[2].type);
            }
            for (int ch = 0; ch < monitor_channel_count; ++ch) {
                ASIOBufferInfo& bi = buffer_infos[3 + ch];
                for (long i = 0; i < n; ++i) float_to_asio_sample(scratch_monitor_out[i], bi.buffers[double_buffer_index], i, channel_infos[3 + ch].type);
            }
        }
    }
#endif  // NAM_HAVE_ASIO_SDK
};

#ifdef NAM_HAVE_ASIO_SDK
namespace {
void asio_buffer_switch(long double_buffer_index, ASIOBool /*direct_process*/) {
    if (g_active) g_active->buffer_switch(double_buffer_index);
}
ASIOTime* asio_buffer_switch_time_info(ASIOTime* params, long double_buffer_index, ASIOBool direct_process) {
    asio_buffer_switch(double_buffer_index, direct_process);
    return params;
}
void asio_sample_rate_did_change(ASIOSampleRate) {}
long asio_message(long selector, long value, void*, double*) {
    // Minimal, conservative host per Steinberg's own "Hosting Application Techniques": answer only what we
    // actually support, refuse resets (this app is short-lived and reopened on demand instead) rather than
    // trying to renegotiate live.
    switch (selector) {
        case kAsioSelectorSupported:
            switch (value) {
                case kAsioResetRequest:
                case kAsioEngineVersion:
                case kAsioResyncRequest:
                case kAsioLatenciesChanged:
                case kAsioSupportsTimeInfo:
                case kAsioSupportsTimeCode:
                    return 1L;
                default: return 0L;
            }
        case kAsioEngineVersion: return 2L;
        case kAsioResetRequest: return 0L;   // acknowledged, ignored: see comment above
        case kAsioSupportsTimeInfo: return 0L;   // we use the plain bufferSwitch callback, not TimeInfo
        default: return 0L;
    }
}
}  // namespace
#endif

AudioEngineWin::AudioEngineWin() : impl_(new Impl()) {}

AudioEngineWin::~AudioEngineWin() {
    close();
    delete impl_;
}

std::vector<std::string> AudioEngineWin::list_driver_names() {
#ifdef NAM_HAVE_ASIO_SDK
    AsioDrivers drivers;
    char names_buf[32][32];
    char* names[32];
    for (int i = 0; i < 32; ++i) names[i] = names_buf[i];
    long count = drivers.getDriverNames(names, 32);
    std::vector<std::string> out;
    for (long i = 0; i < count; ++i) out.emplace_back(names[i]);
    return out;
#else
    return {};
#endif
}

#ifndef NAM_HAVE_ASIO_SDK

std::string AudioEngineWin::open(const std::string&, int, int, const std::vector<int>&, int, double, double) {
    return "ASIO SDK non disponibile in questa build (vedi native/third_party/ASIO_SDK_SETUP.md)";
}
std::string AudioEngineWin::open_dual(const std::string&, int, int, int, const std::vector<int>&, int, double, double) {
    return "ASIO SDK non disponibile in questa build (vedi native/third_party/ASIO_SDK_SETUP.md)";
}
std::string AudioEngineWin::open_player(const std::string&, int, int, int, const std::vector<int>&, int, double, double) {
    return "ASIO SDK non disponibile in questa build (vedi native/third_party/ASIO_SDK_SETUP.md)";
}
void AudioEngineWin::close() {}
void AudioEngineWin::select_leg(int) {}
void AudioEngineWin::set_amp_gain_db(double) {}
void AudioEngineWin::set_b_trim_db(double) {}
void AudioEngineWin::set_b_delay(int) {}
void AudioEngineWin::set_gain_db(double) {}
void AudioEngineWin::set_pre_gain_db(double) {}
void AudioEngineWin::set_guitar_gain_db(double) {}
void AudioEngineWin::set_gate_threshold_db(double) {}
void AudioEngineWin::set_eq_bass_db(double) {}
void AudioEngineWin::set_eq_mid_db(double) {}
void AudioEngineWin::set_eq_treble_db(double) {}
void AudioEngineWin::set_ir_in_gain_db(double) {}
void AudioEngineWin::set_ir_out_gain_db(double) {}
std::string AudioEngineWin::load_model(const std::string&, int) { return "interfaccia non aperta"; }
void AudioEngineWin::unload_model(int) {}
std::string AudioEngineWin::load_pedal(const std::string&) { return "interfaccia non aperta"; }
void AudioEngineWin::unload_pedal() {}
int AudioEngineWin::pedal_sizes() { return 0; }
int AudioEngineWin::set_pedal_size(int) { return -1; }
void AudioEngineWin::set_pedal_pre_gain_db(double) {}
void AudioEngineWin::set_pedal_gain_db(double) {}
void AudioEngineWin::set_amp_pre_gain_db(double) {}
int AudioEngineWin::model_sizes() { return 0; }
int AudioEngineWin::set_model_size(int) { return -1; }
std::string AudioEngineWin::load_ir(const std::string&, int) { return "interfaccia non aperta"; }
void AudioEngineWin::unload_ir(int) {}
AudioEngineWin::State AudioEngineWin::state() const { return State{}; }

#else  // NAM_HAVE_ASIO_SDK

namespace {
std::string open_common(AudioEngineWin::Impl* impl, const std::string& device_name, long blocksize_request,
                         double* out_sample_rate, long* out_buffer_size) {
    if (!loadAsioDriver(const_cast<char*>(device_name.c_str())))
        return "driver ASIO non trovato o non caricabile: " + device_name;

    ASIODriverInfo driver_info{};
    driver_info.asioVersion = 2;
    if (ASIOInit(&driver_info) != ASE_OK) return "ASIOInit fallito per: " + device_name;

    ASIOSampleRate rate = 0.0;
    ASIOGetSampleRate(&rate);
    if (rate <= 0.0) { rate = 48000.0; ASIOSetSampleRate(rate); }
    if (std::fabs(rate - 48000.0) > 0.5)
        return "la scheda gira a " + std::to_string(static_cast<long>(rate)) + " Hz, serve 48000 Hz (imposta il pannello del driver)";

    long min_size = 0, max_size = 0, preferred_size = 0, granularity = 0;
    ASIOGetBufferSize(&min_size, &max_size, &preferred_size, &granularity);
    // Found by testing, not by inspection: re-opening the device right after closing it (exactly what happens when
    // Python's own session hands the interface to this engine for "Prova dal vivo") can catch this Focusrite driver
    // mid-settle, where ASIOGetBufferSize answers with a transient, wrong, degenerate range (min==max, e.g. stuck at
    // whatever the previous session's buffer happened to be) instead of its real one - a log showed min=max=64 on
    // one open in a row of five identical requests, min=16/max=1024 on the other four. A min==max reading this
    // narrow is never legitimate for this device (its real range is always the wide one), so it is treated as a
    // signal to wait for the driver to settle and ask again, rather than trusted and silently clamped to it.
    for (int retry = 0; retry < 5 && min_size > 0 && min_size == max_size; ++retry) {
        Sleep(100);
        ASIOGetBufferSize(&min_size, &max_size, &preferred_size, &granularity);
    }
    long buffer_size = (blocksize_request > 0) ? std::clamp(blocksize_request, min_size, max_size) : preferred_size;
    if (granularity > 0 && buffer_size > min_size) {
        // Sizes between min and max are only valid at min + n*granularity - round to the nearest valid one instead
        // of silently handing the driver something it may reject or reinterpret on its own (see the "requested 8,
        // Focusrite Control says 320" investigation this comment was added for).
        long steps = (buffer_size - min_size + granularity / 2) / granularity;
        buffer_size = min_size + steps * granularity;
    } else if (granularity < 0 && buffer_size > min_size) {
        // granularity == -1 (or any negative value) is ASIO's own convention for "powers of two between min and max"
        long p = min_size;
        while (p * 2 <= buffer_size && p * 2 <= max_size) p *= 2;
        buffer_size = p;
    }
    log_line("[open_common] device=%s richiesto=%ld min=%ld max=%ld preferred=%ld granularity=%ld -> uso=%ld\n",
             device_name.c_str(), blocksize_request, min_size, max_size, preferred_size, granularity, buffer_size);

    *out_sample_rate = rate;
    *out_buffer_size = buffer_size;
    return "";
}

void teardown(AudioEngineWin::Impl* impl) {
    if (g_active == impl) g_active = nullptr;
    ASIOStop();
    // ASIOStop() is not guaranteed by every driver to block until a callback already in flight on the driver's own
    // real-time thread has actually returned. A short sleep, several times longer than one buffer period, gives
    // that callback room to finish before ASIODisposeBuffers()/rt_destroy() free what it might still be touching.
    Sleep(300);
    ASIODisposeBuffers();
    ASIOExit();   // releases the driver itself via the SDK's own global asioDrivers pointer - see Impl's comment
    if (impl->player) {                 // owns the three engines behind `engine`/`guitar_engine`
        rt_player_destroy(impl->player);
        impl->player = nullptr;
        impl->engine = nullptr;
        impl->guitar_engine = nullptr;
    }
    if (impl->engine) { rt_destroy(impl->engine); impl->engine = nullptr; }
    if (impl->guitar_engine) { rt_destroy(impl->guitar_engine); impl->guitar_engine = nullptr; }
    impl->n_buffer_infos = 0;
    impl->dual_mode = false;
}
}  // namespace

std::string AudioEngineWin::open(const std::string& device_name, int in_channel, int out_channel,
                                  const std::vector<int>& extra_out_channels, int blocksize, double gain_db,
                                  double ceiling_db) {
    close();
    log_line("[open] in_channel=%d out_channel=%d blocksize_richiesto=%d\n", in_channel, out_channel, blocksize);
    double rate = 0.0;
    long buffer_size = 0;
    std::string err = open_common(impl_, device_name, blocksize, &rate, &buffer_size);
    if (!err.empty()) { teardown(impl_); return err; }

    std::vector<int> all_out{out_channel};
    all_out.insert(all_out.end(), extra_out_channels.begin(), extra_out_channels.end());
    if (1 + static_cast<long>(all_out.size()) > kMaxBufferInfos)
        return "troppi canali di uscita per il monitor (" + std::to_string(all_out.size()) + ")";

    impl_->buffer_infos[0] = {ASIOTrue, in_channel, {nullptr, nullptr}};
    for (size_t i = 0; i < all_out.size(); ++i) impl_->buffer_infos[1 + i] = {ASIOFalse, all_out[i], {nullptr, nullptr}};
    impl_->n_buffer_infos = 1 + static_cast<long>(all_out.size());
    impl_->single_out_count = static_cast<int>(all_out.size());
    // buffer_switch() reads Impl::buffer_size (never the local above) - without this line it stays at its
    // default-initialized 0, so the sample-copy loops run zero iterations and every meter reads silence forever,
    // even though /open still reports the correct requested size (granted_blocksize_ below is a SEPARATE field).
    // Found by an actual "everything is silent, no crash, no error" report, not by inspection.
    impl_->buffer_size = buffer_size;

    impl_->callbacks = {&asio_buffer_switch, &asio_sample_rate_did_change, &asio_message, &asio_buffer_switch_time_info};
    if (ASIOCreateBuffers(impl_->buffer_infos, impl_->n_buffer_infos, buffer_size, &impl_->callbacks) != ASE_OK) {
        teardown(impl_);
        return "ASIOCreateBuffers fallito (buffer " + std::to_string(buffer_size) + ")";
    }
    for (long i = 0; i < impl_->n_buffer_infos; ++i) {
        impl_->channel_infos[i].channel = impl_->buffer_infos[i].channelNum;
        impl_->channel_infos[i].isInput = impl_->buffer_infos[i].isInput;
        ASIOGetChannelInfo(&impl_->channel_infos[i]);
    }

    RTEngine* e = rt_create(rate);
    if (!e) { teardown(impl_); return "rt_create fallito"; }
    rt_set_ceiling_db(e, ceiling_db);
    rt_set_gain_db(e, gain_db);
    impl_->engine = e;
    impl_->dual_mode = false;

    if (ASIOStart() != ASE_OK) { teardown(impl_); return "ASIOStart fallito"; }
    g_active = impl_;

    sample_rate_ = rate;
    granted_blocksize_ = static_cast<int>(buffer_size);
    is_open_ = true;
    return "";
}

std::string AudioEngineWin::open_dual(const std::string& device_name, int guitar_channel, int return_channel,
                                       int pedal_out_channel, const std::vector<int>& monitor_out_channels,
                                       int blocksize, double guitar_gain_db, double ceiling_db) {
    return open_dual_common(device_name, guitar_channel, return_channel, pedal_out_channel, monitor_out_channels, blocksize,
                            guitar_gain_db, ceiling_db, false);
}

std::string AudioEngineWin::open_player(const std::string& device_name, int guitar_channel, int return_channel,
                                         int reamp_out_channel, const std::vector<int>& monitor_out_channels, int blocksize,
                                         double feed_gain_db, double ceiling_db) {
    return open_dual_common(device_name, guitar_channel, return_channel, reamp_out_channel, monitor_out_channels, blocksize,
                            feed_gain_db, ceiling_db, true);
}

std::string AudioEngineWin::open_dual_common(const std::string& device_name, int guitar_channel, int return_channel,
                                              int pedal_out_channel, const std::vector<int>& monitor_out_channels,
                                              int blocksize, double guitar_gain_db, double ceiling_db, bool player) {
    close();
    log_line("[%s] guitar_channel=%d return_channel=%d pedal_out_channel=%d blocksize_richiesto=%d\n",
             player ? "open_player" : "open_dual", guitar_channel, return_channel, pedal_out_channel, blocksize);
    if (monitor_out_channels.size() != 1 && monitor_out_channels.size() != 2)
        return "monitor_out_channels deve avere 1 o 2 canali";

    double rate = 0.0;
    long buffer_size = 0;
    std::string err = open_common(impl_, device_name, blocksize, &rate, &buffer_size);
    if (!err.empty()) { teardown(impl_); return err; }

    impl_->buffer_infos[0] = {ASIOTrue, guitar_channel, {nullptr, nullptr}};
    impl_->buffer_infos[1] = {ASIOTrue, return_channel, {nullptr, nullptr}};
    impl_->buffer_infos[2] = {ASIOFalse, pedal_out_channel, {nullptr, nullptr}};
    for (size_t i = 0; i < monitor_out_channels.size(); ++i)
        impl_->buffer_infos[3 + i] = {ASIOFalse, monitor_out_channels[i], {nullptr, nullptr}};
    impl_->n_buffer_infos = 3 + static_cast<long>(monitor_out_channels.size());
    impl_->buffer_size = buffer_size;   // see open()'s identical line for why this is required
    impl_->monitor_channel_count = static_cast<int>(monitor_out_channels.size());

    impl_->callbacks = {&asio_buffer_switch, &asio_sample_rate_did_change, &asio_message, &asio_buffer_switch_time_info};
    if (ASIOCreateBuffers(impl_->buffer_infos, impl_->n_buffer_infos, buffer_size, &impl_->callbacks) != ASE_OK) {
        teardown(impl_);
        return "ASIOCreateBuffers fallito (buffer " + std::to_string(buffer_size) + ")";
    }
    for (long i = 0; i < impl_->n_buffer_infos; ++i) {
        impl_->channel_infos[i].channel = impl_->buffer_infos[i].channelNum;
        impl_->channel_infos[i].isInput = impl_->buffer_infos[i].isInput;
        ASIOGetChannelInfo(&impl_->channel_infos[i]);
    }

    if (player) {
        RTPlayer* p = rt_player_create(rate);
        if (!p) { teardown(impl_); return "rt_player_create fallito"; }
        for (RTPlayerPart part : {RT_LEG_FEED, RT_LEG_A, RT_LEG_B}) rt_set_ceiling_db(rt_player_engine(p, part), ceiling_db);
        rt_set_gain_db(rt_player_engine(p, RT_LEG_FEED), guitar_gain_db);
        impl_->player = p;
        impl_->guitar_engine = rt_player_engine(p, RT_LEG_FEED);   // not owned: the player owns them (see Impl::player)
        impl_->engine = rt_player_engine(p, RT_LEG_B);
    } else {
        RTEngine* guitar_engine = rt_create(rate);
        RTEngine* live_engine = rt_create(rate);
        if (!guitar_engine || !live_engine) { teardown(impl_); return "rt_create fallito"; }
        rt_set_ceiling_db(guitar_engine, ceiling_db);
        rt_set_gain_db(guitar_engine, guitar_gain_db);
        rt_set_ceiling_db(live_engine, ceiling_db);
        impl_->guitar_engine = guitar_engine;
        impl_->engine = live_engine;
    }
    impl_->dual_mode = true;

    if (ASIOStart() != ASE_OK) { teardown(impl_); return "ASIOStart fallito"; }
    g_active = impl_;

    sample_rate_ = rate;
    granted_blocksize_ = static_cast<int>(buffer_size);
    is_open_ = true;
    // what is really running, for "I hear nothing" reports (0-based channels: the page's menus show them plus one)
    std::string monitors = std::to_string(monitor_out_channels[0]);
    if (monitor_out_channels.size() > 1) monitors += "+" + std::to_string(monitor_out_channels[1]);
    log_line("[%s] running: rate=%.0f buffer=%ld in(guitar)=%d in(return)=%d out(send)=%d out(monitor)=%s\n", player ? "open_player" : "open_dual", rate,
             buffer_size, guitar_channel, return_channel, pedal_out_channel, monitors.c_str());
    return "";
}

void AudioEngineWin::close() {
    if (!is_open_) return;
    // Fade out first, same "never a click" contract as AudioEngine.swift's close(): the buffer-switch callback
    // keeps running (on the driver's own thread) while we wait here on the control thread.
    if (impl_->player) {
        rt_player_stop(impl_->player);
        for (int tries = 0; rt_player_is_finished(impl_->player) == 0 && tries < 50; ++tries) Sleep(10);
    } else if (impl_->engine) {
        rt_stop(impl_->engine);
        for (int tries = 0; rt_is_finished(impl_->engine) == 0 && tries < 50; ++tries) Sleep(10);
    }
    if (!impl_->player && impl_->dual_mode && impl_->guitar_engine) {
        rt_stop(impl_->guitar_engine);
        for (int tries = 0; rt_is_finished(impl_->guitar_engine) == 0 && tries < 50; ++tries) Sleep(10);
    }
    teardown(impl_);
    is_open_ = false;
    granted_blocksize_ = 0;
}

void AudioEngineWin::set_gain_db(double db) { if (impl_->engine) rt_set_gain_db(impl_->engine, db); }
void AudioEngineWin::set_pre_gain_db(double db) { if (impl_->engine) rt_set_pre_gain_db(impl_->engine, db); }
void AudioEngineWin::set_guitar_gain_db(double db) { if (impl_->guitar_engine) rt_set_gain_db(impl_->guitar_engine, db); }
void AudioEngineWin::set_gate_threshold_db(double db) { if (impl_->engine) rt_set_gate_threshold_db(impl_->engine, db); }
void AudioEngineWin::set_eq_bass_db(double db) { if (impl_->engine) rt_set_eq_bass_db(impl_->engine, db); }
void AudioEngineWin::set_eq_mid_db(double db) { if (impl_->engine) rt_set_eq_mid_db(impl_->engine, db); }
void AudioEngineWin::set_eq_treble_db(double db) { if (impl_->engine) rt_set_eq_treble_db(impl_->engine, db); }
void AudioEngineWin::set_ir_in_gain_db(double db) { if (impl_->engine) rt_set_ir_in_gain_db(impl_->engine, db); }
void AudioEngineWin::set_ir_out_gain_db(double db) { if (impl_->engine) rt_set_ir_out_gain_db(impl_->engine, db); }
void AudioEngineWin::select_leg(int leg) { if (impl_->player) rt_player_select(impl_->player, leg); }
void AudioEngineWin::set_amp_gain_db(double db) { if (impl_->player) rt_set_gain_db(rt_player_engine(impl_->player, RT_LEG_A), db); }
void AudioEngineWin::set_b_trim_db(double db) { if (impl_->player) rt_player_set_b_trim_db(impl_->player, db); }
void AudioEngineWin::set_b_delay(int samples) { if (impl_->player) rt_player_set_b_delay(impl_->player, samples); }

std::string AudioEngineWin::load_model(const std::string& path, int legs) {
    if (!impl_->engine) return "interfaccia non aperta";
    if ((legs & 2) && !impl_->player) return "il modello sul ritorno esiste solo nel lettore NAM";
    RTEngine* leg_a = impl_->player ? rt_player_engine(impl_->player, RT_LEG_A) : nullptr;
    if ((legs & 1) && !rt_load_model(impl_->engine, path.c_str())) return rt_last_error(impl_->engine);
    if ((legs & 2) && !rt_load_model(leg_a, path.c_str())) return rt_last_error(leg_a);
    return "";
}
void AudioEngineWin::unload_model(int legs) {
    if ((legs & 1) && impl_->engine) rt_unload_model(impl_->engine);
    if ((legs & 2) && impl_->player) rt_unload_model(rt_player_engine(impl_->player, RT_LEG_A));
}
int AudioEngineWin::model_sizes() { return impl_->engine ? rt_model_sizes(impl_->engine) : 0; }
int AudioEngineWin::set_model_size(int index) {
    if (!impl_->engine) return -1;
    if (impl_->player) rt_set_model_size(rt_player_engine(impl_->player, RT_LEG_A), index);     // the amp's second copy (no-op when it has no model)
    return rt_set_model_size(impl_->engine, index);
}

std::string AudioEngineWin::load_pedal(const std::string& path) {
    RTEngine* pedal = impl_->player ? rt_player_engine(impl_->player, RT_LEG_PEDAL) : nullptr;
    if (!pedal) return "il test pedali esiste solo nel lettore NAM";
    if (!rt_load_model(pedal, path.c_str())) return rt_last_error(pedal);
    return "";
}
void AudioEngineWin::unload_pedal() { if (impl_->player) rt_unload_model(rt_player_engine(impl_->player, RT_LEG_PEDAL)); }
int AudioEngineWin::pedal_sizes() { return impl_->player ? rt_model_sizes(rt_player_engine(impl_->player, RT_LEG_PEDAL)) : 0; }
int AudioEngineWin::set_pedal_size(int index) { return impl_->player ? rt_set_model_size(rt_player_engine(impl_->player, RT_LEG_PEDAL), index) : -1; }
void AudioEngineWin::set_pedal_pre_gain_db(double db) { if (impl_->player) rt_set_pre_gain_db(rt_player_engine(impl_->player, RT_LEG_PEDAL), db); }
void AudioEngineWin::set_pedal_gain_db(double db) { if (impl_->player) rt_set_gain_db(rt_player_engine(impl_->player, RT_LEG_PEDAL), db); }
void AudioEngineWin::set_amp_pre_gain_db(double db) { if (impl_->player) rt_set_pre_gain_db(rt_player_engine(impl_->player, RT_LEG_A), db); }

std::string AudioEngineWin::load_ir(const std::string& path, int legs) {
    if (!impl_->engine) return "interfaccia non aperta";
    if ((legs & 2) && !impl_->player) return "la IR sul ritorno esiste solo nel lettore NAM";
    WavData wav;
    try {
        wav = read_wav_mono(path);
    } catch (const std::exception& e) {
        return e.what();
    }
    if (std::fabs(wav.sample_rate - sample_rate_) > 0.5)
        return "IR a " + std::to_string(static_cast<long>(wav.sample_rate)) + " Hz, la scheda gira a " +
               std::to_string(static_cast<long>(sample_rate_)) + " Hz: esporta la IR alla stessa frequenza, non viene ricampionata";
    RTEngine* leg_a = impl_->player ? rt_player_engine(impl_->player, RT_LEG_A) : nullptr;
    if ((legs & 1) && !rt_load_ir(impl_->engine, wav.mono.data(), static_cast<int>(wav.mono.size()))) return rt_last_error(impl_->engine);
    if ((legs & 2) && !rt_load_ir(leg_a, wav.mono.data(), static_cast<int>(wav.mono.size()))) return rt_last_error(leg_a);
    return "";
}
void AudioEngineWin::unload_ir(int legs) {
    if ((legs & 1) && impl_->engine) rt_unload_ir(impl_->engine);
    if ((legs & 2) && impl_->player) rt_unload_ir(rt_player_engine(impl_->player, RT_LEG_A));
}

AudioEngineWin::State AudioEngineWin::state() const {
    State s;
    if (impl_->engine) {
        s.out_peak_dbfs = rt_last_peak_db(impl_->engine);
        s.model_loaded = rt_model_loaded(impl_->engine) != 0;
        s.ir_loaded = rt_ir_loaded(impl_->engine) != 0;
        double in = rt_model_input_level_dbu(impl_->engine);
        double out = rt_model_output_level_dbu(impl_->engine);
        if (!std::isnan(in)) s.model_input_level_dbu = in;
        if (!std::isnan(out)) s.model_output_level_dbu = out;
    }
    float in_peak = impl_->in_peak.load(std::memory_order_relaxed);
    s.in_peak_dbfs = in_peak > 0.0f ? 20.0 * std::log10(static_cast<double>(in_peak)) : -240.0;
    s.xruns = impl_->xrun_count.load(std::memory_order_relaxed);
    if (impl_->dual_mode) {
        float g_in = impl_->guitar_in_peak.load(std::memory_order_relaxed);
        s.guitar_in_peak_dbfs = g_in > 0.0f ? 20.0 * std::log10(static_cast<double>(g_in)) : -240.0;
        if (impl_->guitar_engine) s.guitar_out_peak_dbfs = rt_last_peak_db(impl_->guitar_engine);
    }
    // the tuner listens to the guitar: the guitar engine in the dual layouts and the player, the only engine in the single one
    if (RTEngine* guitar = impl_->guitar_engine ? impl_->guitar_engine : impl_->engine) {
        float hz = 0.0f, confidence = 0.0f;
        if (rt_tuner_read(guitar, &hz, &confidence)) {
            s.tuner_hz = hz;
            s.tuner_confidence = confidence;
        }
    }
    if (impl_->player) {
        s.player = true;
        s.leg = rt_player_selected(impl_->player);
        s.out_peak_dbfs = rt_player_peak_out_db(impl_->player);      // what the monitor hears, not just leg B
        s.a_peak_dbfs = rt_player_peak_a_db(impl_->player);
        s.b_peak_dbfs = rt_player_peak_b_db(impl_->player);
        RTEngine* pedal = rt_player_engine(impl_->player, RT_LEG_PEDAL);
        s.pedal_loaded = rt_model_loaded(pedal) != 0;
        if (s.pedal_loaded) {
            double pin = rt_model_input_level_dbu(pedal), pout = rt_model_output_level_dbu(pedal);
            if (!std::isnan(pin)) s.pedal_input_level_dbu = pin;
            if (!std::isnan(pout)) s.pedal_output_level_dbu = pout;
        }
        s.ir_a_loaded = rt_ir_loaded(rt_player_engine(impl_->player, RT_LEG_A)) != 0;
    }
    return s;
}

#endif  // NAM_HAVE_ASIO_SDK
