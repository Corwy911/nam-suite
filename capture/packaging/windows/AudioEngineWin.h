// Windows counterpart of packaging/AudioEngine.swift: native duplex audio I/O for the "Prova dal vivo" engine.
// Same public shape as the Swift class (open/openDual/close, the gain/EQ/gate setters, loadModel/loadIR, state()),
// so ControlServerWin.cpp can mirror ControlServer.swift's route table almost line for line.
//
// Backed by the Steinberg ASIO SDK (Roman's choice, for parity with the Python capture_tool's own ASIO backend -
// see native/third_party/ASIO_SDK_SETUP.md for why the SDK itself is not vendored in this repo). Build with
// NAM_HAVE_ASIO_SDK defined once the SDK is in place (see AudioEngineWin.cpp's top comment); without it, open()/
// openDual() fail with a clear "ASIO SDK not available" error instead of refusing to build, so ControlServerWin
// and everything above it can be built and exercised (against a device that always fails to open) before the SDK
// lands.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

class AudioEngineWin {
public:
    struct State {
        double in_peak_dbfs = -240.0;
        double out_peak_dbfs = -240.0;
        int xruns = 0;
        bool model_loaded = false;
        bool ir_loaded = false;
        std::optional<double> model_input_level_dbu;
        std::optional<double> model_output_level_dbu;
        std::optional<double> guitar_in_peak_dbfs;
        std::optional<double> guitar_out_peak_dbfs;
        std::optional<double> tuner_hz;           // the note on the guitar side (tuner), none while nothing clear is playing
        std::optional<double> tuner_confidence;
        // player mode ("Lettore NAM", open_player) only
        bool player = false;
        int leg = 0;                              // 0 = A (the real gear, via the return), 1 = B (the model)
        bool ir_a_loaded = false;
        std::optional<double> a_peak_dbfs;
        std::optional<double> b_peak_dbfs;
        bool pedal_loaded = false;
        std::optional<double> pedal_input_level_dbu;
        std::optional<double> pedal_output_level_dbu;
    };

    AudioEngineWin();
    ~AudioEngineWin();
    AudioEngineWin(const AudioEngineWin&) = delete;
    AudioEngineWin& operator=(const AudioEngineWin&) = delete;

    // Returns an empty string on success, an error message otherwise (mirrors AudioEngine.swift's `throws`).
    std::string open(const std::string& device_name, int in_channel, int out_channel,
                      const std::vector<int>& extra_out_channels, int blocksize, double gain_db, double ceiling_db);
    std::string open_dual(const std::string& device_name, int guitar_channel, int return_channel,
                           int pedal_out_channel, const std::vector<int>& monitor_out_channels, int blocksize,
                           double guitar_gain_db, double ceiling_db);
    // The NAM player: guitar -> model (leg B) and guitar -> reamp output -> real gear -> return (leg A) at once, the monitor hears
    // one of them (select_leg). Same channel layout as open_dual; all the routing is RTPlayer's (native/engine/RTEngine.h).
    std::string open_player(const std::string& device_name, int guitar_channel, int return_channel,
                             int reamp_out_channel, const std::vector<int>& monitor_out_channels, int blocksize,
                             double feed_gain_db, double ceiling_db);
    void close();

    bool is_open() const { return is_open_; }
    int granted_blocksize() const { return granted_blocksize_; }
    double sample_rate() const { return sample_rate_; }

    void set_gain_db(double db);
    void set_pre_gain_db(double db);
    void set_guitar_gain_db(double db);       // dual mode only, no-op otherwise
    void set_gate_threshold_db(double db);
    void set_eq_bass_db(double db);
    void set_eq_mid_db(double db);
    void set_eq_treble_db(double db);
    void set_ir_in_gain_db(double db);
    void set_ir_out_gain_db(double db);
    // player mode only, no-ops otherwise
    void select_leg(int leg);                 // 0 = A, 1 = B
    void set_amp_gain_db(double db);          // leg A: return -> monitor (Ci_return - Co)
    void set_b_trim_db(double db);
    void set_b_delay(int samples);

    // legs: bit 0 = leg B (the model's chain, the default), bit 1 = leg A (a second copy of the amp's model for the real pedal's return; player mode only)
    std::string load_model(const std::string& path, int legs = 1);   // "" on success
    void unload_model(int legs = 1);
    int model_sizes();                                  // A2 models: 2 (Lite, full); 1 = ordinary model; 0 = none
    int set_model_size(int index);                      // 0 = Lite ... sizes-1 = full; returns the size in use (the amp: both copies)
    // "Test Pedali" (player mode only): a pedal's NAM in front of the amp's, see RTEngine.h
    std::string load_pedal(const std::string& path);    // "" on success
    void unload_pedal();
    int pedal_sizes();
    int set_pedal_size(int index);
    void set_pedal_pre_gain_db(double db);              // the pedal model's input level
    void set_pedal_gain_db(double db);                  // the pedal model's output level
    void set_amp_pre_gain_db(double db);                // leg A: return -> amp model input
    // legs: bit 0 = leg B (the model's chain, the only one outside player mode), bit 1 = leg A (the real gear's, player mode only)
    std::string load_ir(const std::string& path, int legs = 1);   // "" on success; decodes+downmixes the WAV itself (WavRead.h)
    void unload_ir(int legs = 1);

    State state() const;

    // Enumerates the ASIO drivers Windows has registered (HKEY_LOCAL_MACHINE\SOFTWARE\ASIO), "real interfaces
    // first" the same way capture_tool/capture/backends.py's device list already does on the Python side - this
    // is a separate static helper (not tied to an open instance) so ControlServerWin/the shell can list devices
    // without opening one, matching AudioEngine.swift's own findDevice() being usable independently of open().
    static std::vector<std::string> list_driver_names();

    // Public so the free helper functions in AudioEngineWin.cpp's anonymous namespace (open_common, teardown) can
    // take an `Impl*` parameter - the struct's actual definition still only exists in the .cpp (pimpl), this just
    // makes the *name* visible, same as any pimpl class that needs non-member functions operating on it.
    struct Impl;

private:
    std::string open_dual_common(const std::string& device_name, int guitar_channel, int return_channel, int out_channel,
                                  const std::vector<int>& monitor_out_channels, int blocksize, double guitar_gain_db,
                                  double ceiling_db, bool player);

    Impl* impl_;   // pimpl: keeps ASIO SDK types out of this header, same boundary RTEngine.h keeps around NAM/Eigen

    bool is_open_ = false;
    double sample_rate_ = 48000.0;
    int granted_blocksize_ = 0;
};
