// RTEngine: the native, real-time-safe audio chain for "live" listening - unity-gain monitor,
// and now NAM (amp profile) + IR (cabinet). Mono in, mono out, channel-layout-agnostic on purpose:
// the caller (AudioEngine.swift) picks the right input/output channel out of the Core Audio buffer
// and hands this engine a plain contiguous block. rt_process() never allocates, never locks, never
// blocks: it is meant to be called from a real-time audio callback thread.
//
// Chain: input -> [pre-gain] -> [noise gate] -> [NAM model, optional] -> [tone stack] -> [IR, optional] ->
// [output gain + ceiling] -> output. With no model and no IR loaded, and the gate/tone stack left at their
// default (off/flat) values, this is exactly Phase 1's plain gain stage (the unity-gain guitar monitor):
// pre-gain defaults to unity and every optional stage is pass-through. Loading/unloading a model or an IR
// (rt_load_model/rt_load_ir) is NOT real-time safe - call only from a control thread (see the .cpp for the
// hot-swap contract) - but never blocks or glitches rt_process itself.
//
// extern "C": a plain C ABI so Swift can call this directly via a bridging header, no Objective-C++
// glue needed.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RTEngine RTEngine;

// sample_rate: used to Reset() a newly loaded NAM model to the right rate (see rt_load_model).
RTEngine* rt_create(double sample_rate);
void rt_destroy(RTEngine* engine);

// Glides towards this gain (dB) over ~100 ms, same slew as the Python monitor; never clicks. This
// is the OUTPUT-side gain: with the unity-gain monitor (no model/IR loaded) it is the whole story
// (G = Ci - Co); with NAM+IR loaded it is the final trim to the interface's calibrated output level
// (Co - model_output_level_dbu, computed by the caller - RTEngine does not read calibration.json).
void rt_set_gain_db(RTEngine* engine, double gain_db);

// The INPUT-side gain, applied before the NAM model (if any). Also glides, same slew. Unity (0 dB)
// by default: only meaningful once a model is loaded, to present it with the level it was
// calibrated at (model_input_level_dbu - Ci_return, computed by the caller). No-op without a model.
void rt_set_pre_gain_db(RTEngine* engine, double gain_db);

// A signal peak (in dBFS) larger than this is hard-clipped before it ever reaches the output. Default -1 dBFS,
// matching capture/generators.py:PassthroughGen's default; call once after rt_create if a different ceiling is wanted.
void rt_set_ceiling_db(RTEngine* engine, double ceiling_dbfs);

// Starts fading the OUTPUT gain to zero. rt_is_finished() becomes true once it reaches silence: the
// caller can then stop the audio stream without a click, the same contract as
// capture/generators.py's Generator.stop()/finished.
void rt_stop(RTEngine* engine);
int rt_is_finished(const RTEngine* engine);

// Processes n_frames of mono audio through the whole chain. Real-time safe: no allocation, no
// locking (model/IR pointers are read with a single atomic load).
void rt_process(RTEngine* engine, const float* in, float* out, int n_frames);

// The peak (dBFS) of the last block rt_process wrote to `out`, for a live meter. -240.0 before the first block.
double rt_last_peak_db(const RTEngine* engine);

// ---------------------------------------------------------------------------------- NAM model
// Loads a .nam profile (NeuralAmpModelerCore's own JSON format - the same file capture/store.py's
// captures already produce). NOT real-time safe: call only from a control thread, never from the
// audio callback. On success the model is swapped in atomically for the next rt_process call; the
// previously loaded model (if any) is released once the audio thread is done with its last block -
// never synchronously on this call, so rt_load_model itself never glitches audio that is running
// concurrently on another device. Returns 1 on success, 0 on failure (see rt_last_error).
int rt_load_model(RTEngine* engine, const char* nam_path);
void rt_unload_model(RTEngine* engine);
int rt_model_loaded(const RTEngine* engine);

// The profile's own calibration levels, straight from the .nam file's metadata (the same numbers
// capture/store.py writes as input_level_dbu/output_level_dbu when IT creates a capture) - NAN if
// the loaded model doesn't carry them (an older or hand-made file).
double rt_model_input_level_dbu(const RTEngine* engine);
double rt_model_output_level_dbu(const RTEngine* engine);

// A2 / "slimmable" profiles: the official trainer 0.13 exports a SlimmableContainer - a small ("Lite") and a full WaveNet in one file.
// rt_model_sizes: how many sizes the loaded model offers (1 = an ordinary model, 0 = none loaded). rt_set_model_size picks one,
// 0 = the smallest ... sizes-1 = the full one (what a model starts with). NOT real-time safe (control thread): the new size is reset and
// prewarmed before the audio thread switches to it, so the switch does not glitch. Returns the size now in use (-1 without a model).
int rt_model_sizes(const RTEngine* engine);
int rt_set_model_size(RTEngine* engine, int index);
int rt_model_size(const RTEngine* engine);

// ---------------------------------------------------------------------------------- IR (cabinet)
// samples: mono, already decoded and at the engine's own sample rate - the caller (AudioEngine.swift)
// reads the WAV and rejects/downmixes before calling this; RTEngine itself does no file or format
// I/O, on purpose (same hybrid boundary as the rest of this project: file/OS work in Swift, only the
// inner DSP loop here). Copies `samples` internally; the caller's buffer can be freed right after
// this call returns. NOT real-time safe, same hot-swap contract as rt_load_model. Returns 1 on
// success, 0 on failure (see rt_last_error).
int rt_load_ir(RTEngine* engine, const float* samples, int n_samples);
void rt_unload_ir(RTEngine* engine);
int rt_ir_loaded(const RTEngine* engine);

// Empty string if the last rt_load_model/rt_load_ir call succeeded.
const char* rt_last_error(const RTEngine* engine);

// ---------------------------------------------------------------------------------- noise gate + tone stack
// Both sit on the model path - gate right after the pre-gain (before the NAM model), tone stack right after it
// (before the IR) - and both work whether or not a model/IR is loaded, with harmless defaults when left alone,
// same spirit as pre/post gain. Not real-time-critical to change (a plugin knob moving is a rare, user-driven
// event): each setter just publishes a value the audio thread picks up at the start of its next block, no glide.

// A downward noise gate: below this level (dBFS) the pre-gain input fades towards silence over ~150 ms and back
// open over ~3 ms once it's above threshold again (fixed timing, only the threshold itself is a parameter,
// matching how the reference plugin's own single-knob gate behaves). -100 dBFS by default: at ordinary levels
// that is effectively "off" - nothing has to cross it for the gate to stay open.
void rt_set_gate_threshold_db(RTEngine* engine, double threshold_dbfs);

// A simple 3-band tone stack: low shelf (~120 Hz), peaking mid (~800 Hz, Q 0.7), high shelf (~3 kHz). Each 0 dB
// (default) is flat/transparent - this is a practical EQ for shaping the amp's tone, not a circuit-accurate
// recreation of any particular real tone stack.
void rt_set_eq_bass_db(RTEngine* engine, double gain_db);
void rt_set_eq_mid_db(RTEngine* engine, double gain_db);
void rt_set_eq_treble_db(RTEngine* engine, double gain_db);

// The cabinet's own level knobs: a gain right before the IR convolver and one right after it, in dB (0 = unity, the
// default). Same ~100 ms glide as the other gains, so turning them never clicks. They only act while an IR is loaded -
// with none loaded the chain is transparent there, as before.
void rt_set_ir_in_gain_db(RTEngine* engine, double gain_db);
void rt_set_ir_out_gain_db(RTEngine* engine, double gain_db);

// ---------------------------------------------------------------------------------- tuner
// Every engine listens to the samples it is given (the raw input, before any gain) and a poller can ask which note they hold. The audio
// thread only stores the samples; the analysis (Tuner.h) runs inside rt_tuner_read, on the caller's thread - the HTTP handler of the
// shell's /state. Returns 1 and the fundamental frequency in Hz (confidence 0..1) while a clear note is playing, 0 otherwise.
int rt_tuner_read(RTEngine* engine, float* frequency_hz, float* confidence);

// ---------------------------------------------------------------------------------- the player (A/B)
// "Lettore NAM": the guitar plays through the model (leg B) and, at the same time, through the REAL gear (leg A), and the
// monitor hears one of them - a click-free switch between the two, with every level already matched by calibration. The whole
// routing lives here, once, so the Windows shell (ASIO) and the Mac shell (Core Audio) only hand over audio blocks:
//
//     guitar ---> [feed: gain Ci_guitar - Co] -----------------------------------> feed_out   (to the reamp output -> amp)
//     guitar ---> [B: pre -> gate -> NAM -> EQ -> IR -> post] --(trim, delay)--+
//     return ---> [A: (IR) -> gain Ci_return - Co] ----------------------------+--> cross-fade --> monitor_out
//
// Both legs run on every block, whichever is heard, so a switch never has to wake up a model (no click, no state to rebuild).
// The three RTEngine behind it are exposed (rt_player_engine) so the whole existing API - rt_load_model, rt_load_ir, the gains,
// gate, tone stack - works on them unchanged.
typedef struct RTPlayer RTPlayer;
typedef enum { RT_LEG_FEED = 0, RT_LEG_A = 1, RT_LEG_B = 2, RT_LEG_PEDAL = 3 } RTPlayerPart;

// "Test Pedali": a pedal's NAM in front of the amp's. With a model loaded on the RT_LEG_PEDAL engine the player changes shape:
//
//     guitar ---> [gate -> pedal NAM] -> [leg B: EQ -> amp NAM -> IR -> post] --(trim, delay)--+
//     return ---> [leg A: EQ -> amp NAM (a second copy of the same file) -> IR -> post] -------+--> cross-fade --> monitor
//
// A is the REAL pedal (its output comes back on the return input) into the amp's model, B is the pedal's model into the same amp model, so
// the two legs differ by the pedal only. The noise gate stays the first effect: the one set on leg B moves in front of the pedal (and in
// front of the reamp feed to the real pedal); the tone stack moves between the pedal and the amp's model; leg A copies leg B's tone stack and cabinet
// levels, so the amp stage is identical on both.
// Without a model on the pedal engine nothing changes: leg B takes the guitar directly, as before. Loading the amp's file on leg A is the
// caller's job (rt_load_model on RT_LEG_A, a second copy: a model keeps state), and so are the gains (pedal: input / output; leg B's pre
// gain = pedal output level - amp input level; leg A's pre gain = return Ci - amp input level).

RTPlayer* rt_player_create(double sample_rate);
void rt_player_destroy(RTPlayer* player);

// The engine of a part: RT_LEG_FEED (guitar -> reamp output), RT_LEG_A (return -> monitor, the real gear), RT_LEG_B (guitar -> NAM -> IR -> monitor).
// Owned by the player: never rt_destroy it. rt_set_gain_db on it is that part's gain, rt_set_pre_gain_db the model's input gain.
RTEngine* rt_player_engine(RTPlayer* player, RTPlayerPart part);

// Real-time safe: two mono input blocks (the guitar and the return of the gear), n_frames each. Writes the reamp feed and the monitor mix.
void rt_player_process(RTPlayer* player, const float* guitar, const float* ret, float* feed_out, float* monitor_out, int n_frames);

// Which leg the monitor hears: 0 = A (the real gear, through the return), 1 = B (the model). A ~10 ms equal-power cross-fade, never a click.
void rt_player_select(RTPlayer* player, int leg);
int rt_player_selected(const RTPlayer* player);
int rt_player_pedal_active(const RTPlayer* player);       // 1 when a model is loaded on the pedal stage

// A fine level trim of leg B against leg A (dB, glides like every gain): calibration already puts both on the same analog level,
// this is for the ear. And a delay of leg B in samples (0 .. 9600): leg A comes back through DAC, reamp box, gear and ADC, leg B
// does not, so delaying B by the measured round trip lines the two up when switching. Changing it while B is heard may tick.
void rt_player_set_b_trim_db(RTPlayer* player, double trim_db);
void rt_player_set_b_delay(RTPlayer* player, int samples);

// Peaks (dBFS) of the last block: each leg as it leaves its engine (heard or not, to match them by eye) and the final monitor mix.
double rt_player_peak_a_db(const RTPlayer* player);
double rt_player_peak_b_db(const RTPlayer* player);
double rt_player_peak_out_db(const RTPlayer* player);

// Fades all three parts to silence; rt_player_is_finished() is true once they all reached it (same contract as rt_stop).
void rt_player_stop(RTPlayer* player);
int rt_player_is_finished(const RTPlayer* player);

#ifdef __cplusplus
}
#endif
