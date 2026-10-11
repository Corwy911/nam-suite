// Implementation of RTEngine (see RTEngine.h).
//
// The OUTPUT gain math mirrors capture/generators.py:PassthroughGen exactly (same SLEW constant,
// same per-block linear ramp, same hard ceiling) - this was Phase 1, a deliberate choice so the
// native engine sounds and behaves identically to the Python monitor it can replace. The INPUT
// (pre-NAM) gain added in this phase reuses the very same ramp math, just applied earlier in the
// chain and with no ceiling (it feeds an internal float pipeline, not hardware).
//
// Model/IR hot-swap: NAM inference and IR convolution both need real (non-real-time-safe) allocation to load, so
// the control thread builds the new object completely, publishes it with ONE atomic pointer store, and the audio
// thread reads it with ONE atomic pointer load per block - no lock, no reference count, nothing it can block on.
// (This used to be a shared_ptr with std::atomic_load/store, but libc++ implements those free functions with a
// global mutex pool: a control thread inside one of them could make the audio thread wait.)
//
// (All of these are seq_cst on purpose: control does "publish pointer, then read the counter", audio does "bump the counter, then read
// the pointer" - with plain acquire/release both sides may see each other's OLD value, and the old object would be freed under a
// running block. seq_cst forbids that.)
//
// The old object must not be destroyed while a block that loaded it is still running. rt_process bumps a sequence
// counter on entry and on exit (odd = a block is in flight); after publishing, the control thread waits - it may
// sleep, it is not the audio thread - until no block that started before the swap is still running, and only then
// frees the old object. If the audio thread is stuck for longer than the wait, the old object is parked in a
// graveyard until the engine is destroyed: a leak beats a use-after-free.
//
#include "RTEngine.h"
#include "Tuner.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#include <cstdint>

#include <NAM/get_dsp.h>
#include <NAM/slimmable.h>
#include <NAM/model_config.h>
#include <NAM/container.h>
#include <NAM/convnet.h>
#include <NAM/linear.h>
#include <NAM/lstm.h>
#include <NAM/wavenet/model.h>
#include <TwoStageFFTConvolver.h>

namespace {

constexpr double kSlewPerSample = 1.0 / 4800.0; // full-scale amplitude per sample: ~100 ms for a 60 dB change
constexpr double kSilenceFloor = 1e-4;          // below this, rt_stop() is considered to have reached silence
constexpr double kMinDb = -240.0;
constexpr int kMaxBlock = 4096;                 // matches NAM_DEFAULT_MAX_BUFFER_SIZE; rt_process() chunks above this
constexpr size_t kIrHeadBlock = 64;              // low-latency head stage of the two-stage IR convolver
constexpr size_t kIrTailBlock = 4096;            // efficient tail stage (large FFT, amortised over many blocks)

double fromDb(double db) { return std::pow(10.0, db / 20.0); }

double toDb(double linear) {
  double x = std::max(linear, 1e-12);
  return 20.0 * std::log10(x);
}

// One glide-to-target ramp, the same shape rt_process used for the single gain stage in Phase 1.
// Two instances now: pre-NAM (input) and the original post-chain (output, also drives rt_stop()).
struct GainRamp {
  double amp = 0.0;                       // current linear gain, starts silent and glides up (never a step at t=0)
  std::atomic<double> target{1.0};        // linear gain to glide towards; written from the control thread, read from audio

  void setDb(double db) { target.store(fromDb(db), std::memory_order_relaxed); }

  // Computes the per-block step towards target (called once per rt_process chunk); the caller applies
  // amp + step*t per sample, then commits with `amp += step` after the loop.
  double stepFor(int n_frames) const {
    const double t = target.load(std::memory_order_relaxed);
    const double delta = t - amp;
    const double maxStep = kSlewPerSample * static_cast<double>(n_frames);
    return std::clamp(delta, -maxStep, maxStep);
  }
};

// A plain biquad, transposed direct form II (2 state variables, numerically well-behaved for the low/high-shelf
// and peaking coefficients below). Default-constructed as the identity filter (b0=1, everything else 0).
struct Biquad {
  double b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0;
  double z1 = 0.0, z2 = 0.0;

  float process(float x) {
    const double in = static_cast<double>(x);
    const double out = in * b0 + z1;
    z1 = in * b1 + z2 - a1 * out;
    z2 = in * b2 - a2 * out;
    return static_cast<float>(out);
  }
};

// RBJ Audio EQ Cookbook formulas (shelfSlope S=1, Q=0.7 for the peak) - the standard, widely-used derivation,
// not tied to any particular real amp's tone stack. gain_db == 0 is special-cased to the exact identity filter
// (skips the trig entirely, and sidesteps any doubt about the general formula's behaviour right at unity).
void computeLowShelf(Biquad& f, double freqHz, double gainDb, double sampleRate) {
  if (gainDb == 0.0) { f.b0 = 1; f.b1 = 0; f.b2 = 0; f.a1 = 0; f.a2 = 0; return; }
  const double A = std::pow(10.0, gainDb / 40.0);
  const double w0 = 2.0 * M_PI * freqHz / sampleRate;
  const double cosw0 = std::cos(w0), sinw0 = std::sin(w0);
  const double alpha = sinw0 / 2.0 * std::sqrt((A + 1.0 / A) * (1.0 / 1.0 - 1.0) + 2.0); // S=1
  const double twoSqrtAAlpha = 2.0 * std::sqrt(A) * alpha;
  const double a0 = (A + 1) + (A - 1) * cosw0 + twoSqrtAAlpha;
  f.b0 = (A * ((A + 1) - (A - 1) * cosw0 + twoSqrtAAlpha)) / a0;
  f.b1 = (2 * A * ((A - 1) - (A + 1) * cosw0)) / a0;
  f.b2 = (A * ((A + 1) - (A - 1) * cosw0 - twoSqrtAAlpha)) / a0;
  f.a1 = (-2 * ((A - 1) + (A + 1) * cosw0)) / a0;
  f.a2 = ((A + 1) + (A - 1) * cosw0 - twoSqrtAAlpha) / a0;
}

void computeHighShelf(Biquad& f, double freqHz, double gainDb, double sampleRate) {
  if (gainDb == 0.0) { f.b0 = 1; f.b1 = 0; f.b2 = 0; f.a1 = 0; f.a2 = 0; return; }
  const double A = std::pow(10.0, gainDb / 40.0);
  const double w0 = 2.0 * M_PI * freqHz / sampleRate;
  const double cosw0 = std::cos(w0), sinw0 = std::sin(w0);
  const double alpha = sinw0 / 2.0 * std::sqrt((A + 1.0 / A) * (1.0 / 1.0 - 1.0) + 2.0); // S=1
  const double twoSqrtAAlpha = 2.0 * std::sqrt(A) * alpha;
  const double a0 = (A + 1) - (A - 1) * cosw0 + twoSqrtAAlpha;
  f.b0 = (A * ((A + 1) + (A - 1) * cosw0 + twoSqrtAAlpha)) / a0;
  f.b1 = (-2 * A * ((A - 1) + (A + 1) * cosw0)) / a0;
  f.b2 = (A * ((A + 1) + (A - 1) * cosw0 - twoSqrtAAlpha)) / a0;
  f.a1 = (2 * ((A - 1) - (A + 1) * cosw0)) / a0;
  f.a2 = ((A + 1) - (A - 1) * cosw0 - twoSqrtAAlpha) / a0;
}

void computePeaking(Biquad& f, double freqHz, double q, double gainDb, double sampleRate) {
  if (gainDb == 0.0) { f.b0 = 1; f.b1 = 0; f.b2 = 0; f.a1 = 0; f.a2 = 0; return; }
  const double A = std::pow(10.0, gainDb / 40.0);
  const double w0 = 2.0 * M_PI * freqHz / sampleRate;
  const double cosw0 = std::cos(w0), sinw0 = std::sin(w0);
  const double alpha = sinw0 / (2.0 * q);
  const double a0 = 1.0 + alpha / A;
  f.b0 = (1.0 + alpha * A) / a0;
  f.b1 = (-2.0 * cosw0) / a0;
  f.b2 = (1.0 - alpha * A) / a0;
  f.a1 = (-2.0 * cosw0) / a0;
  f.a2 = (1.0 - alpha / A) / a0;
}

// A single-knob downward gate: an envelope follower against a fixed threshold, with fixed attack/hold/release
// timing (only the threshold is a user parameter - see rt_set_gate_threshold_db). configure() must be called
// once, from rt_create, before any process() call (the timing coefficients depend on the sample rate).
struct NoiseGate {
  double thresholdLin = 1e-5;  // -100 dBFS, i.e. effectively open at ordinary playing levels
  double thresholdDbApplied = -100.0;
  double envelope = 0.0;
  double gateGain = 1.0;
  int holdCounter = 0;
  double envCoeff = 0.0, attackCoeff = 0.0, releaseCoeff = 0.0;
  int holdSamples = 0;

  void configure(double sampleRate) {
    auto coeff = [sampleRate](double ms) { return std::exp(-1.0 / (0.001 * ms * sampleRate)); };
    envCoeff = coeff(5.0);
    attackCoeff = coeff(3.0);
    releaseCoeff = coeff(150.0);
    holdSamples = static_cast<int>(0.05 * sampleRate);
  }

  float process(float x) {
    const double in = static_cast<double>(x);
    envelope = envCoeff * envelope + (1.0 - envCoeff) * std::fabs(in);
    if (envelope >= thresholdLin) holdCounter = holdSamples;
    else if (holdCounter > 0) --holdCounter;
    const double target = (holdCounter > 0) ? 1.0 : 0.0;
    const double coeff = (target > gateGain) ? attackCoeff : releaseCoeff;
    gateGain = target + coeff * (gateGain - target);
    return static_cast<float>(in * gateGain);
  }
};

} // namespace

struct RTEngine {
  GainRamp pre;                            // before the NAM model, if any (unity by default: a no-op without one)
  GainRamp irIn;                           // the cabinet's own level knobs: right before / right after the IR convolver
  GainRamp irOut;
  GainRamp post;                           // after IR, if any - the original Phase 1 "gain" (G = Ci - Co), drives rt_stop()
  double ceiling = fromDb(-1.0);           // linear; whatever the signal does, it cannot cross this
  std::atomic<bool> stopping{false};
  double lastPeakDb = kMinDb;
  double sampleRate = 48000.0;

  // What the audio thread reads (raw, lock-free) and what owns it (control thread only, under controlMutex).
  std::atomic<nam::DSP*> modelRaw{nullptr};
  std::atomic<fftconvolver::TwoStageFFTConvolver*> irRaw{nullptr};
  std::unique_ptr<nam::DSP> modelOwner;
  std::unique_ptr<fftconvolver::TwoStageFFTConvolver> irOwner;
  std::atomic<uint64_t> blockSeq{0};                           // odd while rt_process is inside a block (see above)
  std::vector<std::shared_ptr<void>> graveyard;                // old objects the audio thread would not let go of in time

  std::mutex controlMutex;                 // guards the fields below and serialises rt_load_*/rt_unload_* themselves
                                            // (Swift's ControlServer dispatches HTTP handlers concurrently)
  std::string lastError;
  double modelInputLevelDbu = NAN;
  double modelOutputLevelDbu = NAN;
  std::vector<double> modelSizeBreaks;      // a slimmable model's breakpoints (empty: one size only)
  int modelSize = -1;                       // the size in use (index), -1 without a model

  // noise gate (before the model) and tone stack (after it) - see RTEngine.h. The atomics are the only thing the
  // control thread touches; the *Applied caches and the Biquad/NoiseGate state themselves are audio-thread-only,
  // recomputed at the start of a chunk only when the corresponding atomic has actually changed (cheap, and
  // avoids recomputing filter coefficients on every single block for no reason).
  std::atomic<double> gateThresholdDb{-100.0};
  std::atomic<bool> eqBeforeModel{false};   // the player's pedal test puts the tone stack between the pedal and the amp's model, not after it
  std::atomic<bool> gateBypass{false};      // the player moves the gate to the pedal stage: this engine then lets the signal through ungated
  NoiseGate gate;
  std::atomic<double> eqBassDb{0.0}, eqMidDb{0.0}, eqTrebleDb{0.0};
  double eqBassApplied = 0.0, eqMidApplied = 0.0, eqTrebleApplied = 0.0;
  Biquad eqBass, eqMid, eqTreble;

  PitchTuner tuner;                        // fed with the raw input by rt_process, read by rt_tuner_read (see Tuner.h)
  alignas(16) float gateTrace[kMaxBlock];  // the gate's gain for each sample of the last chunk (the player reuses it on the real gear's legs)

  alignas(16) float scratchA[kMaxBlock];
  alignas(16) float scratchB[kMaxBlock];
};

namespace {

// Blocks until no rt_process call that began before this point is still running (or a timeout, returning false).
bool waitAudioQuiescent(const RTEngine* engine) {
  const uint64_t seq = engine->blockSeq.load(std::memory_order_seq_cst);
  if ((seq & 1u) == 0) return true;                             // between blocks (or not running at all)
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
  while (engine->blockSeq.load(std::memory_order_seq_cst) == seq) {
    if (std::chrono::steady_clock::now() > deadline) return false;
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  return true;
}

// The old object is freed here, on the control thread, once the audio thread cannot be using it any more.
template <typename T>
void retire(RTEngine* engine, std::unique_ptr<T> old) {
  if (!old) return;
  if (!waitAudioQuiescent(engine)) engine->graveyard.push_back(std::shared_ptr<void>(std::move(old)));
}

} // namespace

namespace {
// NAM core registers each architecture from a static object in that architecture's own .cpp - but a static library (rtengine.lib, librtengine.a)
// only brings into a program the object files something refers to. The two apps force the whole archive in (/WHOLEARCHIVE, -force_load); any
// other program linking it - the native tests did - silently lost "SlimmableContainer" (what the official trainer 0.13 exports: an A2 model, a
// Lite and a full WaveNet in one file) and failed with "No config parser registered". Naming each create_config here links them all in
// whatever the link options, and registers any that is still missing.
void ensureArchitectures() {
  static std::once_flag once;
  std::call_once(once, [] {
    auto& registry = nam::ConfigParserRegistry::instance();
    const std::pair<const char*, nam::ConfigParserFunction> all[] = {
        {"WaveNet", nam::wavenet::create_config}, {"SlimmableContainer", nam::container::create_config},
        {"LSTM", nam::lstm::create_config}, {"ConvNet", nam::convnet::create_config}, {"Linear", nam::linear::create_config}};
    for (const auto& [name, parser] : all)
      if (!registry.has(name)) registry.registerParser(name, parser);
  });
}
}  // namespace

RTEngine* rt_create(double sample_rate) {
  ensureArchitectures();
  RTEngine* engine = new RTEngine();
  engine->sampleRate = sample_rate;
  engine->tuner.Reset(sample_rate);
  engine->gate.configure(sample_rate);
  engine->irIn.amp = 1.0;    // these two start AT unity (an IR loaded with the knobs untouched must not fade in from silence)
  engine->irOut.amp = 1.0;
  return engine;
}

void rt_destroy(RTEngine* engine) {
  delete engine;
}

void rt_set_gain_db(RTEngine* engine, double gain_db) {
  if (engine == nullptr || engine->stopping.load(std::memory_order_relaxed)) return;
  engine->post.setDb(gain_db);
}

void rt_set_pre_gain_db(RTEngine* engine, double gain_db) {
  if (engine == nullptr) return;
  engine->pre.setDb(gain_db);
}

void rt_set_ceiling_db(RTEngine* engine, double ceiling_dbfs) {
  if (engine == nullptr) return;
  engine->ceiling = fromDb(ceiling_dbfs);
}

void rt_set_ir_in_gain_db(RTEngine* engine, double gain_db) {
  if (engine == nullptr) return;
  engine->irIn.setDb(gain_db);
}

void rt_set_ir_out_gain_db(RTEngine* engine, double gain_db) {
  if (engine == nullptr) return;
  engine->irOut.setDb(gain_db);
}

void rt_set_gate_threshold_db(RTEngine* engine, double threshold_dbfs) {
  if (engine == nullptr) return;
  engine->gateThresholdDb.store(threshold_dbfs, std::memory_order_relaxed);
}

void rt_set_eq_bass_db(RTEngine* engine, double gain_db) {
  if (engine == nullptr) return;
  engine->eqBassDb.store(gain_db, std::memory_order_relaxed);
}

void rt_set_eq_mid_db(RTEngine* engine, double gain_db) {
  if (engine == nullptr) return;
  engine->eqMidDb.store(gain_db, std::memory_order_relaxed);
}

void rt_set_eq_treble_db(RTEngine* engine, double gain_db) {
  if (engine == nullptr) return;
  engine->eqTrebleDb.store(gain_db, std::memory_order_relaxed);
}

void rt_stop(RTEngine* engine) {
  if (engine == nullptr) return;
  engine->stopping.store(true, std::memory_order_relaxed);
  engine->post.target.store(0.0, std::memory_order_relaxed);
}

int rt_is_finished(const RTEngine* engine) {
  if (engine == nullptr) return 1;
  return engine->stopping.load(std::memory_order_relaxed) && engine->post.amp < kSilenceFloor ? 1 : 0;
}

namespace {

// One chunk, n <= kMaxBlock: the actual chain. Split out of rt_process so arbitrarily large blocks
// (in principle - the page's buffer selector tops out at 1024) can be processed in kMaxBlock pieces
// without ever allocating.
// Scales buf in place by a GainRamp, gliding towards its target across the chunk (same maths as the pre/post gains).
void applyRamp(GainRamp& ramp, float* buf, int n) {
  const double step = ramp.stepFor(n);
  for (int i = 0; i < n; ++i) {
    const double t = static_cast<double>(i + 1) / static_cast<double>(n);
    buf[i] = static_cast<float>(static_cast<double>(buf[i]) * (ramp.amp + step * t));
  }
  ramp.amp += step;
}

void processChunk(RTEngine* engine, const float* in, float* out, int n) {
  // 0. pick up gate/EQ parameter changes, if any, once per chunk (not per sample - a plugin knob moving is a
  // rare event, not something worth checking 48000 times a second)
  const double gateDb = engine->gateThresholdDb.load(std::memory_order_relaxed);
  if (gateDb != engine->gate.thresholdDbApplied) {
    engine->gate.thresholdLin = fromDb(gateDb);
    engine->gate.thresholdDbApplied = gateDb;
  }
  const double bassDb = engine->eqBassDb.load(std::memory_order_relaxed);
  if (bassDb != engine->eqBassApplied) {
    computeLowShelf(engine->eqBass, 120.0, bassDb, engine->sampleRate);
    engine->eqBassApplied = bassDb;
  }
  const double midDb = engine->eqMidDb.load(std::memory_order_relaxed);
  if (midDb != engine->eqMidApplied) {
    computePeaking(engine->eqMid, 800.0, 0.7, midDb, engine->sampleRate);
    engine->eqMidApplied = midDb;
  }
  const double trebleDb = engine->eqTrebleDb.load(std::memory_order_relaxed);
  if (trebleDb != engine->eqTrebleApplied) {
    computeHighShelf(engine->eqTreble, 3000.0, trebleDb, engine->sampleRate);
    engine->eqTrebleApplied = trebleDb;
  }

  // 1. pre-gain (feeds the NAM model, if any, at the level it was calibrated at)
  const double preStep = engine->pre.stepFor(n);
  for (int i = 0; i < n; ++i) {
    const double t = static_cast<double>(i + 1) / static_cast<double>(n);
    const double src = (in != nullptr) ? static_cast<double>(in[i]) : 0.0;
    engine->scratchA[i] = static_cast<float>(src * (engine->pre.amp + preStep * t));
  }
  engine->pre.amp += preStep;
  float* cur = engine->scratchA;

  // 1.5 noise gate (right before the model: gates the raw guitar signal, not whatever the model does to it)
  if (engine->gateBypass.load(std::memory_order_relaxed)) {
    for (int i = 0; i < n; ++i) engine->gateTrace[i] = 1.0f;
  } else {
    for (int i = 0; i < n; ++i) {
      cur[i] = engine->gate.process(cur[i]);
      engine->gateTrace[i] = static_cast<float>(engine->gate.gateGain);
    }
  }

  // 1.7 tone stack in front of the model (only in the player's pedal test: pedal -> EQ -> amp)
  const bool eqFirst = engine->eqBeforeModel.load(std::memory_order_relaxed);
  if (eqFirst) {
    for (int i = 0; i < n; ++i) cur[i] = engine->eqTreble.process(engine->eqMid.process(engine->eqBass.process(cur[i])));
  }

  // 2. NAM model (optional)
  nam::DSP* model = engine->modelRaw.load(std::memory_order_seq_cst);
  if (model) {
    NAM_SAMPLE* inCh[1] = {cur};
    NAM_SAMPLE* outCh[1] = {engine->scratchB};
    model->process(inCh, outCh, n);
    cur = engine->scratchB;
  }

  // 2.5 tone stack (bass/mid/treble - after the model, before the cab): flat/no-op at the default 0 dB each
  if (!eqFirst) {
    for (int i = 0; i < n; ++i) cur[i] = engine->eqTreble.process(engine->eqMid.process(engine->eqBass.process(cur[i])));
  }

  // 3. IR / cabinet convolution (optional) - always into the OTHER scratch buffer, never in place
  fftconvolver::TwoStageFFTConvolver* ir = engine->irRaw.load(std::memory_order_seq_cst);
  if (ir) {
    applyRamp(engine->irIn, cur, n);
    float* dst = (cur == engine->scratchA) ? engine->scratchB : engine->scratchA;
    ir->process(cur, dst, static_cast<size_t>(n));
    applyRamp(engine->irOut, dst, n);
    cur = dst;
  }

  // 4. output gain + safety ceiling (unchanged from Phase 1)
  const double postStep = engine->post.stepFor(n);
  double peakLinear = 0.0;
  for (int i = 0; i < n; ++i) {
    const double t = static_cast<double>(i + 1) / static_cast<double>(n);
    const double gain = engine->post.amp + postStep * t;
    double sample = static_cast<double>(cur[i]) * gain;
    sample = std::clamp(sample, -engine->ceiling, engine->ceiling);
    out[i] = static_cast<float>(sample);
    peakLinear = std::max(peakLinear, std::fabs(sample));
  }
  engine->post.amp += postStep;
  engine->lastPeakDb = (n > 0) ? toDb(peakLinear) : engine->lastPeakDb;
}

} // namespace

void rt_process(RTEngine* engine, const float* in, float* out, int n_frames) {
  if (engine == nullptr || out == nullptr || n_frames <= 0) return;
  if (rt_is_finished(engine)) {
    std::fill(out, out + n_frames, 0.0f);
    return;
  }
  engine->blockSeq.fetch_add(1, std::memory_order_seq_cst);   // odd: a block is in flight
  if (in != nullptr) engine->tuner.Process(in, n_frames);
  int offset = 0;
  while (offset < n_frames) {
    const int n = std::min(n_frames - offset, kMaxBlock);
    processChunk(engine, in != nullptr ? in + offset : nullptr, out + offset, n);
    offset += n;
  }
  engine->blockSeq.fetch_add(1, std::memory_order_seq_cst);   // even again
}

int rt_tuner_read(RTEngine* engine, float* frequency_hz, float* confidence) {
  if (engine == nullptr) return 0;
  const PitchTuner::Result r = engine->tuner.Poll();
  if (frequency_hz != nullptr) *frequency_hz = r.valid ? r.frequency : 0.0f;
  if (confidence != nullptr) *confidence = r.valid ? r.confidence : 0.0f;
  return r.valid ? 1 : 0;
}

double rt_last_peak_db(const RTEngine* engine) {
  return (engine != nullptr) ? engine->lastPeakDb : kMinDb;
}

// ---------------------------------------------------------------------------------- NAM model

int rt_load_model(RTEngine* engine, const char* nam_path) {
  if (engine == nullptr || nam_path == nullptr) return 0;
  std::lock_guard<std::mutex> lock(engine->controlMutex);
  try {
    std::unique_ptr<nam::DSP> model = nam::get_dsp(std::filesystem::path(nam_path));
    if (!model) throw std::runtime_error("get_dsp returned null");
    model->Reset(engine->sampleRate, kMaxBlock); // also prewarms by default (see NAM/dsp.h)
    engine->modelInputLevelDbu = model->HasInputLevel() ? model->GetInputLevel() : NAN;
    engine->modelOutputLevelDbu = model->HasOutputLevel() ? model->GetOutputLevel() : NAN;
    auto* slim = dynamic_cast<nam::SlimmableModel*>(model.get());
    engine->modelSizeBreaks = slim ? slim->GetSlimmableSizeBreakpoints() : std::vector<double>{};
    engine->modelSize = static_cast<int>(engine->modelSizeBreaks.size());     // a container starts on its full model
    engine->modelRaw.store(model.get(), std::memory_order_seq_cst);
    std::unique_ptr<nam::DSP> old = std::move(engine->modelOwner);
    engine->modelOwner = std::move(model);
    retire(engine, std::move(old));
    engine->lastError.clear();
    return 1;
  } catch (const std::exception& e) {
    engine->lastError = e.what();
    return 0;
  } catch (...) {
    engine->lastError = "unknown error loading the model";
    return 0;
  }
}

void rt_unload_model(RTEngine* engine) {
  if (engine == nullptr) return;
  std::lock_guard<std::mutex> lock(engine->controlMutex);
  engine->modelRaw.store(nullptr, std::memory_order_seq_cst);
  retire(engine, std::move(engine->modelOwner));
  engine->modelInputLevelDbu = NAN;
  engine->modelOutputLevelDbu = NAN;
  engine->modelSizeBreaks.clear();
  engine->modelSize = -1;
}

int rt_model_sizes(const RTEngine* engine) {
  if (engine == nullptr || engine->modelSize < 0) return 0;
  return static_cast<int>(engine->modelSizeBreaks.size()) + 1;
}

int rt_set_model_size(RTEngine* engine, int index) {
  if (engine == nullptr) return -1;
  std::lock_guard<std::mutex> lock(engine->controlMutex);
  if (engine->modelSize < 0) return -1;
  auto* slim = dynamic_cast<nam::SlimmableModel*>(engine->modelOwner.get());
  const int last = static_cast<int>(engine->modelSizeBreaks.size());
  if (slim == nullptr || last == 0) return engine->modelSize;
  index = std::clamp(index, 0, last);
  // the container picks the first submodel whose max_value is above the value: ask for the middle of that submodel's range
  const double lo = index == 0 ? 0.0 : engine->modelSizeBreaks[index - 1];
  const double value = index == last ? 1.0 : 0.5 * (lo + engine->modelSizeBreaks[index]);
  slim->SetSlimmableSize(value);
  engine->modelSize = index;
  return index;
}

int rt_model_size(const RTEngine* engine) {
  return engine == nullptr ? -1 : engine->modelSize;
}

int rt_model_loaded(const RTEngine* engine) {
  if (engine == nullptr) return 0;
  return engine->modelRaw.load(std::memory_order_acquire) != nullptr ? 1 : 0;
}

double rt_model_input_level_dbu(const RTEngine* engine) {
  if (engine == nullptr) return NAN;
  return engine->modelInputLevelDbu;
}

double rt_model_output_level_dbu(const RTEngine* engine) {
  if (engine == nullptr) return NAN;
  return engine->modelOutputLevelDbu;
}

// ---------------------------------------------------------------------------------- IR (cabinet)

int rt_load_ir(RTEngine* engine, const float* samples, int n_samples) {
  if (engine == nullptr || samples == nullptr || n_samples <= 0) return 0;
  std::lock_guard<std::mutex> lock(engine->controlMutex);
  auto conv = std::make_unique<fftconvolver::TwoStageFFTConvolver>();
  if (!conv->init(kIrHeadBlock, kIrTailBlock, samples, static_cast<size_t>(n_samples))) {
    engine->lastError = "the IR convolver failed to initialise";
    return 0;
  }
  engine->irRaw.store(conv.get(), std::memory_order_seq_cst);
  std::unique_ptr<fftconvolver::TwoStageFFTConvolver> old = std::move(engine->irOwner);
  engine->irOwner = std::move(conv);
  retire(engine, std::move(old));
  engine->lastError.clear();
  return 1;
}

void rt_unload_ir(RTEngine* engine) {
  if (engine == nullptr) return;
  std::lock_guard<std::mutex> lock(engine->controlMutex);
  engine->irRaw.store(nullptr, std::memory_order_seq_cst);
  retire(engine, std::move(engine->irOwner));
}

int rt_ir_loaded(const RTEngine* engine) {
  if (engine == nullptr) return 0;
  return engine->irRaw.load(std::memory_order_acquire) != nullptr ? 1 : 0;
}

const char* rt_last_error(const RTEngine* engine) {
  if (engine == nullptr) return "";
  return engine->lastError.c_str();
}

// ---------------------------------------------------------------------------------- the player (A/B)

namespace {
constexpr int kFadeSamples = 480;        // 10 ms: long enough not to click, short enough to feel instant
constexpr int kMaxBDelay = 9600;         // 200 ms of round trip is already far more than any interface has
constexpr int kDelayBuf = 16384;         // a ring: must be a power of two, and larger than kMaxBDelay
}

struct RTPlayer {
  RTEngine* feed = nullptr;
  RTEngine* legA = nullptr;
  RTEngine* legB = nullptr;
  RTEngine* pedal = nullptr;               // the pedal's NAM stage in front of leg B (idle, and skipped, until a model is loaded on it)
  double sampleRate = 48000.0;

  std::atomic<int> target{0};             // 0 = A, 1 = B: written by the control thread
  double pos = 0.0;                        // audio thread: 0 = all A ... 1 = all B
  GainRamp trimB;                          // starts at unity (see rt_player_create)
  std::atomic<int> delayB{0};

  std::vector<float> ring;                 // leg B's delay line
  std::vector<float> gateRing;             // leg B's gate gain, delayed like the return of the real gear
  size_t writePos = 0;
  size_t gateWrite = 0;
  alignas(16) float scratchA[kMaxBlock];
  alignas(16) float scratchB[kMaxBlock];
  alignas(16) float pedalOut[kMaxBlock];

  std::atomic<double> peakA{kMinDb}, peakB{kMinDb}, peakOut{kMinDb};
};

RTPlayer* rt_player_create(double sample_rate) {
  RTPlayer* p = new RTPlayer();
  p->sampleRate = sample_rate;
  p->feed = rt_create(sample_rate);
  p->legA = rt_create(sample_rate);
  p->legB = rt_create(sample_rate);
  p->pedal = rt_create(sample_rate);
  rt_set_ceiling_db(p->pedal, 60.0);        // its output is a level on the pedal's own scale, scaled again before the amp: never clip here
  p->trimB.amp = 1.0;
  p->ring.assign(kDelayBuf, 0.0f);
  p->gateRing.assign(kDelayBuf, 1.0f);
  return p;
}

void rt_player_destroy(RTPlayer* player) {
  if (player == nullptr) return;
  rt_destroy(player->feed);
  rt_destroy(player->legA);
  rt_destroy(player->legB);
  rt_destroy(player->pedal);
  delete player;
}

RTEngine* rt_player_engine(RTPlayer* player, RTPlayerPart part) {
  if (player == nullptr) return nullptr;
  switch (part) {
    case RT_LEG_FEED: return player->feed;
    case RT_LEG_A: return player->legA;
    case RT_LEG_B: return player->legB;
    case RT_LEG_PEDAL: return player->pedal;
  }
  return nullptr;
}

void rt_player_select(RTPlayer* player, int leg) {
  if (player == nullptr) return;
  player->target.store(leg == 0 ? 0 : 1, std::memory_order_relaxed);
}

int rt_player_pedal_active(const RTPlayer* player) {
  return player != nullptr && rt_model_loaded(player->pedal) ? 1 : 0;
}

int rt_player_selected(const RTPlayer* player) {
  return player != nullptr ? player->target.load(std::memory_order_relaxed) : 0;
}

void rt_player_set_b_trim_db(RTPlayer* player, double trim_db) {
  if (player == nullptr) return;
  player->trimB.setDb(trim_db);
}

void rt_player_set_b_delay(RTPlayer* player, int samples) {
  if (player == nullptr) return;
  player->delayB.store(std::clamp(samples, 0, kMaxBDelay), std::memory_order_relaxed);
}

double rt_player_peak_a_db(const RTPlayer* player) { return player ? player->peakA.load(std::memory_order_relaxed) : kMinDb; }
double rt_player_peak_b_db(const RTPlayer* player) { return player ? player->peakB.load(std::memory_order_relaxed) : kMinDb; }
double rt_player_peak_out_db(const RTPlayer* player) { return player ? player->peakOut.load(std::memory_order_relaxed) : kMinDb; }

void rt_player_stop(RTPlayer* player) {
  if (player == nullptr) return;
  rt_stop(player->feed);
  rt_stop(player->pedal);
  rt_stop(player->legA);
  rt_stop(player->legB);
}

int rt_player_is_finished(const RTPlayer* player) {
  if (player == nullptr) return 1;
  return rt_is_finished(player->feed) && rt_is_finished(player->legA) && rt_is_finished(player->legB) ? 1 : 0;
}

namespace {
double peakOf(const float* x, int n) {
  float p = 0.0f;
  for (int i = 0; i < n; ++i) p = std::max(p, std::fabs(x[i]));
  return toDb(p);
}

void playerChunk(RTPlayer* p, const float* guitar, const float* ret, float* feedOut, float* monitorOut, int n) {
  // "Test Pedali": with a pedal model loaded the gate set on leg B works in front of the pedal (it is the first effect, for the real pedal's
  // feed too), leg B takes the pedal's output, and leg A (the real pedal's return into a copy of the amp's model) copies leg B's tone stack and
  // cabinet levels so the amp stage is the same on both.
  const bool pedalOn = rt_model_loaded(p->pedal) != 0;
  RTEngine* gateEngine = p->legB;
  if (pedalOn) {
    p->pedal->gateThresholdDb.store(p->legB->gateThresholdDb.load(std::memory_order_relaxed), std::memory_order_relaxed);
    p->legB->gateBypass.store(true, std::memory_order_relaxed);
    p->legB->eqBeforeModel.store(true, std::memory_order_relaxed);        // gate -> pedal -> EQ -> amp -> IR, on both legs
    p->legA->eqBeforeModel.store(true, std::memory_order_relaxed);
    p->legA->eqBassDb.store(p->legB->eqBassDb.load(std::memory_order_relaxed), std::memory_order_relaxed);
    p->legA->eqMidDb.store(p->legB->eqMidDb.load(std::memory_order_relaxed), std::memory_order_relaxed);
    p->legA->eqTrebleDb.store(p->legB->eqTrebleDb.load(std::memory_order_relaxed), std::memory_order_relaxed);
    p->legA->irIn.target.store(p->legB->irIn.target.load(std::memory_order_relaxed), std::memory_order_relaxed);
    p->legA->irOut.target.store(p->legB->irOut.target.load(std::memory_order_relaxed), std::memory_order_relaxed);
    rt_process(p->pedal, guitar, p->pedalOut, n);
    rt_process(p->legB, p->pedalOut, p->scratchB, n);
    gateEngine = p->pedal;
  } else {
    p->legB->gateBypass.store(false, std::memory_order_relaxed);
    p->legB->eqBeforeModel.store(false, std::memory_order_relaxed);
    p->legA->eqBeforeModel.store(false, std::memory_order_relaxed);
    rt_process(p->legB, guitar, p->scratchB, n);
  }
  rt_process(p->feed, guitar, feedOut, n);
  rt_process(p->legA, ret, p->scratchA, n);
  // The noise gate is set on leg B (the model's input), but the comparison must be fair: the same gate, opening and closing on the same
  // guitar, also sits in front of the real gear (the reamp feed, like a gate pedal before the amp) and on what comes back from it (so the
  // amp's own hiss goes quiet as the model's does), that one delayed by B's delay - the measured round trip, i.e. when the return arrives.
  const bool gated = gateEngine->gate.thresholdDbApplied > -99.5;       // -100 dB = the gate is off: leave the real gear alone
  const size_t gmask = p->gateRing.size() - 1;
  const size_t gdelay = static_cast<size_t>(std::max(0, p->delayB.load(std::memory_order_relaxed)));
  for (int i = 0; i < n; ++i) {
    const float g = gated ? gateEngine->gateTrace[i] : 1.0f;
    p->gateRing[(p->gateWrite + static_cast<size_t>(i)) & gmask] = g;
    feedOut[i] *= g;
    p->scratchA[i] *= p->gateRing[(p->gateWrite + static_cast<size_t>(i) - gdelay) & gmask];
  }
  p->gateWrite += static_cast<size_t>(n);
  p->peakA.store(peakOf(p->scratchA, n), std::memory_order_relaxed);
  p->peakB.store(peakOf(p->scratchB, n), std::memory_order_relaxed);

  applyRamp(p->trimB, p->scratchB, n);
  const int delay = p->delayB.load(std::memory_order_relaxed);
  if (delay > 0) {                                   // B reaches the mix `delay` samples late, like A coming back from the real gear
    const size_t mask = p->ring.size() - 1;
    for (int i = 0; i < n; ++i) {
      p->ring[p->writePos & mask] = p->scratchB[i];
      p->scratchB[i] = p->ring[(p->writePos - static_cast<size_t>(delay)) & mask];
      ++p->writePos;
    }
  } else {                                           // keep the line warm, so turning a delay on later has no silence to play
    const size_t mask = p->ring.size() - 1;
    for (int i = 0; i < n; ++i) p->ring[(p->writePos++) & mask] = p->scratchB[i];
  }

  const double goal = static_cast<double>(p->target.load(std::memory_order_relaxed));
  const double step = 1.0 / static_cast<double>(kFadeSamples);
  const double ceiling = p->legB->ceiling;           // both legs already clip at the engine's ceiling; the sum must not cross it either
  double peak = 0.0;
  for (int i = 0; i < n; ++i) {
    if (p->pos < goal) p->pos = std::min(goal, p->pos + step);
    else if (p->pos > goal) p->pos = std::max(goal, p->pos - step);
    const double a = std::cos(p->pos * M_PI / 2.0), b = std::sin(p->pos * M_PI / 2.0);   // equal power; exactly 1/0 at the ends
    const double y = std::clamp(a * static_cast<double>(p->scratchA[i]) + b * static_cast<double>(p->scratchB[i]), -ceiling, ceiling);
    monitorOut[i] = static_cast<float>(y);
    peak = std::max(peak, std::fabs(y));
  }
  p->peakOut.store(toDb(peak), std::memory_order_relaxed);
}
}  // namespace

void rt_player_process(RTPlayer* player, const float* guitar, const float* ret, float* feed_out, float* monitor_out, int n_frames) {
  if (player == nullptr || feed_out == nullptr || monitor_out == nullptr || n_frames <= 0) return;
  int offset = 0;
  while (offset < n_frames) {
    const int n = std::min(n_frames - offset, kMaxBlock);
    playerChunk(player, guitar != nullptr ? guitar + offset : nullptr, ret != nullptr ? ret + offset : nullptr,
                feed_out + offset, monitor_out + offset, n);
    offset += n;
  }
}
