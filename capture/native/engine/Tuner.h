// Pitch detector for the built-in tuner (header only, no dependencies, same code on Windows and macOS).
//
// Based on NAMTunerDetector from NAM-Oversampler (https://github.com/DLC86/NAM-Oversampler, NeuralAmpModeler/Tuner.h), MIT licence,
// Copyright (c) 2022 Steven Atkinson (see THIRD_PARTY.md). What was kept: the YIN detector at about 24 kHz (cumulative mean normalised
// difference, threshold 0.15, parabolic peak), the confidence gate, the median of the last five readings and the smoothing that follows
// a note but jumps at once to a new one. What changed, measured against synthetic guitar notes (tests/test_tuner.cpp):
//   - the audio thread only stores samples; everything else runs on whoever polls (the HTTP thread), so rt_process() stays untouched
//   - the YIN result is a coarse period only: it is refined at the FULL sample rate on a lag of several periods (about 25 ms), where the
//     minimum of the difference function is sharp, and then replaced by the measured FUNDAMENTAL (spectral peak near it): the period of a
//     real string reads sharp by 1-5 cent because its partials are stretched. The error on the highest notes drops from about 1.4 cent (1 kHz) to nearly nothing and
//     the low notes get steadier, because the length of the lag divides the error
//   - low range down to 28 Hz (5-string bass B0), a longer window for it, and a quiet gate in dBFS instead of an almost-zero threshold
//   - all pending hops are analysed in order when polled, so the median/smoothing mean the same thing at any polling rate
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <vector>

class PitchTuner {
 public:
  struct Result {
    float frequency = 0.0f;   // Hz
    float confidence = 0.0f;  // 0..1 (1 - YIN value)
    bool valid = false;
  };

  PitchTuner() : ring_(kRingSize, 0.0f) {}

  void Reset(double sampleRate) {
    sampleRate_ = std::max(8000.0, sampleRate);
    decimation_ = std::clamp(static_cast<int>(std::lround(sampleRate_ / 24000.0)), 1, 16);
    total_.store(0, std::memory_order_relaxed);
    lastAnalysed_ = 0;
    history_.fill(0.0f);
    historyCount_ = historyPos_ = 0;
    smoothed_ = 0.0;
    Publish({});
  }

  // Audio thread: store the samples, nothing else.
  void Process(const float* in, int n) {
    uint64_t total = total_.load(std::memory_order_relaxed);
    for (int i = 0; i < n; ++i) ring_[static_cast<size_t>((total + static_cast<uint64_t>(i)) & kRingMask)] = in[i];
    total_.store(total + static_cast<uint64_t>(n), std::memory_order_release);
  }

  // Any other thread: analyses what arrived since the last call (at most the last kMaxHops hops) and returns the latest reading.
  Result Poll() {
    if (busy_.test_and_set(std::memory_order_acquire)) return Latest();       // another poll is running: its answer is as good
    const uint64_t total = total_.load(std::memory_order_acquire);
    const uint64_t hop = static_cast<uint64_t>(kHop) * static_cast<uint64_t>(decimation_);
    const uint64_t need = NeededSamples();
    if (total < need) {
      busy_.clear(std::memory_order_release);
      return Latest();
    }
    if (total < lastAnalysed_) lastAnalysed_ = 0;                              // Reset() ran meanwhile
    uint64_t first = lastAnalysed_ == 0 ? total : lastAnalysed_ + hop;
    if (total - first > hop * (kMaxHops - 1)) first = total - hop * (kMaxHops - 1);   // far behind: skip to the recent past
    if (first < need) first = need;
    for (uint64_t end = first; end <= total; end += hop) {
      Analyse(end);
      lastAnalysed_ = end;
      if (end + hop > total) break;
    }
    busy_.clear(std::memory_order_release);
    return Latest();
  }

  Result Latest() const {
    Result r;
    r.frequency = pubFrequency_.load(std::memory_order_relaxed);
    r.confidence = pubConfidence_.load(std::memory_order_relaxed);
    r.valid = pubValid_.load(std::memory_order_acquire);
    return r;
  }

 private:
  static constexpr int kWindow = 3072;              // YIN window, in analysis-rate samples (about 128 ms)
  static constexpr int kHop = 1024;                 // analysis-rate samples between readings (about 43 ms)
  static constexpr int kMaxHops = 6;
  static constexpr double kMinHz = 28.0, kMaxHz = 1400.0;
  static constexpr float kYinThreshold = 0.15f;
  static constexpr float kMinConfidence = 0.72f;
  static constexpr double kMinRms = 0.0006;         // about -64 dBFS: below that there is no note, only the interface's own noise
  static constexpr double kRefineLagSeconds = 0.025;
  static constexpr double kRefineWindowSeconds = 0.128;
  static constexpr size_t kRingSize = 65536;        // raw samples: 1.3 s at 48 kHz, 0.34 s at 192 kHz
  static constexpr uint64_t kRingMask = kRingSize - 1;

  uint64_t NeededSamples() const {
    const uint64_t coarse = static_cast<uint64_t>(kWindow) * static_cast<uint64_t>(decimation_);
    const uint64_t refine = static_cast<uint64_t>((kRefineWindowSeconds + kRefineLagSeconds * 1.6) * sampleRate_) + 128;
    const uint64_t fundamental = static_cast<uint64_t>(0.30 * sampleRate_) + 128;     // the longest window of RefineFundamental
    return std::max({coarse, refine, fundamental});
  }

  float Raw(uint64_t index) const { return ring_[static_cast<size_t>(index & kRingMask)]; }

  void Analyse(uint64_t end) {
    const int k = decimation_;
    const double rate = sampleRate_ / k;
    // decimated window: mean of k samples (the same simple low-pass as the original), mean removed
    std::array<float, kWindow>& x = window_;
    const uint64_t start = end - static_cast<uint64_t>(kWindow) * static_cast<uint64_t>(k);
    double mean = 0.0;
    for (int i = 0; i < kWindow; ++i) {
      double s = 0.0;
      for (int j = 0; j < k; ++j) s += Raw(start + static_cast<uint64_t>(i) * k + j);
      x[static_cast<size_t>(i)] = static_cast<float>(s / k);
      mean += x[static_cast<size_t>(i)];
    }
    mean /= kWindow;
    double sumSq = 0.0;
    for (auto& v : x) {
      v -= static_cast<float>(mean);
      sumSq += static_cast<double>(v) * v;
    }
    if (std::sqrt(sumSq / kWindow) < kMinRms) return Silence();

    const int minTau = std::max(2, static_cast<int>(std::floor(rate / kMaxHz)));
    const int maxTau = std::min(kWindow / 2, static_cast<int>(std::ceil(rate / kMinHz)));
    const int span = kWindow - maxTau;

    // YIN: difference function and its cumulative mean normalisation, stopping at the first dip under the threshold
    std::vector<float>& cm = cmnd_;
    cm.assign(static_cast<size_t>(maxTau) + 2, 1.0f);
    double running = 0.0;
    int candidate = -1;
    for (int tau = 1; tau <= maxTau; ++tau) {
      double d = 0.0;
      for (int i = 0; i < span; ++i) {
        const double delta = static_cast<double>(x[static_cast<size_t>(i)]) - static_cast<double>(x[static_cast<size_t>(i + tau)]);
        d += delta * delta;
      }
      running += d;
      cm[static_cast<size_t>(tau)] = running > 0.0 ? static_cast<float>(d * tau / running) : 1.0f;
      if (tau - 1 >= minTau && cm[static_cast<size_t>(tau - 1)] < kYinThreshold && cm[static_cast<size_t>(tau)] >= cm[static_cast<size_t>(tau - 1)]) {
        candidate = tau - 1;       // the bottom of the first dip under the threshold; cm[tau], just computed, is its right neighbour
        break;
      }
    }
    if (candidate < 0) {                                     // no dip under the threshold: the deepest one, if it is convincing
      candidate = minTau;
      for (int tau = minTau + 1; tau <= maxTau; ++tau)
        if (cm[static_cast<size_t>(tau)] < cm[static_cast<size_t>(candidate)]) candidate = tau;
    }
    const float confidence = std::clamp(1.0f - cm[static_cast<size_t>(candidate)], 0.0f, 1.0f);
    if (confidence < kMinConfidence) return Silence();

    double tauFine = candidate;
    if (candidate > 1 && candidate <= maxTau) {
      const double l = cm[static_cast<size_t>(candidate - 1)], c = cm[static_cast<size_t>(candidate)], r = cm[static_cast<size_t>(candidate + 1)];
      const double den = l - 2.0 * c + r;
      if (std::abs(den) > 1e-12) tauFine += 0.5 * (l - r) / den;
    }
    double frequency = rate / tauFine;
    frequency = Refine(end, frequency);
    frequency = RefineFundamental(end, frequency);
    if (!std::isfinite(frequency) || frequency < kMinHz || frequency > kMaxHz) return Silence();

    history_[static_cast<size_t>(historyPos_)] = static_cast<float>(frequency);
    historyPos_ = (historyPos_ + 1) % static_cast<int>(history_.size());
    historyCount_ = std::min(historyCount_ + 1, static_cast<int>(history_.size()));
    auto sorted = history_;
    std::sort(sorted.begin(), sorted.begin() + historyCount_);
    const double median = sorted[static_cast<size_t>(historyCount_ / 2)];
    if (smoothed_ <= 0.0 || std::abs(1200.0 * std::log2(median / smoothed_)) > 80.0) smoothed_ = median;
    else smoothed_ += 0.28 * (median - smoothed_);

    Result r;
    r.frequency = static_cast<float>(smoothed_);
    r.confidence = confidence;
    r.valid = true;
    Publish(r);
  }

  // The period found at 24 kHz is good to about a third of a sample; on a lag of `mult` periods at the full rate the same error is
  // mult times smaller in relative terms, and the dip of the difference function is sharp enough to be read with a parabola.
  double Refine(uint64_t end, double coarseHz) const {
    const double fs = sampleRate_;
    const double t0 = fs / coarseHz;                                   // period, full-rate samples
    const int mult = std::clamp(static_cast<int>(std::lround(kRefineLagSeconds * fs / t0)), 1, std::max(1, static_cast<int>(t0 / 4.0)));
    const double m0 = mult * t0;
    const int span = static_cast<int>(std::max(2.0, std::min(0.4 * t0, mult * 1.5 + 3.0)));
    const int lo = static_cast<int>(std::lround(m0)) - span, hi = static_cast<int>(std::lround(m0)) + span;
    const int W = static_cast<int>(kRefineWindowSeconds * fs);
    if (end < static_cast<uint64_t>(W + hi) || lo < 2) return coarseHz;
    const uint64_t base = end - static_cast<uint64_t>(W) - static_cast<uint64_t>(hi);

    double mean = 0.0;
    for (int i = 0; i < W + hi; ++i) mean += Raw(base + static_cast<uint64_t>(i));
    mean /= (W + hi);
    std::vector<double>& v = refine_;
    v.assign(static_cast<size_t>(hi - lo + 1), 0.0);
    for (int lag = lo; lag <= hi; ++lag) {
      double num = 0.0, den = 0.0;
      for (int i = 0; i < W; ++i) {
        const double a = Raw(base + static_cast<uint64_t>(i)) - mean, b = Raw(base + static_cast<uint64_t>(i + lag)) - mean;
        num += (a - b) * (a - b);
        den += a * a + b * b;
      }
      v[static_cast<size_t>(lag - lo)] = den > 0.0 ? num / den : 1.0;
    }
    size_t best = 0;
    for (size_t i = 1; i < v.size(); ++i)
      if (v[i] < v[best]) best = i;
    if (best == 0 || best + 1 == v.size() || v[best] > 0.6) return coarseHz;     // no clean dip inside the range: keep the coarse reading
    double lag = static_cast<double>(lo) + static_cast<double>(best);
    const double l = v[best - 1], c = v[best], r = v[best + 1];
    const double den = l - 2.0 * c + r;
    if (std::abs(den) > 1e-18) lag += 0.5 * (l - r) / den;
    const double refined = fs * mult / lag;
    return std::abs(1200.0 * std::log2(refined / coarseHz)) < 60.0 ? refined : coarseHz;   // a refinement may not move the note
  }

  // The period of the whole waveform is not the pitch of the string: the partials of a real string are stretched a little (inharmonicity),
  // so the period follows them and reads sharp by one to five cents. What a tuner has to show is the FUNDAMENTAL. So, around the frequency
  // found above, the Hann-windowed spectrum is scanned in 2-cent steps and the peak is read with a parabola on the log magnitude. The window
  // is longer for low notes (about ten periods) so that the peak is narrower than the +-50 cents searched. If the fundamental does not stand
  // out of its surroundings (not there, or too wide to measure) the period-based reading is kept.
  double RefineFundamental(uint64_t end, double f) const {
    const double fs = sampleRate_;
    const int W = static_cast<int>(std::clamp(10.0 / f, 0.128, 0.30) * fs);
    if (end < static_cast<uint64_t>(W)) return f;
    const uint64_t base = end - static_cast<uint64_t>(W);
    double mean = 0.0;
    for (int i = 0; i < W; ++i) mean += Raw(base + static_cast<uint64_t>(i));
    mean /= W;
    std::vector<double>& seg = spectrum_;
    seg.resize(static_cast<size_t>(W));
    for (int i = 0; i < W; ++i) seg[static_cast<size_t>(i)] = (Raw(base + static_cast<uint64_t>(i)) - mean) * (0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * i / W));
    auto magnitude = [&](double hz) {
      const double w = 2.0 * 3.14159265358979323846 * hz / fs, c = std::cos(w), s = std::sin(w);
      double cr = 1.0, ci = 0.0, re = 0.0, im = 0.0;
      for (int i = 0; i < W; ++i) {
        const double v = seg[static_cast<size_t>(i)];
        re += v * cr;
        im += v * ci;
        const double nr = cr * c + ci * s, ni = ci * c - cr * s;           // e^(-j w (i+1))
        cr = nr;
        ci = ni;
      }
      return std::sqrt(re * re + im * im);
    };
    constexpr int kSteps = 25;                                                  // +-50 cents in 2-cent steps
    double m[2 * kSteps + 1];
    int best = 0;
    for (int i = -kSteps; i <= kSteps; ++i) {
      m[i + kSteps] = magnitude(f * std::pow(2.0, 2.0 * i / 1200.0));
      if (m[i + kSteps] > m[best]) best = i + kSteps;
    }
    if (best == 0 || best == 2 * kSteps) return f;
    if (m[best] < 3.0 * std::max(magnitude(f * 0.8), magnitude(f * 1.25))) return f;
    const double l = std::log(m[best - 1] + 1e-30), c = std::log(m[best] + 1e-30), r = std::log(m[best + 1] + 1e-30);
    const double den = l - 2.0 * c + r;
    const double offset = std::abs(den) > 1e-12 ? 0.5 * (l - r) / den : 0.0;            // in grid steps, within +-0.5
    return f * std::pow(2.0, 2.0 * ((best - kSteps) + std::clamp(offset, -0.5, 0.5)) / 1200.0);
  }

  void Silence() {
    history_.fill(0.0f);
    historyCount_ = historyPos_ = 0;
    Publish({});
  }

  void Publish(const Result& r) {
    pubValid_.store(false, std::memory_order_release);
    pubFrequency_.store(r.frequency, std::memory_order_relaxed);
    pubConfidence_.store(r.confidence, std::memory_order_relaxed);
    pubValid_.store(r.valid, std::memory_order_release);
  }

  double sampleRate_ = 48000.0;
  int decimation_ = 2;
  std::vector<float> ring_;
  std::atomic<uint64_t> total_{0};
  std::atomic_flag busy_ = ATOMIC_FLAG_INIT;
  uint64_t lastAnalysed_ = 0;

  std::array<float, kWindow> window_{};
  std::vector<float> cmnd_;
  mutable std::vector<double> refine_;
  mutable std::vector<double> spectrum_;
  std::array<float, 5> history_{};
  int historyCount_ = 0, historyPos_ = 0;
  double smoothed_ = 0.0;

  std::atomic<float> pubFrequency_{0.0f};
  std::atomic<float> pubConfidence_{0.0f};
  std::atomic<bool> pubValid_{false};
};
