// Tests of the tuner's pitch detector (engine/Tuner.h) on synthetic guitar-like notes.
// Windows: native\run_tests.bat, macOS/Linux: native/run_tests.sh. Exit code 0 = all passed.
#include "../Tuner.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

int g_failed = 0;
const double kPi = 3.14159265358979323846;

void check(bool ok, const std::string& what) {
  if (!ok) {
    ++g_failed;
    std::printf("  FAIL: %s\n", what.c_str());
  }
}

struct Scenario {
  const char* name;
  double noiseDb, humDb, inharm, glideCents;
  bool weakFundamental;
  double maxMeanCents, maxCents;
};

// A plucked string: partials 1..12 that fall off and decay faster the higher they are, slightly stretched (inharmonicity), the pitch
// starting `glideCents` sharp and relaxing in 150 ms, plus the interface's noise and a little mains hum.
std::vector<float> note(double f0, double secs, double fs, const Scenario& sc, unsigned seed, double detuneCents = 0.0) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> gauss(0.0, 1.0);
  std::uniform_real_distribution<double> uni(0.0, 6.2831853);
  const double f = f0 * std::pow(2.0, detuneCents / 1200.0);
  const int n = static_cast<int>(secs * fs);
  std::vector<double> x(static_cast<size_t>(n), 0.0);
  double phase = 0.0;
  std::vector<double> ph0(13);
  for (auto& p : ph0) p = uni(rng);
  for (int i = 0; i < n; ++i) {
    const double t = i / fs;
    const double inst = f * std::pow(2.0, (sc.glideCents * std::exp(-t / 0.15)) / 1200.0);
    phase += 2.0 * kPi * inst / fs;
    double s = 0.0;
    for (int h = 1; h <= 12; ++h) {
      const double fh = h * std::sqrt(1.0 + sc.inharm * h * h);
      const double amp = std::pow(h, -1.1) * ((sc.weakFundamental && h == 1) ? 0.12 : 1.0);
      s += amp * std::sin(phase * fh + ph0[static_cast<size_t>(h)]) * std::exp(-t * (0.8 + 0.4 * h));
    }
    x[static_cast<size_t>(i)] = s;
  }
  double peak = 0.0;
  for (double v : x) peak = std::max(peak, std::fabs(v));
  std::vector<float> out(static_cast<size_t>(n));
  const double noise = std::pow(10.0, sc.noiseDb / 20.0), hum = std::pow(10.0, sc.humDb / 20.0);
  for (int i = 0; i < n; ++i) {
    const double t = i / fs;
    out[static_cast<size_t>(i)] = static_cast<float>(0.3 * x[static_cast<size_t>(i)] / peak + noise * gauss(rng) + hum * std::sin(2 * kPi * 50 * t) +
                                                   0.4 * hum * std::sin(2 * kPi * 150 * t));
  }
  return out;
}

// Feeds `x` the way the audio thread would (blocks of `block`), polling every `pollEvery` blocks like the UI does; returns the readings
// taken from 0.5 s on (the first part of a pluck is the attack, where nobody reads the needle).
std::vector<PitchTuner::Result> run(const std::vector<float>& x, double fs, int block, int pollEvery, double fromSeconds = 0.5) {
  PitchTuner t;
  t.Reset(fs);
  std::vector<PitchTuner::Result> readings;
  int blocks = 0;
  for (size_t pos = 0; pos + block <= x.size(); pos += static_cast<size_t>(block)) {
    t.Process(x.data() + pos, block);
    if (++blocks % pollEvery == 0) {
      const auto r = t.Poll();
      if (static_cast<double>(pos) / fs >= fromSeconds) readings.push_back(r);
    }
  }
  return readings;
}

double cents(double f, double ref) { return 1200.0 * std::log2(f / ref); }

void accuracy() {
  const std::vector<std::pair<const char*, double>> notes = {{"B0", 30.8677}, {"E1", 41.2034}, {"E2", 82.4069}, {"A2", 110.0}, {"D3", 146.8324},
                                                             {"G3", 195.9977}, {"B3", 246.9417}, {"E4", 329.6276}, {"E5", 659.2551}, {"C6", 1046.502}};
  const Scenario scenarios[] = {
      {"clean", -80, -90, 0.0, 0.0, false, 0.1, 0.25},
      {"typical (noise, hum, inharmonic strings)", -50, -50, 1e-4, 0.0, false, 0.4, 1.0},
      {"pluck glide (8 cents sharp, settling)", -50, -50, 1e-4, 8.0, false, 1.0, 3.0},
      {"weak fundamental", -55, -55, 2e-4, 0.0, true, 0.8, 2.5},
  };
  for (const auto& sc : scenarios) {
    for (double fs : {44100.0, 48000.0, 96000.0}) {
      double worstMean = 0.0, worst = 0.0;
      int misses = 0, total = 0;
      for (const auto& [name, f0] : notes) {
        double sum = 0.0;
        int count = 0;
        for (unsigned seed = 1; seed <= 3; ++seed) {
          const double detune = seed == 2 ? 23.0 : seed == 3 ? -31.0 : 0.0;            // the note is not always in tune
          const auto x = note(f0, 1.4, fs, sc, seed, detune);
          const double target = f0 * std::pow(2.0, detune / 1200.0);
          for (const auto& r : run(x, fs, 256, 12)) {                                       // a poll about every 64 ms
            ++total;
            if (!r.valid) { ++misses; continue; }
            const double e = std::fabs(cents(r.frequency, target));
            sum += e;
            ++count;
            worst = std::max(worst, e);
            if (e > sc.maxCents) std::printf("    %s %.0f Hz %s: %.2f cents off at %.3f Hz\n", name, fs, sc.name, e, r.frequency);
          }
        }
        if (count) worstMean = std::max(worstMean, sum / count);
      }
      std::printf("  %-44s %6.0f Hz: worst mean %.2f cent, worst %.2f cent, %d/%d without reading\n", sc.name, fs, worstMean, worst, misses, total);
      check(worstMean <= sc.maxMeanCents, std::string(sc.name) + ": mean error within bounds");
      check(worst <= sc.maxCents, std::string(sc.name) + ": no reading further than the bound");
      check(misses * 4 <= total, std::string(sc.name) + ": the note is read most of the time");
    }
  }
}

void stays_silent() {
  const double fs = 48000.0;
  std::mt19937 rng(7);
  std::normal_distribution<double> gauss(0.0, 1.0);
  {
    std::vector<float> x(static_cast<size_t>(fs * 2));                  // digital silence
    const auto r = run(x, fs, 256, 6, 0.1);
    int valid = 0;
    for (const auto& v : r) valid += v.valid;
    check(valid == 0, "digital silence gives no reading");
  }
  {
    std::vector<float> x(static_cast<size_t>(fs * 2));                  // the interface's own noise (-70 dBFS rms)
    for (auto& v : x) v = static_cast<float>(3e-4 * gauss(rng));
    int valid = 0;
    for (const auto& v : run(x, fs, 256, 6, 0.1)) valid += v.valid;
    check(valid == 0, "background noise gives no reading");
  }
  {
    std::vector<float> x(static_cast<size_t>(fs * 2));                  // loud white noise: loud, but no pitch
    for (auto& v : x) v = static_cast<float>(0.1 * gauss(rng));
    int valid = 0;
    for (const auto& v : run(x, fs, 256, 6, 0.1)) valid += v.valid;
    check(valid <= 1, "white noise gives no (stable) reading");
  }
}

void follows_the_note() {
  // E2 for one second, then A2: the display must jump to the new note at once, not glide over it.
  const double fs = 48000.0;
  const Scenario sc{"clean", -80, -90, 0.0, 0.0, false, 0, 0};
  auto a = note(82.4069, 1.0, fs, sc, 1), b = note(110.0, 1.0, fs, sc, 2);
  a.insert(a.end(), b.begin(), b.end());
  PitchTuner t;
  t.Reset(fs);
  int firstA2 = -1, blocks = 0;
  for (size_t pos = 0; pos + 256 <= a.size(); pos += 256) {
    t.Process(a.data() + pos, 256);
    if (++blocks % 6 == 0) {
      const auto r = t.Poll();
      if (pos / fs > 1.0 && r.valid && std::fabs(cents(r.frequency, 110.0)) < 3.0 && firstA2 < 0) firstA2 = static_cast<int>(pos);
    }
  }
  const double after = firstA2 < 0 ? 99.0 : firstA2 / fs - 1.0;
  std::printf("  the new note is shown %.3f s after it starts\n", after);
  check(firstA2 >= 0 && after < 0.45, "a new note is picked up within 0.45 s");
}

void polling_rate_does_not_matter() {
  const double fs = 48000.0;
  const Scenario sc{"typical", -50, -50, 1e-4, 0.0, false, 0, 0};
  const auto x = note(196.0, 1.5, fs, sc, 4);
  double mean[3] = {};
  int idx = 0;
  for (int every : {2, 12, 90}) {                      // 10 ms, 64 ms, 0.5 s between polls
    double sum = 0.0;
    int n = 0;
    for (const auto& r : run(x, fs, 256, every, 0.6))
      if (r.valid) { sum += std::fabs(cents(r.frequency, 195.9977)); ++n; }
    mean[idx++] = n ? sum / n : 99.0;
  }
  std::printf("  mean error with a poll every 10 ms / 64 ms / 0.5 s: %.2f / %.2f / %.2f cent\n", mean[0], mean[1], mean[2]);
  for (double m : mean) check(m < 2.0, "slow or fast polling reads the same pitch");
}

}  // namespace

int main() {
  std::printf("accuracy on synthetic notes\n");
  accuracy();
  std::printf("silence and noise\n");
  stays_silent();
  std::printf("note changes\n");
  follows_the_note();
  std::printf("polling rate\n");
  polling_rate_does_not_matter();
  std::printf(g_failed ? "\n%d check(s) FAILED\n" : "\nall tuner checks passed\n", g_failed);
  return g_failed ? 1 : 0;
}
