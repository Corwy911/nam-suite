// Plain assert-based tests for RTEngine's gain stage, mirroring tests/test_capture_generators.py's PassthroughGenTests
// (same properties, same numbers where they overlap) - no test framework, matching this project's "only what's
// needed" build. Run with native/run_tests.sh.

#include "../RTEngine.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <atomic>
#include <cstdio>
#include <fstream>
#include <thread>
#include <string>
#include <vector>

namespace {

double fromDb(double db) { return std::pow(10.0, db / 20.0); }
double toDb(double x) { return 20.0 * std::log10(std::max(x, 1e-12)); }

std::vector<float> block(RTEngine* e, int n, float srcAmp) {
  std::vector<float> in(n, srcAmp), out(n, 0.0f);
  rt_process(e, in.data(), out.data(), n);
  return out;
}

void test_starts_silent_and_glides_to_target() {
  RTEngine* e = rt_create(48000.0);
  rt_set_gain_db(e, 0.0); // unity gain
  std::vector<float> all;
  for (int i = 0; i < 60; ++i) {
    auto b = block(e, 256, 0.2f);
    all.insert(all.end(), b.begin(), b.end());
  }
  assert(std::fabs(all.front()) < 1e-3);              // no step at the very first sample
  assert(std::fabs(all.back() - 0.2f) < 0.01f);         // settles at the target
  rt_destroy(e);
  std::puts("ok: starts_silent_and_glides_to_target");
}

void test_settled_gain_multiplies_the_input() {
  RTEngine* e = rt_create(48000.0);
  rt_set_gain_db(e, -7.32); // the manual's real unity-gain example
  for (int i = 0; i < 60; ++i) block(e, 256, 0.3f);
  auto y = block(e, 256, 0.3f);
  double expected = 0.3 * fromDb(-7.32);
  for (float v : y) assert(std::fabs(v - expected) < 1e-4);
  rt_destroy(e);
  std::puts("ok: settled_gain_multiplies_the_input");
}

void test_ceiling_clips() {
  RTEngine* e = rt_create(48000.0);
  rt_set_ceiling_db(e, -1.0);
  rt_set_gain_db(e, 20.0);
  for (int i = 0; i < 60; ++i) block(e, 256, 0.9f);
  auto y = block(e, 256, 0.9f);
  double ceiling = fromDb(-1.0);
  for (float v : y) assert(std::fabs(v) <= ceiling + 1e-6);
  rt_destroy(e);
  std::puts("ok: ceiling_clips");
}

void test_stop_fades_to_silence_and_finishes() {
  RTEngine* e = rt_create(48000.0);
  rt_set_gain_db(e, 0.0);
  for (int i = 0; i < 60; ++i) block(e, 256, 0.5f);
  rt_stop(e);
  std::vector<float> last;
  for (int i = 0; i < 60 && !rt_is_finished(e); ++i) last = block(e, 256, 0.5f);
  assert(rt_is_finished(e));
  assert(std::fabs(last.back()) < 0.01f);
  auto silent = block(e, 256, 0.5f); // once finished, further calls are silent
  for (float v : silent) assert(v == 0.0f);
  rt_destroy(e);
  std::puts("ok: stop_fades_to_silence_and_finishes");
}

void test_gain_change_while_running_glides() {
  RTEngine* e = rt_create(48000.0);
  rt_set_gain_db(e, -20.0);
  for (int i = 0; i < 60; ++i) block(e, 256, 0.3f);
  auto before = block(e, 256, 0.3f);
  rt_set_gain_db(e, 0.0);
  std::vector<float> after;
  for (int i = 0; i < 60; ++i) {
    auto b = block(e, 256, 0.3f);
    after.insert(after.end(), b.begin(), b.end());
  }
  double maxJump = 0.0;
  std::vector<float> seam(before);
  seam.insert(seam.end(), after.begin(), after.end());
  for (size_t i = 1; i < seam.size(); ++i) maxJump = std::max(maxJump, static_cast<double>(std::fabs(seam[i] - seam[i - 1])));
  assert(maxJump < fromDb(-20.0) * 0.5); // no jump at the seam where the target changed
  assert(std::fabs(after.back() - 0.3f) < 0.005f);
  rt_destroy(e);
  std::puts("ok: gain_change_while_running_glides");
}

void test_pre_gain_multiplies_before_the_output_gain() {
  // No model/IR loaded: pre-gain and the (post) output gain both apply, in series - this is the
  // chain's own contract (input -> pre-gain -> [NAM] -> [IR] -> output gain), not the settled-output
  // test above, which only exercises the default (unity) pre-gain.
  RTEngine* e = rt_create(48000.0);
  rt_set_pre_gain_db(e, 6.0);
  rt_set_gain_db(e, -3.0);
  for (int i = 0; i < 60; ++i) block(e, 256, 0.2f);
  auto y = block(e, 256, 0.2f);
  double expected = 0.2 * fromDb(6.0) * fromDb(-3.0);
  for (float v : y) assert(std::fabs(v - expected) < 1e-4);
  rt_destroy(e);
  std::puts("ok: pre_gain_multiplies_before_the_output_gain");
}

void test_load_model_missing_file_fails_cleanly() {
  RTEngine* e = rt_create(48000.0);
  assert(rt_model_loaded(e) == 0);
  int ok = rt_load_model(e, "/nonexistent/path/definitely-not-a-model.nam");
  assert(ok == 0);
  assert(rt_model_loaded(e) == 0);
  assert(std::string(rt_last_error(e)).size() > 0);
  // the engine keeps working as a plain passthrough after a failed load, no half-loaded state
  rt_set_gain_db(e, 0.0);
  auto y = block(e, 256, 0.2f);
  for (int i = 0; i < 60; ++i) y = block(e, 256, 0.2f);
  assert(std::fabs(y.back() - 0.2f) < 0.01f);
  rt_destroy(e);
  std::puts("ok: load_model_missing_file_fails_cleanly");
}

void test_ir_unit_impulse_is_transparent() {
  // A single-sample IR of amplitude 1.0 is the identity element of convolution: loading it must not
  // change the signal (and, unlike a real cabinet IR, must not add any latency either - verified by
  // this exact-value comparison, not just an energy check).
  RTEngine* e = rt_create(48000.0);
  rt_set_gain_db(e, 0.0);
  std::vector<float> ir = {1.0f};
  assert(rt_ir_loaded(e) == 0);
  int ok = rt_load_ir(e, ir.data(), static_cast<int>(ir.size()));
  assert(ok == 1);
  assert(rt_ir_loaded(e) == 1);
  for (int i = 0; i < 60; ++i) block(e, 256, 0.3f);
  auto y = block(e, 256, 0.3f);
  for (float v : y) assert(std::fabs(v - 0.3f) < 1e-4);
  rt_unload_ir(e);
  assert(rt_ir_loaded(e) == 0);
  rt_destroy(e);
  std::puts("ok: ir_unit_impulse_is_transparent");
}

void test_gate_mutes_below_threshold_passes_above() {
  RTEngine* e = rt_create(48000.0);
  rt_set_gain_db(e, 0.0);
  rt_set_gate_threshold_db(e, -40.0);
  for (int i = 0; i < 60; ++i) block(e, 256, 0.3f); // let the output-gain ramp settle first (unrelated to the gate)
  auto loud = block(e, 256, 0.3f); // -10.5 dBFS: well above -40 dB, the gate must be fully open
  for (float v : loud) assert(std::fabs(v - 0.3f) < 0.02f);
  for (int i = 0; i < 60; ++i) block(e, 256, 0.001f); // -60 dBFS: well below -40 dB - give hold+release (~200 ms) time to close
  auto quiet = block(e, 256, 0.001f);
  for (float v : quiet) assert(std::fabs(v) < 0.0003f); // gated well down from the 0.001 input
  rt_destroy(e);
  std::puts("ok: gate_mutes_below_threshold_passes_above");
}

void test_eq_mid_boost_increases_amplitude_at_center_frequency() {
  const double sr = 48000.0;
  const double freq = 800.0; // matches RTEngine.cpp's own peaking-band center
  RTEngine* e = rt_create(sr);
  rt_set_gain_db(e, 0.0);
  rt_set_eq_mid_db(e, 12.0);
  std::vector<float> in(4096), out(4096);
  for (size_t i = 0; i < in.size(); ++i) in[i] = 0.2f * static_cast<float>(std::sin(2.0 * M_PI * freq * static_cast<double>(i) / sr));
  for (int i = 0; i < 20; ++i) rt_process(e, in.data(), out.data(), static_cast<int>(in.size())); // reach the biquad's steady state
  rt_process(e, in.data(), out.data(), static_cast<int>(in.size()));
  float outPeak = 0.0f;
  for (size_t i = in.size() / 2; i < out.size(); ++i) outPeak = std::max(outPeak, std::fabs(out[i])); // skip any leftover transient
  double ratio = static_cast<double>(outPeak) / 0.2;
  double expected = fromDb(12.0); // a peaking filter's gain exactly at its own center frequency
  assert(std::fabs(ratio - expected) < 0.15);
  rt_destroy(e);
  std::puts("ok: eq_mid_boost_increases_amplitude_at_center_frequency");
}

void test_hot_swapping_the_ir_while_audio_runs_is_safe() {
  // The audio thread keeps calling rt_process while the control thread loads / replaces / unloads the IR over and
  // over: the old convolver must never be freed under a running block (this would crash or trip a sanitizer), and the
  // output must stay finite the whole time.
  RTEngine* e = rt_create(48000.0);
  rt_set_gain_db(e, 0.0);
  std::atomic<bool> run{true};
  std::atomic<bool> finite{true};
  std::thread audio([&] {
    std::vector<float> in(64, 0.1f), out(64);
    while (run.load()) {
      rt_process(e, in.data(), out.data(), 64);
      for (float v : out) if (!std::isfinite(v)) finite = false;
    }
  });
  std::vector<float> ir(2000);
  for (size_t i = 0; i < ir.size(); ++i) ir[i] = (i == 0) ? 1.0f : 0.0005f * static_cast<float>(std::sin(0.1 * static_cast<double>(i)));
  for (int i = 0; i < 300; ++i) {
    assert(rt_load_ir(e, ir.data(), static_cast<int>(ir.size())) == 1);
    if (i % 3 == 0) rt_unload_ir(e);
  }
  run = false;
  audio.join();
  assert(finite.load());
  rt_destroy(e);
  std::puts("ok: hot_swapping_the_ir_while_audio_runs_is_safe");
}

void test_ir_level_knobs_scale_before_and_after_the_ir() {
  RTEngine* e = rt_create(48000.0);
  rt_set_gain_db(e, 0.0);
  std::vector<float> ir = {1.0f};                 // unit impulse: the IR itself changes nothing
  assert(rt_load_ir(e, ir.data(), 1) == 1);
  rt_set_ir_in_gain_db(e, 6.0);
  rt_set_ir_out_gain_db(e, -3.0);
  for (int i = 0; i < 60; ++i) block(e, 256, 0.1f);
  auto y = block(e, 256, 0.1f);
  double expected = 0.1 * fromDb(6.0) * fromDb(-3.0);
  for (float v : y) assert(std::fabs(v - expected) < 1e-4);
  rt_unload_ir(e);                                  // no IR: the knobs do nothing at all
  for (int i = 0; i < 60; ++i) block(e, 256, 0.1f);
  y = block(e, 256, 0.1f);
  for (float v : y) assert(std::fabs(v - 0.1f) < 1e-4);
  rt_destroy(e);
  std::puts("ok: ir_level_knobs_scale_before_and_after_the_ir");
}

void test_last_peak_db_tracks_the_output() {
  RTEngine* e = rt_create(48000.0);
  rt_set_gain_db(e, 0.0);
  for (int i = 0; i < 60; ++i) block(e, 256, 0.25f);
  auto y = block(e, 256, 0.25f);
  double reported = rt_last_peak_db(e);
  float peakSample = *std::max_element(y.begin(), y.end(), [](float a, float b) { return std::fabs(a) < std::fabs(b); });
  double measured = toDb(std::fabs(peakSample));
  assert(std::fabs(reported - measured) < 0.5);
  rt_destroy(e);
  std::puts("ok: last_peak_db_tracks_the_output");
}


// ------------------------------------------------------------------------------------------------ the player (A/B)

// Runs `blocks` blocks of `n` frames with constant inputs, returns the LAST block's monitor mix (and feed).
struct PlayerRun { std::vector<float> feed, mon; };
PlayerRun runPlayer(RTPlayer* p, int blocks, int n, float guitar, float ret) {
  std::vector<float> g(n, guitar), r(n, ret), f(n), m(n);
  for (int i = 0; i < blocks; ++i) rt_player_process(p, g.data(), r.data(), f.data(), m.data(), n);
  return {f, m};
}

RTPlayer* makePlayer(double feedDb, double ampDb, double legBPostDb) {
  RTPlayer* p = rt_player_create(48000.0);
  rt_set_gain_db(rt_player_engine(p, RT_LEG_FEED), feedDb);
  rt_set_gain_db(rt_player_engine(p, RT_LEG_A), ampDb);
  rt_set_gain_db(rt_player_engine(p, RT_LEG_B), legBPostDb);
  return p;
}

void test_player_feed_is_the_guitar_at_unity_gain() {
  RTPlayer* p = makePlayer(-7.32, 0.0, 0.0);              // feed = Ci_guitar - Co, the manual's example
  auto r = runPlayer(p, 80, 256, 0.3f, 0.0f);
  for (float v : r.feed) assert(std::fabs(v - 0.3 * fromDb(-7.32)) < 1e-4);
  rt_player_destroy(p);
  std::puts("ok: player_feed_is_the_guitar_at_unity_gain");
}

void test_player_selects_between_the_real_gear_and_the_model() {
  RTPlayer* p = makePlayer(0.0, -6.0, 3.0);               // A: return x -6 dB ; B (no model, no IR): guitar x +3 dB
  rt_player_select(p, 0);
  auto a = runPlayer(p, 80, 256, 0.2f, 0.4f);
  for (float v : a.mon) assert(std::fabs(v - 0.4 * fromDb(-6.0)) < 1e-4);   // only A is heard: the return, with its gain
  rt_player_select(p, 1);
  auto b = runPlayer(p, 80, 256, 0.2f, 0.4f);
  for (float v : b.mon) assert(std::fabs(v - 0.2 * fromDb(3.0)) < 1e-4);    // only B: the guitar through the model's chain
  assert(rt_player_selected(p) == 1);
  rt_player_destroy(p);
  std::puts("ok: player_selects_between_the_real_gear_and_the_model");
}

void test_player_switch_is_a_short_click_free_cross_fade() {
  RTPlayer* p = makePlayer(0.0, 0.0, 0.0);
  rt_player_select(p, 0);
  runPlayer(p, 80, 256, 0.5f, 0.1f);                      // A = 0.1, B = 0.5 (both settled)
  rt_player_select(p, 1);
  std::vector<float> g(64, 0.5f), r(64, 0.1f), f(64), m(64), all;
  for (int i = 0; i < 20; ++i) {                          // 1280 frames: the fade (480) plus a settled tail
    rt_player_process(p, g.data(), r.data(), f.data(), m.data(), 64);
    all.insert(all.end(), m.begin(), m.end());
  }
  double maxJump = 0.0;
  for (size_t i = 1; i < all.size(); ++i) maxJump = std::max(maxJump, static_cast<double>(std::fabs(all[i] - all[i - 1])));
  assert(maxJump < 0.01);                                 // no step anywhere (a hard switch would jump by 0.4)
  assert(std::fabs(all.front() - 0.1f) < 0.02f);          // starts on A
  assert(std::fabs(all.back() - 0.5f) < 1e-3f);           // ends on B exactly
  assert(std::fabs(all[200] - 0.1f) > 0.001f);            // and it really is a fade, not an instant jump at the start
  rt_player_destroy(p);
  std::puts("ok: player_switch_is_a_short_click_free_cross_fade");
}

void test_player_b_trim_and_the_ceiling() {
  RTPlayer* p = makePlayer(0.0, 0.0, 0.0);
  rt_player_select(p, 1);
  rt_player_set_b_trim_db(p, -6.0);
  auto r = runPlayer(p, 80, 256, 0.4f, 0.0f);
  for (float v : r.mon) assert(std::fabs(v - 0.4 * fromDb(-6.0)) < 1e-4);
  rt_player_set_b_trim_db(p, 20.0);                        // way too hot: never crosses the engine's -1 dBFS ceiling
  r = runPlayer(p, 200, 256, 0.9f, 0.0f);
  for (float v : r.mon) assert(std::fabs(v) <= fromDb(-1.0) + 1e-6);
  rt_player_destroy(p);
  std::puts("ok: player_b_trim_and_the_ceiling");
}

void test_player_b_delay_lines_the_model_up_with_the_gear() {
  RTPlayer* p = makePlayer(0.0, 0.0, 0.0);
  rt_player_select(p, 1);
  runPlayer(p, 80, 256, 0.01f, 0.0f);                      // settle on a faint hum (true silence would close the noise gate)
  rt_player_set_b_delay(p, 100);
  std::vector<float> g(512, 0.01f), r(512, 0.0f), f(512), m(512);
  g[10] = 0.5f;                                            // one impulse at sample 10
  rt_player_process(p, g.data(), r.data(), f.data(), m.data(), 512);
  size_t at = 0;
  for (size_t i = 0; i < m.size(); ++i) if (std::fabs(m[i]) > 0.2f) { at = i; break; }
  assert(at == 110);                                       // 10 + the 100 samples of delay
  rt_player_destroy(p);
  std::puts("ok: player_b_delay_lines_the_model_up_with_the_gear");
}

void test_player_peaks_show_each_leg_even_when_not_heard() {
  RTPlayer* p = makePlayer(0.0, 0.0, 0.0);
  rt_player_select(p, 0);
  runPlayer(p, 80, 256, 0.5f, 0.1f);
  assert(std::fabs(rt_player_peak_a_db(p) - toDb(0.1)) < 0.2);
  assert(std::fabs(rt_player_peak_b_db(p) - toDb(0.5)) < 0.2);         // B is not heard, but its level is still measured
  assert(std::fabs(rt_player_peak_out_db(p) - toDb(0.1)) < 0.2);
  rt_player_destroy(p);
  std::puts("ok: player_peaks_show_each_leg_even_when_not_heard");
}

void test_player_stop_fades_every_part_and_finishes() {
  RTPlayer* p = makePlayer(0.0, 0.0, 0.0);
  runPlayer(p, 80, 256, 0.3f, 0.3f);
  assert(rt_player_is_finished(p) == 0);
  rt_player_stop(p);
  auto r = runPlayer(p, 100, 256, 0.3f, 0.3f);
  assert(rt_player_is_finished(p) == 1);
  for (float v : r.feed) assert(v == 0.0f);
  for (float v : r.mon) assert(v == 0.0f);
  rt_player_destroy(p);
  std::puts("ok: player_stop_fades_every_part_and_finishes");
}

void test_player_ir_on_the_gear_leg_only_changes_that_leg() {
  // an IR that is a unit impulse delayed by nothing is transparent: loaded on A it must leave A's level alone and not touch B
  RTPlayer* p = makePlayer(0.0, 0.0, 0.0);
  std::vector<float> ir(64, 0.0f); ir[0] = 0.5f;           // a cabinet that is just -6 dB
  assert(rt_load_ir(rt_player_engine(p, RT_LEG_A), ir.data(), static_cast<int>(ir.size())) == 1);
  rt_player_select(p, 0);
  auto a = runPlayer(p, 100, 256, 0.2f, 0.4f);
  for (float v : a.mon) assert(std::fabs(v - 0.4 * 0.5) < 1e-3);
  rt_player_select(p, 1);
  auto b = runPlayer(p, 100, 256, 0.2f, 0.4f);
  for (float v : b.mon) assert(std::fabs(v - 0.2) < 1e-4);                // B has no IR: untouched
  rt_player_destroy(p);
  std::puts("ok: player_ir_on_the_gear_leg_only_changes_that_leg");
}

} // namespace

void test_tuner_reads_the_raw_input_of_an_engine_and_of_the_players_feed() {
  // 110 Hz, 5 cents sharp, with a second partial: through the feed engine of a player and through a plain engine, whatever the gains do
  const double f0 = 110.0 * std::pow(2.0, 5.0 / 1200.0);
  auto feedThrough = [&](RTEngine* e) {
    for (int i = 0; i < 90; ++i) {
      std::vector<float> in(512), out(512);
      for (int k = 0; k < 512; ++k) {
        const double t = (i * 512.0 + k) / 48000.0;
        in[static_cast<size_t>(k)] = static_cast<float>(0.2 * std::sin(2 * M_PI * f0 * t) + 0.08 * std::sin(4 * M_PI * f0 * t));
      }
      rt_process(e, in.data(), out.data(), 512);
      if (i % 8 == 0) { float hz, conf; rt_tuner_read(e, &hz, &conf); }
    }
    float hz = 0.0f, conf = 0.0f;
    const int ok = rt_tuner_read(e, &hz, &conf);
    assert(ok == 1);
    assert(std::fabs(1200.0 * std::log2(hz / 110.0) - 5.0) < 0.5);
    assert(conf > 0.72f);
  };
  RTEngine* e = rt_create(48000.0);
  rt_set_pre_gain_db(e, -20.0);                // the tuner hears the input as it comes in, not as the chain scales it
  rt_set_gain_db(e, -30.0);
  feedThrough(e);
  rt_destroy(e);
  RTPlayer* p = rt_player_create(48000.0);
  feedThrough(rt_player_engine(p, RT_LEG_FEED));
  rt_player_destroy(p);
  // silence: no note
  RTEngine* q = rt_create(48000.0);
  for (int i = 0; i < 90; ++i) block(q, 512, 0.0f);
  float hz = 1.0f, conf = 1.0f;
  assert(rt_tuner_read(q, &hz, &conf) == 0 && hz == 0.0f && conf == 0.0f);
  rt_destroy(q);
  std::puts("ok: tuner_reads_the_raw_input_of_an_engine_and_of_the_players_feed");
}

// ------------------------------------------------------------------------------------------------ A2 / slimmable models

// A SlimmableContainer like the official trainer 0.13 writes, with two one-tap Linear submodels instead of WaveNets: the small one
// multiplies by 0.25, the full one by 1.0 - so the output says which one is playing.
std::string writeSlimmableModel() {
  const std::string path = "build/test-slimmable.nam";
  std::ofstream f(path);
  auto linear = [](double w) {
    return std::string(R"({"version": "0.7.0", "architecture": "Linear", "config": {"receptive_field": 1, "bias": false}, "weights": [)") +
           std::to_string(w) + R"(], "sample_rate": 48000})";
  };
  f << R"({"version": "0.7.0", "architecture": "SlimmableContainer", "config": {"submodels": [)"
    << R"({"max_value": 0.5, "model": )" << linear(0.25) << "}, "
    << R"({"max_value": 1.0, "model": )" << linear(1.0) << "}"
    << R"(]}, "weights": [], "sample_rate": 48000})";
  return path;
}

void test_slimmable_model_switches_between_lite_and_full() {
  RTEngine* e = rt_create(48000.0);
  rt_set_gain_db(e, 0.0);
  assert(rt_model_sizes(e) == 0 && rt_model_size(e) == -1);
  int ok = rt_load_model(e, writeSlimmableModel().c_str());
  if (!ok) { std::fprintf(stderr, "load failed: %s\n", rt_last_error(e)); std::fflush(stderr); }
  assert(ok == 1);
  assert(rt_model_sizes(e) == 2);
  assert(rt_model_size(e) == 1);                                 // starts on the full model
  std::vector<float> b;
  for (int i = 0; i < 60; ++i) b = block(e, 256, 0.2f);
  assert(std::fabs(b.back() - 0.2f) < 0.01f);
  assert(rt_set_model_size(e, 0) == 0);                          // Lite
  for (int i = 0; i < 10; ++i) b = block(e, 256, 0.2f);
  assert(std::fabs(b.back() - 0.05f) < 0.005f);
  assert(rt_set_model_size(e, 7) == 1);                          // clamped to the full one
  for (int i = 0; i < 10; ++i) b = block(e, 256, 0.2f);
  assert(std::fabs(b.back() - 0.2f) < 0.01f);
  rt_unload_model(e);
  assert(rt_model_sizes(e) == 0);
  rt_destroy(e);
  std::puts("ok: slimmable_model_switches_between_lite_and_full");
}

void test_player_gate_also_closes_the_real_gear_legs() {
  // the gate is set on leg B only; with the guitar below it, the feed to the amp and the amp's return go quiet too (amp hiss included)
  RTPlayer* p = makePlayer(0.0, 0.0, 0.0);
  rt_player_select(p, 0);
  auto open = runPlayer(p, 60, 256, 0.01f, 0.1f);                 // gate off (-100 dB): the real gear is left alone
  assert(std::fabs(open.feed.back() - 0.01f) < 1e-4);
  assert(std::fabs(open.mon.back() - 0.1f) < 1e-3);
  rt_set_gate_threshold_db(rt_player_engine(p, RT_LEG_B), -20.0);
  auto quiet = runPlayer(p, 200, 256, 0.01f, 0.1f);              // guitar at -40 dBFS, under the -20 dB gate; the "amp hiss" on the return
  assert(std::fabs(quiet.feed.back()) < 1e-4);
  assert(std::fabs(quiet.mon.back()) < 1e-3);
  auto loud = runPlayer(p, 60, 256, 0.3f, 0.1f);                 // playing: open again
  assert(std::fabs(loud.feed.back() - 0.3f) < 1e-3);
  assert(std::fabs(loud.mon.back() - 0.1f) < 1e-3);
  rt_player_destroy(p);
  std::puts("ok: player_gate_also_closes_the_real_gear_legs");
}

// ------------------------------------------------------------------------------------------------ "Test Pedali"

// A one-tap Linear model (output = weight * input), so what each stage does to the level can be read off the numbers.
std::string writeLinearModel(const std::string& name, double weight) {
  const std::string path = "build/" + name + ".nam";
  std::ofstream f(path);
  f << R"({"version": "0.7.0", "architecture": "Linear", "config": {"receptive_field": 1, "bias": false}, "weights": [)"
    << std::to_string(weight) << R"(], "sample_rate": 48000})";
  return path;
}

void test_pedal_stage_sits_in_front_of_the_amp_on_leg_b_and_leg_a_has_its_own_amp() {
  RTPlayer* p = makePlayer(0.0, 0.0, 0.0);
  assert(rt_player_pedal_active(p) == 0);
  RTEngine* pedal = rt_player_engine(p, RT_LEG_PEDAL);
  RTEngine* legA = rt_player_engine(p, RT_LEG_A);
  RTEngine* legB = rt_player_engine(p, RT_LEG_B);
  assert(rt_load_model(pedal, writeLinearModel("test-pedal", 0.5).c_str()) == 1);
  assert(rt_load_model(legB, writeLinearModel("test-amp", 2.0).c_str()) == 1);
  assert(rt_load_model(legA, writeLinearModel("test-amp-a", 2.0).c_str()) == 1);
  assert(rt_player_pedal_active(p) == 1);
  rt_set_gain_db(pedal, 0.0);
  rt_player_select(p, 1);                                          // B: guitar 0.1 -> pedal x0.5 -> amp x2 = 0.1
  auto b = runPlayer(p, 200, 256, 0.1f, 0.05f);
  for (float v : b.mon) assert(std::fabs(v - 0.1f) < 2e-3);
  rt_player_select(p, 0);                                          // A: the real pedal's return 0.05 -> amp copy x2 = 0.1: the same
  auto a = runPlayer(p, 200, 256, 0.1f, 0.05f);
  for (float v : a.mon) assert(std::fabs(v - 0.1f) < 2e-3);
  rt_player_destroy(p);
  std::puts("ok: pedal_stage_sits_in_front_of_the_amp_on_leg_b_and_leg_a_has_its_own_amp");
}

void test_pedal_levels_scale_between_the_stages() {
  RTPlayer* p = makePlayer(0.0, 0.0, 0.0);
  RTEngine* pedal = rt_player_engine(p, RT_LEG_PEDAL);
  RTEngine* legB = rt_player_engine(p, RT_LEG_B);
  assert(rt_load_model(pedal, writeLinearModel("test-pedal", 1.0).c_str()) == 1);
  assert(rt_load_model(legB, writeLinearModel("test-amp", 1.0).c_str()) == 1);
  rt_set_pre_gain_db(pedal, -6.0);                                 // the pedal's input level
  rt_set_gain_db(pedal, 0.0);
  rt_set_pre_gain_db(legB, 12.0);                                  // pedal output scale -> amp input scale
  rt_player_select(p, 1);
  auto r = runPlayer(p, 200, 256, 0.1f, 0.0f);
  assert(std::fabs(r.mon.back() - 0.1 * fromDb(-6.0) * fromDb(12.0)) < 2e-3);
  // the pedal's own output may go above the amp stage's -1 dBFS ceiling (it is only a level on the pedal's scale, scaled again before the amp):
  // 0.5 * -6 dB = 0.25, +16 dB = 1.58 out of the pedal stage, then -10 dB on leg B's output = 0.50 - if the pedal clipped at -1 dBFS it would be 0.28
  rt_set_gain_db(pedal, 16.0);
  rt_set_pre_gain_db(legB, 0.0);
  rt_set_gain_db(legB, -10.0);
  auto loud = runPlayer(p, 250, 256, 0.5f, 0.0f);
  assert(std::fabs(loud.mon.back() - 0.5 * fromDb(-6.0) * fromDb(16.0) * fromDb(-10.0)) < 5e-3);
  rt_player_destroy(p);
  std::puts("ok: pedal_levels_scale_between_the_stages");
}

void test_gate_comes_first_with_a_pedal_and_closes_the_real_pedals_feed() {
  RTPlayer* p = makePlayer(0.0, 0.0, 0.0);
  RTEngine* pedal = rt_player_engine(p, RT_LEG_PEDAL);
  RTEngine* legA = rt_player_engine(p, RT_LEG_A);
  RTEngine* legB = rt_player_engine(p, RT_LEG_B);
  assert(rt_load_model(pedal, writeLinearModel("test-pedal", 1.0).c_str()) == 1);
  assert(rt_load_model(legB, writeLinearModel("test-amp", 1.0).c_str()) == 1);
  assert(rt_load_model(legA, writeLinearModel("test-amp-a", 1.0).c_str()) == 1);
  rt_set_gate_threshold_db(legB, -20.0);                           // set where the page always sets it: on leg B
  rt_player_select(p, 1);
  auto quiet = runPlayer(p, 200, 256, 0.01f, 0.01f);               // -40 dBFS guitar: under the gate
  assert(std::fabs(quiet.feed.back()) < 1e-4);                     // the reamp feed to the real pedal is gated
  assert(std::fabs(quiet.mon.back()) < 1e-4);                      // and so is the model chain, before the pedal
  auto loud = runPlayer(p, 100, 256, 0.3f, 0.0f);
  assert(std::fabs(loud.feed.back() - 0.3f) < 1e-3);
  assert(std::fabs(loud.mon.back() - 0.3f) < 5e-3);
  rt_player_destroy(p);
  std::puts("ok: gate_comes_first_with_a_pedal_and_closes_the_real_pedals_feed");
}

void test_leg_a_copies_the_amps_tone_stack_when_a_pedal_is_on() {
  RTPlayer* p = makePlayer(0.0, 0.0, 0.0);
  RTEngine* pedal = rt_player_engine(p, RT_LEG_PEDAL);
  RTEngine* legA = rt_player_engine(p, RT_LEG_A);
  RTEngine* legB = rt_player_engine(p, RT_LEG_B);
  assert(rt_load_model(pedal, writeLinearModel("test-pedal", 1.0).c_str()) == 1);
  assert(rt_load_model(legB, writeLinearModel("test-amp", 1.0).c_str()) == 1);
  assert(rt_load_model(legA, writeLinearModel("test-amp-a", 1.0).c_str()) == 1);
  rt_set_eq_mid_db(legB, 12.0);                                    // only ever set on leg B by the shells
  std::vector<float> in(256), out(256), g(256, 0.0f), f(256), m(256);
  double rmsA = 0.0, rmsB = 0.0;
  for (int pass = 0; pass < 2; ++pass) {
    rt_player_select(p, pass == 0 ? 0 : 1);
    double sum = 0.0; int count = 0;
    for (int b = 0; b < 300; ++b) {
      for (int i = 0; i < 256; ++i) { const float x = 0.05f * std::sin(2.0 * M_PI * 800.0 * (b * 256 + i) / 48000.0); g[i] = x; in[i] = x; }
      rt_player_process(p, g.data(), in.data(), f.data(), m.data(), 256);
      if (b >= 200) for (float v : m) { sum += v * v; ++count; }
    }
    (pass == 0 ? rmsA : rmsB) = std::sqrt(sum / count);
  }
  assert(rmsA > 0.05 / std::sqrt(2.0) * 2.5);                      // +12 dB at 800 Hz on the real pedal's leg too (x4 in amplitude, minus the filter's skirt)
  assert(std::fabs(rmsA / rmsB - 1.0) < 0.05);                     // and the same on both
  rt_player_destroy(p);
  std::puts("ok: leg_a_copies_the_amps_tone_stack_when_a_pedal_is_on");
}

int main() {
  test_tuner_reads_the_raw_input_of_an_engine_and_of_the_players_feed();
  test_starts_silent_and_glides_to_target();
  test_settled_gain_multiplies_the_input();
  test_ceiling_clips();
  test_stop_fades_to_silence_and_finishes();
  test_gain_change_while_running_glides();
  test_pre_gain_multiplies_before_the_output_gain();
  test_load_model_missing_file_fails_cleanly();
  test_ir_unit_impulse_is_transparent();
  test_hot_swapping_the_ir_while_audio_runs_is_safe();
  test_ir_level_knobs_scale_before_and_after_the_ir();
  test_gate_mutes_below_threshold_passes_above();
  test_eq_mid_boost_increases_amplitude_at_center_frequency();
  test_last_peak_db_tracks_the_output();
  test_player_feed_is_the_guitar_at_unity_gain();
  test_player_selects_between_the_real_gear_and_the_model();
  test_player_switch_is_a_short_click_free_cross_fade();
  test_player_b_trim_and_the_ceiling();
  test_player_b_delay_lines_the_model_up_with_the_gear();
  test_player_peaks_show_each_leg_even_when_not_heard();
  test_player_stop_fades_every_part_and_finishes();
  test_player_ir_on_the_gear_leg_only_changes_that_leg();
  test_slimmable_model_switches_between_lite_and_full();
  test_player_gate_also_closes_the_real_gear_legs();
  test_pedal_stage_sits_in_front_of_the_amp_on_leg_b_and_leg_a_has_its_own_amp();
  test_pedal_levels_scale_between_the_stages();
  test_gate_comes_first_with_a_pedal_and_closes_the_real_pedals_feed();
  test_leg_a_copies_the_amps_tone_stack_when_a_pedal_is_on();
  std::puts("all RTEngine tests passed");
  return 0;
}
