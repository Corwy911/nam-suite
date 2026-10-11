// The tuner: the note played on the guitar input (Hi-Z, chosen in step 1), read by the native engine (native/engine/Tuner.h) and shown here.
//
// Where the note comes from, in this order: the NAM player, if it runs; a live session, if it runs WITH the guitar (together); otherwise the
// tuner starts the engine on its own, on the guitar channel alone and with the output muted (capture_api/tuner.py), and stops it again when the
// panel is closed or when anything else needs the interface (the server then hands the device back to Python by itself). Whatever the source,
// the note is `tuner_hz` of the engine's /state, which the page polls already; this file only adds the standalone session and the display.
// Loaded after app.js and player.js: it uses their globals (S, P, claimDevice, restoreDevice, ...) only inside functions, at run time.

const TUNER = { on: false, standalone: false, starting: false, state: null, prev: null, msg: "", last: null, lastAt: 0, ref: 440 };
try { const r = parseFloat(localStorage.getItem("tunerRef")); if (r >= 400 && r <= 480) TUNER.ref = r; } catch { /* the default */ }

const TUNER_LETTERS = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];
const TUNER_SOLFEGE = ["Do", "Do♯", "Re", "Re♯", "Mi", "Fa", "Fa♯", "Sol", "Sol♯", "La", "La♯", "Si"];
const TUNER_IN_TUNE_CENTS = 2.5;          // the green lamp: within +-2.5 cents
const TUNER_HOLD_MS = 1200;               // the last note stays (dimmed) this long after the string stops

function tunerNote(hz, ref) {
  const midi = 69 + 12 * Math.log2(hz / ref), n = Math.round(midi), k = ((n % 12) + 12) % 12;
  return { name: TUNER_LETTERS[k], solfege: TUNER_SOLFEGE[k], octave: Math.floor(n / 12) - 1, cents: (midi - n) * 100, target: ref * Math.pow(2, (n - 69) / 12) };
}

// Who is listening to the guitar right now.
function tunerSource() {
  if (typeof P !== "undefined" && P.running) return { state: P.state };
  if (S.live && S.live.running) return { state: S.live.state, viaReturn: !S.live.together };   // without the guitar, the engine's only input is the return
  if (TUNER.standalone) return { state: TUNER.state };
  return { idle: true };
}

async function tunerStartAlone() {
  if (TUNER.starting || TUNER.standalone) return;
  TUNER.starting = true; TUNER.msg = ""; tunerRender();
  const dev = $("#devSelect").value;
  try {
    if (!dev || !deviceById(dev)) throw new Error(t("cap.tuner.need_device"));
    const prev = snapshotOpen();
    await claimDevice(dev, [+$("#guitarCh").value], [+$("#outCh").value]);
    try { await post("/api/capture/native/tuner/start"); }
    catch (e) { try { await restoreDevice(prev); } catch { /* best effort */ } throw e; }
    TUNER.prev = prev; TUNER.standalone = true; TUNER.state = null;
    tunerPoll();
  } catch (e) {
    TUNER.msg = isNativeUnreachable(e) ? t("cap.live.need_native_app") : e.message;
    TUNER.on = false;
  } finally { TUNER.starting = false; tunerRender(); }
}
async function tunerStopAlone() {
  if (!TUNER.standalone) return;
  TUNER.standalone = false;                                 // the poll loop ends
  try {
    await post("/api/capture/native/tuner/stop?restore=0");      // the device is restored just below, once: no Python session in between
    const prev = TUNER.prev; TUNER.prev = null;
    await restoreDevice(prev);                              // Python gets back what it had: never the guitar-only channel a capture would read
  } catch (e) { TUNER.msg = e.message; }
  TUNER.state = null;
  try { S.engine = (await api("/api/capture/state")).engine; } catch { /* the next poll brings it */ }   // the page must see the device Python got back at once
  tunerRender(); if (typeof render === "function") render();
}
async function tunerPoll() {
  let gone = 0;
  while (TUNER.standalone) {
    try {
      const s = await api("/api/capture/native/monitor/state");
      // the state proxy answers "not opened" when the engine does not reply: a single miss is a hiccup, half a second of them is "something
      // else took the interface" (the server then handed the device back to Python by itself)
      gone = s.opened ? 0 : gone + 1;
      if (gone >= 5) { TUNER.standalone = false; TUNER.prev = null; TUNER.state = null; TUNER.msg = t("cap.tuner.paused"); break; }
      if (s.opened) TUNER.state = s;
    } catch { /* transient: keep the last reading */ }
    await new Promise((r) => setTimeout(r, 100));
  }
  tunerRender();
}
async function tunerToggle() {
  if (TUNER.on) { TUNER.on = false; TUNER.msg = ""; await tunerStopAlone(); tunerRender(); return; }
  TUNER.on = true; TUNER.msg = "";
  const src = tunerSource();
  if (src.idle) await tunerStartAlone();
  tunerRender();
}

// ------------------------------------------------------------------ the display (one per host element; all show the same reading)
const tunerPanels = [];
function buildTuner(host) {
  const toggle = el("button", { type: "button", className: "small primary" });     // the gold key: it must be seen at once among the hints
  const srcLine = el("span", { className: "tuner-src" });
  const lampFlat = el("div", { className: "tuner-lamp flat" }, el("i"), el("span"));
  const lampSharp = el("div", { className: "tuner-lamp sharp" }, el("i"), el("span"));
  const lampOk = el("div", { className: "tuner-lamp ok" }, el("i"), el("span"));
  const noteName = el("b", { textContent: "–" }), noteOct = el("sup"), solf = el("small");
  const needle = el("div", { className: "tuner-needle" });
  const labels = el("div", { className: "tuner-labels" }, ...["−50", "−25", "0", "+25", "+50"].map((x) => el("span", { textContent: x })));
  const cents = el("b"), hz = el("b"), target = el("b");
  const tag = (k) => { const e = el("small"); e.dataset.k = k; return e; };       // dataset is read-only: el() cannot set it through props
  const nums = el("div", { className: "tuner-nums" }, el("span", {}, tag("cents"), cents), el("span", {}, tag("hz"), hz), el("span", {}, tag("target"), target));
  const msg = el("div", { className: "tuner-msg" });
  const restart = el("button", { type: "button", className: "small", hidden: true });
  const glass = el("div", { className: "tuner-glass idle" },
    el("div", { className: "tuner-top" }, lampFlat, el("div", { className: "tuner-name" }, noteName, noteOct, solf, lampOk), lampSharp),
    el("div", { className: "tuner-scale" }, el("div", { className: "tuner-centre" }), needle, labels), nums, msg);
  const refInput = el("input", { type: "number", min: "400", max: "480", step: "1", value: String(TUNER.ref) });
  const refLabel = el("label");
  const refMinus = el("button", { type: "button", textContent: "−" }), refPlus = el("button", { type: "button", textContent: "+" });
  const setRef = (v) => {
    const r = Math.min(480, Math.max(400, Math.round(v) || 440));
    TUNER.ref = r;
    try { localStorage.setItem("tunerRef", String(r)); } catch { /* not remembered */ }
    tunerRender();
  };
  refInput.addEventListener("change", () => setRef(parseFloat(refInput.value)));
  refMinus.addEventListener("click", () => setRef(TUNER.ref - 1));
  refPlus.addEventListener("click", () => setRef(TUNER.ref + 1));
  const body = el("div", { className: "tuner-body", hidden: true }, glass, restart,
    el("div", { className: "tuner-ref" }, refLabel, el("div", { className: "stepfine" }, refMinus, refInput, refPlus)));
  toggle.addEventListener("click", tunerToggle);
  restart.addEventListener("click", async () => { TUNER.msg = ""; await tunerStartAlone(); });
  host.replaceChildren(el("div", { className: "tuner-head" }, toggle, srcLine), body);
  tunerPanels.push({ toggle, srcLine, body, glass, lampFlat, lampSharp, lampOk, noteName, noteOct, solf, needle, cents, hz, target, nums, msg, restart, refInput, refLabel });
}

function tunerRender() {
  if (!tunerPanels.length) return;
  const src = tunerSource();
  const hzNow = src.state && src.state.tuner_hz != null ? src.state.tuner_hz : null;
  const now = Date.now();
  if (hzNow != null) { TUNER.last = tunerNote(hzNow, TUNER.ref); TUNER.lastHz = hzNow; TUNER.lastAt = now; }
  const holding = hzNow == null && TUNER.last && now - TUNER.lastAt < TUNER_HOLD_MS;
  const shown = hzNow != null ? TUNER.last : holding ? tunerNote(TUNER.lastHz, TUNER.ref) : null;
  const live = hzNow != null;
  const guitarCh = +($("#guitarCh").value || 0) + 1, returnCh = +($("#returnCh").value || 0) + 1;
  for (const p of tunerPanels) {
    p.toggle.textContent = t(TUNER.on ? "cap.tuner.close" : "cap.tuner.open");
    p.toggle.disabled = TUNER.starting;
    p.srcLine.textContent = !TUNER.on ? "" : src.viaReturn ? t("cap.tuner.source_return", { ch: returnCh }) : t("cap.tuner.source", { ch: guitarCh });
    p.body.hidden = !TUNER.on;
    p.refLabel.textContent = t("cap.tuner.ref");
    if (document.activeElement !== p.refInput) p.refInput.value = String(TUNER.ref);
    for (const k of ["cents", "hz", "target"]) p.nums.querySelector(`[data-k=${k}]`).textContent = t("cap.tuner." + k);
    p.lampFlat.lastChild.textContent = t("cap.tuner.flat"); p.lampSharp.lastChild.textContent = t("cap.tuner.sharp"); p.lampOk.lastChild.textContent = t("cap.tuner.ok");
    if (!TUNER.on) continue;
    p.glass.classList.toggle("idle", !live);
    const message = TUNER.starting ? t("cap.tuner.starting") : src.idle ? (TUNER.msg || t("cap.tuner.paused")) : (!shown ? t("cap.tuner.play") : "");
    p.msg.textContent = TUNER.msg && !src.idle ? TUNER.msg : message;
    p.restart.hidden = !(src.idle && !TUNER.starting);
    p.restart.textContent = t("cap.tuner.restart");
    if (!shown) {
      p.noteName.textContent = "–"; p.noteOct.textContent = ""; p.solf.textContent = "";
      p.cents.textContent = p.hz.textContent = p.target.textContent = "–";
      p.needle.style.left = "50%"; p.needle.classList.remove("ok");
      for (const l of [p.lampFlat, p.lampSharp, p.lampOk]) l.classList.remove("on");
      continue;
    }
    const inTune = Math.abs(shown.cents) <= TUNER_IN_TUNE_CENTS;
    p.noteName.textContent = shown.name; p.noteOct.textContent = String(shown.octave);
    p.solf.textContent = LANG === "it" ? shown.solfege : "";
    p.cents.textContent = `${shown.cents >= 0 ? "+" : "−"}${Math.abs(shown.cents).toFixed(1)} ¢`;
    p.hz.textContent = `${(live ? hzNow : TUNER.lastHz).toFixed(2)} Hz`;
    p.target.textContent = `${shown.target.toFixed(2)} Hz`;
    p.needle.style.left = `${50 + Math.max(-50, Math.min(50, shown.cents)) * 0.94}%`;
    p.needle.classList.toggle("ok", inTune);
    p.lampOk.classList.toggle("on", inTune);
    p.lampFlat.classList.toggle("on", !inTune && shown.cents < 0);
    p.lampSharp.classList.toggle("on", !inTune && shown.cents > 0);
  }
}

for (const id of ["#liveTuner", "#plTuner"]) { const host = $(id); if (host) buildTuner(host); }
tunerRender();
setInterval(() => { if (!document.hidden) tunerRender(); }, 100);
// a page opened (or reloaded) while the tuner runs on its own: pick it up again, or it would hold the interface with nobody to stop it
(async function tunerResume() {
  try {
    const r = await api("/api/capture/native/tuner/active");
    if (!r.active) return;
    TUNER.on = true; TUNER.standalone = true; TUNER.prev = snapshotOpen();
    tunerPoll();
  } catch { /* nothing to resume */ }
})();
