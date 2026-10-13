// The NAM player: the second tab. Loaded after app.js, whose helpers it uses (the shared document-level `$`, `el`, `t`, `api`, `post`, `S`, the
// knob builders, the picker menus' closing). Server side: capture_api/player.py; audio: native/engine/RTEngine.h's RTPlayer.
//
// What the tab does, in the order the user lives it: read the calibration numbers (the guitar's Hi-Z one is also what the official plugin in
// the DAW wants), pick the model the capture produced (from a file, or straight from the trainer), start, and switch between A (the real gear,
// heard through the return) and B (the model, played by the guitar) with every level matched by calibration.
"use strict";

const P = {
  running: false, leg: "A", ref: { model: "", ir: "", pedal: "" }, temp: { model: "", ir: "", pedal: "" },
  start: null,                 // what /native/player/start (or the swap) answered: gains, the model's own levels
  state: null,                 // the native engine's last /state while running
  prev: null,                  // what Python had open before the player took the interface (put back on stop)
  trainer: { loaded: false, loading: false, online: false, host: null, models: [] },
  irGearTouched: false, editTab: "nam", lanesShown: "", copyTimer: null,
};
const playerRunning = () => P.running;
const plSleep = (ms) => new Promise((r) => setTimeout(r, ms));
const signed = (x, d = 2) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d);

// ------------------------------------------------------------------ the tabs
function showTab(name) {
  for (const b of document.querySelectorAll("#mainTabs [data-tab]")) b.setAttribute("aria-selected", String(b.dataset.tab === name));
  $("#paneCapture").hidden = name !== "capture";
  $("#panePlayer").hidden = name !== "player";
  try { localStorage.setItem("tab", name); } catch {}
  if (name === "player") { loadPlayerTrainerModels(); renderPlayer(); }
}
document.querySelectorAll("#mainTabs [data-tab]").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

// ------------------------------------------------------------------ the calibration numbers
// Straight from the calibration list, by what step 1 has chosen right now: the interface does not have to be open to read them.
function playerCals() {
  const tg = liveTargets();
  if (!tg) return null;
  return { tg, co: findCal("out", tg.deviceName, tg.outCh), ciG: findCal("in", tg.deviceName, tg.guitarCh), ciR: findCal("in", tg.deviceName, tg.returnCh) };
}
const READOUTS = ["guitar", "out", "ret", "unity"];
function buildReadouts() {
  $("#plReadouts").replaceChildren(...READOUTS.map((id) => {
    const node = el("div", { className: "pl-readout" + (id === "guitar" ? " main" : ""), id: "plRo_" + id },
      el("div", { className: "lab" }), el("div", { className: "val" }, el("span", { className: "n" }), el("small", { className: "u" })), el("div", { className: "sub" }));
    if (id === "guitar") {
      const copy = el("button", { type: "button", className: "small", id: "plCopyGuitar" });
      copy.addEventListener("click", copyGuitarValue);
      node.append(copy);
    }
    return node;
  }));
}
function setReadout(id, label, value, unit, sub, missing) {
  const node = $("#plRo_" + id), [lab, val, subEl] = node.children;
  node.classList.toggle("missing", !!missing);
  lab.textContent = label;
  val.firstChild.textContent = missing ? "—" : value;
  val.lastChild.textContent = unit;
  subEl.textContent = sub;
}
async function copyGuitarValue() {
  const c = playerCals();
  if (!c || !c.ciG) return;
  const text = c.ciG.ci_dbu.toFixed(2);
  try { await navigator.clipboard.writeText(text); }
  catch {                                                    // no clipboard API in this web view: the old way
    const ta = el("textarea", { value: text }); document.body.append(ta); ta.select();
    try { document.execCommand("copy"); } catch {} ta.remove();
  }
  $("#plCopyGuitar").textContent = t("cap.player.cal.copied");
  clearTimeout(P.copyTimer);
  P.copyTimer = setTimeout(() => { $("#plCopyGuitar").textContent = t("cap.player.cal.copy"); }, 1500);
}
function renderPlayerCal() {
  const c = playerCals();
  const nums = c ? { n: (ch) => ch + 1 } : null;
  const date = (e) => (e && e.date ? t("cap.player.cal.date", { date: fmtDate(e.date) }) : "");
  const g = c && c.ciG, o = c && c.co, r = c && c.ciR;
  setReadout("guitar", t("cap.player.cal.guitar", { n: nums ? nums.n(c.tg.guitarCh) : "–" }), g ? signed(g.ci_dbu) : "", "dBu",
    g ? `${t("cap.player.cal.guitar_sub")} · ${date(g)}` : t("cap.player.cal.none_guitar"), !g);
  setReadout("out", t("cap.player.cal.out", { n: nums ? nums.n(c.tg.outCh) : "–" }), o ? signed(o.co_dbu) : "", "dBu",
    o ? `${t("cap.player.cal.out_sub")} · ${date(o)}` : t("cap.player.cal.none_out"), !o);
  setReadout("ret", t("cap.player.cal.ret", { n: nums ? nums.n(c.tg.returnCh) : "–" }), r ? signed(r.ci_dbu) : "", "dBu",
    r ? `${t("cap.player.cal.ret_sub")} · ${date(r)}` : t("cap.player.cal.none_ret"), !r);
  setReadout("unity", t("cap.player.cal.unity"), g && o ? signed(g.ci_dbu - o.co_dbu) : "", "dB", g && o ? t("cap.player.cal.unity_sub") : t("cap.player.cal.none"), !(g && o));
  if (!$("#plCopyGuitar").textContent) $("#plCopyGuitar").textContent = t("cap.player.cal.copy");
  $("#plCopyGuitar").hidden = !g;
  $("#plCalPlugin").textContent = c && c.tg.guitarCh === c.tg.returnCh ? t("cap.player.cal.same") : g ? t("cap.player.cal.plugin", { dbu: g.ci_dbu.toFixed(2) }) : "";
}
$("#plCalGo").addEventListener("click", () => { showTab("capture"); $("#cardIn").scrollIntoView({ behavior: "smooth" }); });

// ------------------------------------------------------------------ model / IR pickers (the file lists are shared with the live tool: S.liveFiles)
const PL = { model: { trigger: "#plModelTrigger", text: "#plModelText", menu: "#plModelMenu" }, ir: { trigger: "#plIrTrigger", text: "#plIrText", menu: "#plIrMenu" },
  pedal: { trigger: "#plPedalTrigger", text: "#plPedalText", menu: "#plPedalMenu" } };
// the pedal's NAM comes from the same list as the amp's (kind "model" on the server)
const plFiles = (kind) => (kind === "ir" ? S.liveFiles.irs : S.liveFiles.models) || [];
function plLabel(kind, ref) {
  if (!ref) return t(kind === "model" ? "cap.player.model_none" : kind === "pedal" ? "cap.player.pedal_none" : "cap.live.none");
  if (ref.startsWith("tmp:")) return P.temp[kind] || ref;
  const e = plFiles(kind).find((x) => x.name === ref);
  return !e ? ref : e.name;
}
function plRow(label, active, onClick, extra = {}) {
  const row = el("div", { className: "live-picker-row" + (active ? " active" : ""), role: "option" }, el("span", { textContent: label }));
  if (onClick) row.addEventListener("click", (e) => { e.stopPropagation(); onClick(); });
  if (extra.context) row.addEventListener("contextmenu", (ev) => { ev.preventDefault(); ev.stopPropagation(); extra.context(ev); });
  return row;
}
function plHeader(text) { return el("div", { className: "live-picker-row", style: "cursor:default;opacity:.7;font-size:11px;letter-spacing:.14em;text-transform:uppercase", textContent: text }); }
function buildPlayerMenu(kind) {
  const menu = $(PL[kind].menu);
  const list = plFiles(kind);
  const rows = [];
  if (kind === "ir") rows.push(plRow(t("cap.live.none"), !P.ref.ir, () => plSelect("ir", "")));
  if (kind === "pedal") rows.push(plRow(t("cap.player.pedal_none"), !P.ref.pedal, () => plSelect("pedal", "")));
  if (P.ref[kind].startsWith("tmp:")) rows.push(plRow(P.temp[kind] || P.ref[kind], true, () => plSelect(kind, P.ref[kind])));
  for (const e of list) {
    rows.push(plRow(e.name, e.name === P.ref[kind], () => plSelect(kind, e.name),
      { context: (ev) => openLiveContextMenu(kind === "pedal" ? "model" : kind, e.name, ev.clientX, ev.clientY) }));
  }
  if (kind === "model" || kind === "pedal") {                 // the way back: what the trainer has finished
    rows.push(plHeader(t("cap.player.trainer_models")));
    const tr = P.trainer;
    if (tr.loading && !tr.loaded) rows.push(plHeader(t("cap.player.trainer_loading")));
    else if (!tr.online) rows.push(plHeader(t("cap.player.trainer_off")));
    else if (!tr.models.length) rows.push(plHeader(t("cap.player.trainer_empty")));
    else for (const m of tr.models) rows.push(plRow(`${m.name}${m.created ? " — " + fmtDate(m.created * 1000) : ""}`, false, () => plFetchTrainerModel(m, kind)));
  }
  menu.replaceChildren(...rows);
}
function togglePlayerMenu(kind) {
  const menu = $(PL[kind].menu), willOpen = menu.hidden;
  closeLiveMenus(); closeLiveContextMenu();
  if (willOpen) {
    buildPlayerMenu(kind); menu.hidden = false;
    if (kind === "model" || kind === "pedal") loadPlayerTrainerModels(true).then(() => { if (!menu.hidden) buildPlayerMenu(kind); });
  }
}
async function loadPlayerTrainerModels(force = false) {
  if (P.trainer.loading || (P.trainer.loaded && !force && Date.now() - (P.trainer.at || 0) < 8000)) return;
  P.trainer.loading = true;
  try {
    const r = await api("/api/capture/trainer/models");
    Object.assign(P.trainer, { online: r.online, host: r.host, models: r.models, loaded: true, at: Date.now() });
  } catch { P.trainer.online = false; P.trainer.models = []; P.trainer.loaded = true; }
  P.trainer.loading = false;
  renderPlayer();
}
async function plFetchTrainerModel(m, kind = "model") {
  closeLiveMenus();
  setMsg("#plMsg", "");
  try {
    const r = await post("/api/capture/trainer/models/fetch", { id: m.id, name: m.name, do_import: $("#plImport").checked });
    if (r.imported) await loadLiveFiles();
    else P.temp[kind] = `${m.name} — ${t("cap.player.trainer_tag")}`;
    await plSelect(kind, r.ref);
  } catch (e) { setMsg("#plMsg", e.message); }
}
async function plSelect(kind, ref) {
  const prev = P.ref[kind];
  P.ref[kind] = ref;
  closeLiveMenus(); renderPlayer();
  if (!P.running) return;
  setMsg("#plAbMsg", "");
  try { await playerSwap(); }
  catch (e) { P.ref[kind] = prev; setMsg("#plAbMsg", e.message); }
  renderPlayer();
}
async function uploadPlayerFile(kind, file) {
  if (!file) return;
  const fd = new FormData();
  fd.append("kind", kind === "pedal" ? "model" : kind); fd.append("do_import", $("#plImport").checked ? "true" : "false"); fd.append("file", file);
  try {
    const r = await api("/api/capture/native/live/upload", { method: "POST", body: fd });
    if (!r.imported) P.temp[kind] = `${file.name} — ${t("cap.live.not_saved")}`;
    else await loadLiveFiles();
    await plSelect(kind, r.ref);
  } catch (e) { setMsg(P.running ? "#plAbMsg" : "#plMsg", e.message); }
}
$("#plModelTrigger").addEventListener("click", (e) => { e.stopPropagation(); togglePlayerMenu("model"); });
$("#plIrTrigger").addEventListener("click", (e) => { e.stopPropagation(); togglePlayerMenu("ir"); });
// the X next to each picker takes the file away: that stage is then off (no amp NAM, no pedal NAM, no cabinet)
for (const kind of ["model", "pedal", "ir"]) $("#pl" + kind[0].toUpperCase() + kind.slice(1) + "Clear").addEventListener("click", () => plSelect(kind, ""));
$("#plPedalTrigger").addEventListener("click", (e) => { e.stopPropagation(); togglePlayerMenu("pedal"); });
$("#plPedalTrainer").addEventListener("click", (e) => { e.stopPropagation(); closeLiveMenus(); togglePlayerMenu("pedal"); });
$("#plPedalBrowse").addEventListener("click", () => pickFile("model", $("#plPedalFile"), (f) => uploadPlayerFile("pedal", f)));
$("#plPedalFile").addEventListener("change", async () => { await uploadPlayerFile("pedal", $("#plPedalFile").files[0]); $("#plPedalFile").value = ""; });
$("#plPedalFolder").addEventListener("click", async () => { try { await post("/api/capture/native/live/files/reveal?kind=model"); } catch (e) { setMsg("#plMsg", e.message); } });
$("#plModelTrainer").addEventListener("click", (e) => { e.stopPropagation(); closeLiveMenus(); togglePlayerMenu("model"); });
$("#plModelBrowse").addEventListener("click", () => pickFile("model", $("#plModelFile"), (f) => uploadPlayerFile("model", f)));
$("#plIrBrowse").addEventListener("click", () => pickFile("ir", $("#plIrFile"), (f) => uploadPlayerFile("ir", f)));
$("#plModelFile").addEventListener("change", async () => { await uploadPlayerFile("model", $("#plModelFile").files[0]); $("#plModelFile").value = ""; });
$("#plIrFile").addEventListener("change", async () => { await uploadPlayerFile("ir", $("#plIrFile").files[0]); $("#plIrFile").value = ""; });
$("#plModelFolder").addEventListener("click", async () => { try { await post("/api/capture/native/live/files/reveal?kind=model"); } catch (e) { setMsg("#plMsg", e.message); } });
$("#plIrFolder").addEventListener("click", async () => { try { await post("/api/capture/native/live/files/reveal?kind=ir"); } catch (e) { setMsg("#plMsg", e.message); } });
$("#plRefresh").addEventListener("click", async () => { await loadLiveFiles(); loadPlayerTrainerModels(true); });

// ------------------------------------------------------------------ start / stop / swap
function playerReady() {
  const c = playerCals();
  if (!c) return { ok: false, why: "cap.player.need_device" };
  if (c.tg.guitarCh === c.tg.returnCh) return { ok: false, why: "cap.player.cal.same" };
  if (c.tg.outCh === c.tg.monL || c.tg.outCh === c.tg.monR) return { ok: false, why: "cap.player.out_clash", params: { n: c.tg.outCh + 1 } };
  if (!P.ref.model && !P.ref.pedal) return { ok: false, why: "cap.player.need_model" };      // the amp's NAM and the pedal's are each optional, one of them is needed
  if ($("#plCalibrated").checked && !(c.co && c.ciG)) {
    // the output and the guitar input are required; say exactly which is missing, with the channel numbers chosen in step 1 (a calibration belongs to ONE channel)
    const what = [!c.co && t("cap.player.miss_out", { n: c.tg.outCh + 1 }), !c.ciG && t("cap.player.miss_guitar", { n: c.tg.guitarCh + 1 })].filter(Boolean).join(", ");
    return { ok: false, why: "cap.player.need_cal", params: { what } };
  }
  // the return input only matches the real gear (A) to the model (B): without it the player starts, with A at 0 dB, and says so
  return { ok: true, cals: c, calibrated: $("#plCalibrated").checked, useLevels: $("#plModelLevels").checked, noReturn: $("#plCalibrated").checked && !c.ciR };
}
function applyPlayerResult(r) {
  P.start = { ...(P.start || {}), ...r };
  renderLite();
  renderPedalLite();
  if (r.pedal && r.gains) { $("#plPedalInDb").value = f2(r.gains.pedal_pre); $("#plPedalOutDb").value = f2(0); }
  const g = r.gains;
  if (!g) return;
  $("#plInputDb").value = f2(g.pre);                         // the model's knobs show the REAL calibrated values, as "Edit NAM" does
  $("#plOutputDb").value = f2(g.post);
}
async function playerSwap() {
  const r = await post("/api/capture/native/player/swap", { confirm: true, model: P.ref.model, ir: P.ref.ir || null, ir_on_gear: $("#plIrGear").checked, pedal: P.ref.pedal || null });
  applyPlayerResult(r);
}
async function playerStart() {
  setMsg("#plMsg", ""); setMsg("#plAbMsg", ""); setMsg("#plEditMsg", "");
  const ready = playerReady();
  if (!ready.ok) return;
  const tg = ready.cals.tg;
  if (!confirm(t("cap.player.confirm"))) return;
  const prev = snapshotOpen();
  try {
    await claimDevice(tg.device, [tg.guitarCh, tg.returnCh], [tg.outCh]);
    P.prev = prev;
    const r = await post("/api/capture/native/player/start", {
      confirm: true, model: P.ref.model, ir: P.ref.ir || null, ir_on_gear: $("#plIrGear").checked, pedal: P.ref.pedal || null,
      monitor_out_channels: [tg.monL, ...(tg.monR != null ? [tg.monR] : [])], calibrated: ready.calibrated, use_model_levels: ready.useLevels, master_db: parseFloat($("#plMasterDb").value), leg: "B",
      b_delay_ms: 0 });
    P.running = true; P.leg = "B"; P.start = null; P.state = null; P.since = Date.now(); P.lastGuitar = P.lastReturn = 0;
    applyPlayerResult(r);
    resetPlayerKnobs(false);
    if (!ready.calibrated || !ready.useLevels) $("#plEditPanel").hidden = false;     // by hand: the Input / Output knobs are right there
    playerStatePoll();
    renderPlayer(); render();
  } catch (e) {
    setMsg("#plMsg", isNativeUnreachable(e) ? t("cap.live.need_native_app") : e.message);
    try { await restoreDevice(prev); } catch { /* best effort */ }
    P.prev = null;
    renderPlayer(); render();
  }
}
async function playerStop() {
  try {
    P.running = false;                                       // stops the poll loop first
    await post("/api/capture/native/player/stop?restore=0");     // the device is restored just below, once: no Python session in between
    const prev = P.prev; P.prev = null;
    await restoreDevice(prev);                               // Python gets back what it had: never the guitar+return pair a capture would read
  } catch (e) { setMsg("#plMsg", e.message); }
  P.state = null;
  renderPlayer(); render();
}
async function playerStatePoll() {
  while (P.running) {
    try { P.state = await api("/api/capture/native/monitor/state"); } catch { /* transient: keep the last reading */ }
    if (!document.hidden) renderPlayer();
    await plSleep(150);
  }
}
$("#plStart").addEventListener("click", playerStart);
$("#plStop").addEventListener("click", playerStop);
$("#plIrGear").addEventListener("change", async () => {
  P.irGearTouched = true;
  if (!P.running) return;
  try { await playerSwap(); } catch (e) { setMsg("#plAbMsg", e.message); }
  renderPlayer();
});
$("#plCalibrated").addEventListener("change", renderPlayer);

// A page opened (or reloaded) while the player is running on the native engine must be able to stop it: the server remembers what runs.
async function resumePlayerIfRunning() {
  try {
    const r = await api("/api/capture/native/player/active");
    if (!r.active) return;
    const engine = await api("/api/capture/native/monitor/state");
    if (!(engine.opened && engine.player)) return;          // the server remembers a player the native engine no longer has: nothing to resume
    P.running = true; P.leg = engine.leg || r.active.leg;
    P.start = { gains: r.active.gains, feed_gain_db: r.active.feed_gain_db, model_input_level_dbu: r.active.model_input_level_dbu, model_output_level_dbu: r.active.model_output_level_dbu,
                model_sizes: r.active.model_sizes, model_size: r.active.model_size, pedal: r.active.pedal };
    P.ref.pedal = r.active.pedal_ref || "";
    renderLite(); renderPedalLite();
    if (r.active.pedal && r.active.gains) { $("#plPedalInDb").value = f2(r.active.gains.pedal_pre); $("#plPedalOutDb").value = f2(0); }
    P.ref.model = r.active.model_ref || ""; P.ref.ir = r.active.ir_ref || "";
    $("#plIrGear").checked = !!r.active.ir_on_gear;
    $("#plCalibrated").checked = r.active.calibrated !== false; $("#plModelLevels").checked = r.active.use_model_levels !== false;
    if (r.active.master_db != null) $("#plMasterDb").value = f2(r.active.master_db);
    P.since = Date.now(); P.lastGuitar = P.lastReturn = 0;
    if (r.active.gains) { $("#plInputDb").value = f2(r.active.gains.pre); $("#plOutputDb").value = f2(r.active.gains.post); }
    playerStatePoll();
    renderPlayer(); render();
  } catch { /* no player to resume */ }
}

// ------------------------------------------------------------------ the A/B switch
async function setLeg(leg) {
  if (!P.running || P.leg === leg) return;
  const before = P.leg;
  P.leg = leg; renderPlayer();                               // the key moves at once; the engine follows within a block
  try { await post("/api/capture/native/player/leg", { leg }); }
  catch (e) { P.leg = before; setMsg("#plAbMsg", e.message); renderPlayer(); }
}
for (const [id, leg] of [["#plKeyA", "A"], ["#plKeyB", "B"]]) $(id).addEventListener("click", (e) => { setLeg(leg); e.currentTarget.blur(); });
document.addEventListener("keydown", (e) => {
  if (!P.running || $("#panePlayer").hidden || e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
  const target = e.target, tag = (target && target.tagName) || "";
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || (target && target.isContentEditable)) return;
  const k = e.key.toLowerCase();
  if (k === "a") setLeg("A");
  else if (k === "b") setLeg("B");
  else if (k === " " && tag !== "BUTTON") { e.preventDefault(); setLeg(P.leg === "A" ? "B" : "A"); }
});

// ------------------------------------------------------------------ the model's knobs, the B trim, the delay
const PL_GROUPS_NAM = [
  ["cap.live.sec_nam", [["plInputDb", "cap.live.input_level", -24, 24, 0.5, 0], ["plGateDb", "cap.live.gate", -100, 0, 1, -100], ["plOutputDb", "cap.live.output_level", -24, 24, 0.5, 0]]],
  ["cap.live.sec_eq", [["plEqBassDb", "cap.live.eq_bass", -12, 12, 0.5, 0], ["plEqMidDb", "cap.live.eq_mid", -12, 12, 0.5, 0], ["plEqTrebleDb", "cap.live.eq_treble", -12, 12, 0.5, 0]], "sliders"],
];
const PL_GROUPS_IR = [["cap.live.sec_ir", [["plIrInDb", "cap.live.ir_in", -24, 24, 0.5, 0], ["plIrOutDb", "cap.live.ir_out", -24, 24, 0.5, 0]]]];
const PL_GROUPS_PEDAL = [["cap.player.sec_pedal", [["plPedalInDb", "cap.player.pedal_in", -24, 24, 0.5, 0], ["plPedalGateDb", "cap.live.gate", -100, 0, 1, -100],
  ["plPedalOutDb", "cap.player.pedal_out", -24, 24, 0.5, 0]]]];
$("#plKnobsPedal").replaceChildren(...buildKnobGroup(PL_GROUPS_PEDAL));
const PL_GROUPS_TRIM = [["cap.player.sec_ab", [["plBTrim", "cap.player.b_trim", -12, 12, 0.5, 0]]]];
$("#plKnobsNam").replaceChildren(...buildKnobGroup(PL_GROUPS_NAM));
$("#plKnobsIr").replaceChildren(...buildKnobGroup(PL_GROUPS_IR));
$("#plTrimKnobs").replaceChildren(...buildKnobGroup(PL_GROUPS_TRIM));
// The listening volume: always on the page, before and while playing, and it starts at -18 dB every time (a NAM profile clips easily). It rides on the output gain of
// BOTH legs, before their ceiling, so A and B stay matched and the clipping inside the model's leg is really avoided - not a trim after the damage is done.
const PL_GROUPS_MASTER = [["cap.player.sec_master", [["plMasterDb", "cap.player.master", -48, 12, 0.5, -18]]]];
$("#plMasterKnob").replaceChildren(...buildKnobGroup(PL_GROUPS_MASTER));
$("#plMasterDb").value = f2(-18);

let plControlsTimer = null, plPending = {};
function pushPlayerControls(patch) {                         // patches inside the debounce window are merged: none is lost
  if (!P.running) return;
  Object.assign(plPending, patch);
  clearTimeout(plControlsTimer);
  plControlsTimer = setTimeout(() => {
    const body = plPending; plPending = {};
    post("/api/capture/native/player/controls", body).then(() => setMsg("#plEditMsg", "")).catch((e) => setMsg("#plEditMsg", e.message));
  }, 40);
}
const knobValue = (id) => parseFloat($("#" + id).value);
[["plInputDb", "pre_gain_db"], ["plOutputDb", "gain_db"], ["plGateDb", "gate_threshold_db"], ["plEqBassDb", "eq_bass_db"], ["plEqMidDb", "eq_mid_db"],
  ["plEqTrebleDb", "eq_treble_db"], ["plIrInDb", "ir_in_db"], ["plIrOutDb", "ir_out_db"], ["plBTrim", "b_trim_db"], ["plMasterDb", "master_db"],
  ["plPedalInDb", "pedal_pre_gain_db"], ["plPedalOutDb", "pedal_gain_db"], ["plPedalGateDb", "gate_threshold_db"]]
  .forEach(([id, field]) => $("#" + id).addEventListener("input", () => pushPlayerControls({ [field]: knobValue(id) })));
// ONE noise gate, the first effect: its knob is in the NAM tab and in the pedal tab, and the two always show the same value
$("#plGateDb").addEventListener("input", () => { $("#plPedalGateDb").value = $("#plGateDb").value; });
$("#plPedalGateDb").addEventListener("input", () => { $("#plGateDb").value = $("#plPedalGateDb").value; });
function resetPlayerKnobs(push) {                            // back to the calibrated values the engine started with
  const g = P.start && P.start.gains;
  const values = { plInputDb: g ? g.pre : 0, plOutputDb: g ? g.post : 0, plGateDb: -100, plPedalGateDb: -100, plEqBassDb: 0, plEqMidDb: 0, plEqTrebleDb: 0, plIrInDb: 0, plIrOutDb: 0, plBTrim: 0,
    plPedalInDb: g && g.pedal_pre != null ? g.pedal_pre : 0, plPedalOutDb: 0 };
  for (const [id, v] of Object.entries(values)) $("#" + id).value = typeof v === "number" ? f2(v) : v;
  if (push) pushPlayerControls({ pre_gain_db: values.plInputDb, gain_db: values.plOutputDb, gate_threshold_db: -100, eq_bass_db: 0, eq_mid_db: 0, eq_treble_db: 0,
    ir_in_db: 0, ir_out_db: 0, b_trim_db: 0, ...(P.start && P.start.pedal ? { pedal_pre_gain_db: values.plPedalInDb, pedal_gain_db: 0 } : {}) });
}
$("#plKnobsReset").addEventListener("click", () => resetPlayerKnobs(true));
// A2 models (official trainer 0.13, standard WaveNet) carry a Lite version next to the full one: the switch shows only for them.
function renderLite() {
  const sizes = (P.start && P.start.model_sizes) || 1;
  $("#plLiteWrap").hidden = sizes < 2;
  $("#plLite").checked = sizes >= 2 && P.start.model_size === 0;
}
function renderPedalLite() {
  const p = P.start && P.start.pedal, sizes = (p && p.sizes) || 1;
  $("#plPedalLiteWrap").hidden = sizes < 2;
  $("#plPedalLite").checked = sizes >= 2 && p.size === 0;
}
$("#plPedalLite").addEventListener("change", async () => {
  const p = P.start && P.start.pedal, sizes = (p && p.sizes) || 1;
  const index = $("#plPedalLite").checked ? 0 : sizes - 1;
  try { await post("/api/capture/native/player/controls", { pedal_size: index }); p.size = index; }
  catch (e) { setMsg("#plEditMsg", e.message); renderPedalLite(); }
});
$("#plTabPedal").addEventListener("click", () => { P.editTab = "pedal"; renderPlayer(); });
$("#plLite").addEventListener("change", async () => {
  const sizes = (P.start && P.start.model_sizes) || 1;
  const index = $("#plLite").checked ? 0 : sizes - 1;
  try { await post("/api/capture/native/player/controls", { model_size: index }); P.start.model_size = index; }
  catch (e) { setMsg("#plEditMsg", e.message); renderLite(); }
});
$("#plEditToggle").addEventListener("click", () => { $("#plEditPanel").hidden = !$("#plEditPanel").hidden; });
$("#plTabNam").addEventListener("click", () => { P.editTab = "nam"; renderPlayer(); });
$("#plTabIr").addEventListener("click", () => { P.editTab = "ir"; renderPlayer(); });

// ------------------------------------------------------------------ the routing, drawn from what is chosen
function lane(tag, active, nodes) {
  const box = el("div", { className: "pl-lane" + (active ? " active" : "") }, el("span", { className: "tag", textContent: tag }));
  nodes.forEach(([text, kind], i) => {
    if (i) box.append(el("span", { className: "pl-arrow", textContent: "›" }));
    box.append(el("span", { className: "pl-node" + (kind ? " " + kind : ""), textContent: text }));
  });
  return box;
}
function renderLanes(c) {
  if (!c) { $("#plLanes").replaceChildren(); P.lanesShown = ""; return; }
  const tg = c.tg, head = gear() === "head", g = P.running && P.start && P.start.gains;
  const feed = g ? P.start.feed_gain_db : c.ciG && c.co ? c.ciG.ci_dbu - c.co.co_dbu : null;
  const amp = g ? g.amp : c.ciR && c.co ? c.ciR.ci_dbu - c.co.co_dbu : null;
  const db = (x) => (x == null ? "? dB" : `${signed(x, 1)} dB`);
  const irOn = !!P.ref.ir, irGear = $("#plIrGear").checked && irOn;
  const monitor = `${t("cap.player.lane.mon", { n: tg.monR != null ? `${tg.monL + 1}-${tg.monR + 1}` : tg.monL + 1 })}`;
  const pedalTest = !!P.ref.pedal;                           // "Test Pedali": the real gear in A is the pedal, B is its model in front of the amp's
  const ampNodes = P.ref.model ? [[`${t("cap.player.lane.model")} · ${plLabel("model", P.ref.model)}`, "gear"]] : [];
  const pg = g && g.pedal;
  const laneA = pedalTest
    ? [[t("cap.player.lane.guitar", { n: tg.guitarCh + 1 })], [db(feed), "gain"], [t("cap.player.lane.out", { n: tg.outCh + 1 })], [t("cap.player.lane.pedal"), "gear"],
       [t("cap.player.lane.ret", { n: tg.returnCh + 1 })], [pg ? db(g.amp_pre_a) : "? dB", "gain"], ...ampNodes, ...(irOn ? [[t("cap.player.lane.ir")]] : []),
       [g ? db(g.post) : "? dB", "gain"], [monitor, "end"]]
    : [[t("cap.player.lane.guitar", { n: tg.guitarCh + 1 })], [db(feed), "gain"], [t("cap.player.lane.out", { n: tg.outCh + 1 })], [t("cap.player.lane.reamp")],
    [t(pedalTest ? "cap.player.lane.pedal" : "cap.player.lane.amp"), "gear"], [t("cap.player.lane.ret", { n: tg.returnCh + 1 })], [db(amp), "gain"],
    ...(irGear ? [[t("cap.player.lane.ir")]] : []), [monitor, "end"]];
  const laneB = pedalTest
    ? [[t("cap.player.lane.guitar", { n: tg.guitarCh + 1 })], [pg ? db(g.pedal_pre) : "? dB", "gain"], [`${t("cap.player.lane.pedal_model")} · ${plLabel("pedal", P.ref.pedal)}`, "gear"],
       [g ? db(g.pre) : "? dB", "gain"], ...ampNodes, ...(irOn ? [[t("cap.player.lane.ir")]] : []), [g ? db(g.post) : "? dB", "gain"], [monitor, "end"]]
    : [[t("cap.player.lane.guitar", { n: tg.guitarCh + 1 })], [g ? db(g.pre) : "? dB", "gain"], ...ampNodes,
    ...(irOn ? [[t("cap.player.lane.ir")]] : []), [g ? db(g.post) : "? dB", "gain"], [monitor, "end"]];
  const signature = JSON.stringify([laneA, laneB, P.running && P.leg]);
  if (signature === P.lanesShown) return;
  P.lanesShown = signature;
  $("#plLanes").replaceChildren(lane("A", P.running && P.leg === "A", laneA), lane("B", P.running && P.leg === "B", laneB));
}

// ------------------------------------------------------------------ "why do I hear nothing?"
// The meters say it, but only to who reads them: after a few seconds of running, the commonest causes are named in words. A signal counts as
// "there" above -55 dBFS (a guitar, even a soft one, is far above that; the interface's own noise is below -85).
const DIAG_FLOOR_DBFS = -55, DIAG_GRACE_MS = 5000;
function renderPlayerDiag(c, st) {
  const now = Date.now();
  if ((st.guitar_in_peak_dbfs ?? -240) > DIAG_FLOOR_DBFS) P.lastGuitar = now;
  if ((st.in_peak_dbfs ?? -240) > DIAG_FLOOR_DBFS) P.lastReturn = now;
  const quiet = (stamp) => now - Math.max(stamp || 0, P.since || now) > DIAG_GRACE_MS;
  let key = "", params = {};
  if (!P.since || now - P.since < DIAG_GRACE_MS || !c) key = "";
  else if (quiet(P.lastGuitar)) { key = "cap.player.diag.guitar"; params = { n: c.tg.guitarCh + 1 }; }
  else if (P.leg === "A" && quiet(P.lastReturn)) { key = "cap.player.diag.return"; params = { n: c.tg.returnCh + 1, out: c.tg.outCh + 1 }; }
  else if (now - P.lastGuitar < 1500 && (st.out_peak_dbfs ?? -240) < -80 && ((P.leg === "A" ? st.a_peak_dbfs : st.b_peak_dbfs) ?? -240) > -60) {
    key = "cap.player.diag.monitor"; params = { n: c.tg.monR != null ? `${c.tg.monL + 1}-${c.tg.monR + 1}` : c.tg.monL + 1 };
  }
  $("#plDiag").textContent = key ? t(key, params) : "";
}

// ------------------------------------------------------------------ render
function renderPlayer() {
  $("#plTabLamp").classList.toggle("on", P.running);
  if ($("#panePlayer").hidden) return;
  const c = playerCals(), ready = playerReady(), st = P.state || {};
  renderPlayerCal();
  // the model that was just picked may have been deleted from the list (right-click "Elimina")
  if (P.ref.model && !P.ref.model.startsWith("tmp:") && S.liveFiles.models.length && !S.liveFiles.models.some((m) => m.name === P.ref.model) && !P.running) P.ref.model = "";
  if (P.ref.ir && !P.ref.ir.startsWith("tmp:") && S.liveFiles.irs.length && !S.liveFiles.irs.some((m) => m.name === P.ref.ir) && !P.running) P.ref.ir = "";
  $("#plModelText").textContent = plLabel("model", P.ref.model);
  $("#plIrText").textContent = plLabel("ir", P.ref.ir);
  if (P.ref.pedal && !P.ref.pedal.startsWith("tmp:") && S.liveFiles.models.length && !S.liveFiles.models.some((m) => m.name === P.ref.pedal) && !P.running) P.ref.pedal = "";
  $("#plPedalText").textContent = plLabel("pedal", P.ref.pedal);
  for (const kind of ["model", "pedal", "ir"]) $("#pl" + kind[0].toUpperCase() + kind.slice(1) + "Clear").disabled = !P.ref[kind];
  $("#plPedalHint").hidden = !!P.ref.pedal;
  const tr = S.trainerStatus;
  setLamp("#plTrainerDot", "#plTrainerText", !!(tr && tr.online), tr && tr.host);
  const head = gear() === "head";
  if (!P.irGearTouched && !P.running) $("#plIrGear").checked = head;       // a head's DI/load-box return has no cabinet; a pedal usually none either way
  $("#plIrGear").disabled = !P.ref.ir;
  $("#plIrGearHint").textContent = t(head ? "cap.player.ir_gear_head" : "cap.player.ir_gear_pedal");
  $("#plCalibrated").disabled = P.running;
  $("#plModelLevels").disabled = P.running;
  $("#plManualHint").textContent = !$("#plCalibrated").checked || !$("#plModelLevels").checked ? t("cap.player.manual_hint") : "";

  const takeBusy = S.take && (S.take.status === "recording" || S.take.status === "analyzing");
  const liveBusy = !!(S.live && S.live.running);
  $("#plStart").hidden = P.running; $("#plStop").hidden = !P.running;
  $("#plStart").disabled = !!op() || takeBusy || liveBusy || !ready.ok;
  $("#plNeeded").textContent = takeBusy ? t("cap.live.need_take_done") : liveBusy ? t("cap.player.live_busy") : !ready.ok ? t(ready.why, ready.params)
    : !ready.calibrated ? t("cap.player.uncal") : ready.noReturn ? t("cap.player.no_return", { n: c.tg.returnCh + 1 }) : "";

  // the A/B console
  const pedalTest = !!P.ref.pedal;
  const gearName = t(pedalTest ? "cap.player.ab.a_pedal" : "cap.player.ab.a_amp");    // not the step-2 radio: with no pedal loaded A is the real amp, whatever step 2 says
  $("#plKeyALabel").textContent = gearName;
  $("#plKeyBLabel").textContent = t(pedalTest ? "cap.player.ab.b_pedal" : "cap.player.ab.b");
  for (const [id, leg] of [["#plKeyA", "A"], ["#plKeyB", "B"]]) { $(id).setAttribute("aria-pressed", String(P.running && P.leg === leg)); $(id).disabled = !P.running; }
  renderLanes(c);
  $("#plEditToggle").hidden = !P.running;
  if (!P.running) $("#plEditPanel").hidden = true;
  const pedalOn = P.running && !!(P.start && P.start.pedal);          // the pedal's controls exist only while a pedal test is running
  if (!pedalOn && P.editTab === "pedal") P.editTab = "nam";
  $("#plTabPedal").hidden = !pedalOn;
  $("#plTabNam").classList.toggle("primary", P.editTab === "nam"); $("#plTabIr").classList.toggle("primary", P.editTab === "ir");
  $("#plTabPedal").classList.toggle("primary", P.editTab === "pedal");
  $("#plEditNam").hidden = P.editTab !== "nam"; $("#plEditIr").hidden = P.editTab !== "ir"; $("#plEditPedal").hidden = P.editTab !== "pedal";
  $("#plLevelHint").textContent = P.running ? t("cap.player.trim_hint") : "";
  if (!P.running) { $("#plInfo").textContent = ""; $("#plInfoDetail").textContent = ""; $("#plDiag").textContent = ""; setMeters($("#plMeters"), []); return; }
  const g = P.start && P.start.gains, sp = P.start || {};
  $("#plInfo").textContent = t("cap.player.listening", { leg: t(P.leg === "A" ? "cap.player.leg_a" : "cap.player.leg_b") })
    + (st.xruns ? " — " + t("cap.warn.xruns_live", { n: st.xruns }) : "");
  const levels = sp.model_input_level_dbu != null && sp.model_output_level_dbu != null
    ? " " + t("cap.player.model_levels", { in: f2(sp.model_input_level_dbu), out: f2(sp.model_output_level_dbu) }) : "";
  $("#plInfoDetail").textContent = g ? t("cap.player.running", { feed: f2(sp.feed_gain_db), pre: f2(g.pre), post: f2(g.post), amp: f2(g.amp) }) + levels
    + (g.model_levels === false && !g.pre_from_profile ? " " + t("cap.player.no_levels") : "")
    + (g.post_assumed ? " " + t("cap.player.output_assumed") : (g.pre_from_profile && g.model_levels === false ? " " + t("cap.player.output_manual") : ""))
    + (sp.cal && g.amp_matched === false ? " " + t("cap.player.no_return_running") : "") : "";
  renderPlayerDiag(c, st);
  setMeters($("#plMeters"), [[t("cap.player.meter_a"), st.a_peak_dbfs], [t("cap.player.meter_b"), st.b_peak_dbfs], [t("cap.player.meter_out"), st.out_peak_dbfs],
    [t("cap.player.meter_gin"), st.guitar_in_peak_dbfs], [t("cap.player.meter_gout"), st.guitar_out_peak_dbfs]]);
}

buildReadouts();
(async function playerBoot() {
  showTab("capture");                                        // the program always opens on the capture tab
  await resumePlayerIfRunning();
})();
document.querySelectorAll("#devSelect, #outCh, #returnCh, #guitarCh, #liveMonL, #liveMonR, input[name=gear]").forEach((e) => e.addEventListener("change", renderPlayer));
