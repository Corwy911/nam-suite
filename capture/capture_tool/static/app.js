const $ = (s) => document.querySelector(s);
// render() runs on every poll tick: writing the same text again still swaps the text node and can force a relayout,
// so an assignment that would change nothing is skipped (only for a plain-text element; anything else is written as usual).
{
  const desc = Object.getOwnPropertyDescriptor(Node.prototype, "textContent");
  Object.defineProperty(Element.prototype, "textContent", {
    configurable: true, get() { return desc.get.call(this); },
    set(v) {
      const next = v == null ? "" : String(v), f = this.firstChild;
      if (f === null ? next === "" : (f === this.lastChild && f.nodeType === 3 && f.data === next)) return;
      desc.set.call(this, next);
    } });
}
const LANGS = { it: "Italiano", en: "English" };
let I18N = { it: {}, en: {} };
function pickLang() {
  try { const saved = localStorage.getItem("lang"); if (saved in LANGS) return saved; } catch {}
  return (navigator.language || "en").toLowerCase().startsWith("it") ? "it" : "en";
}
let LANG = pickLang();
const locale = () => (LANG === "it" ? "it-IT" : "en-GB");
function t(key, params = {}) {
  const text = (I18N[LANG] && I18N[LANG][key]) ?? (I18N.it && I18N.it[key]) ?? key;
  return text.replace(/\{(\w+)\}/g, (m, k) => (k in params ? params[k] : m));
}
async function api(path, opts = {}) {
  const r = await fetch(path, { ...opts, headers: { "X-Lang": LANG, ...(opts.headers || {}) } });
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch {}
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return r.json();
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
function el(tag, props = {}, ...children) { const e = document.createElement(tag); Object.assign(e, props); e.append(...children); return e; }
function setMsg(id, text) { const box = $(id); box.hidden = !text; box.textContent = text || ""; }
const f1 = (x) => (x == null ? "–" : Number(x).toFixed(1));
const f2 = (x) => (x == null ? "–" : Number(x).toFixed(2));
const fmtDate = (iso) => { try { return new Date(iso).toLocaleString(locale(), { dateStyle: "short", timeStyle: "short" }); } catch { return iso || ""; } };
const fmtTime = (s) => { s = Math.max(0, Math.round(s)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };

function applyStatic() {
  document.documentElement.lang = LANG;
  document.title = t("cap.title") + " · NAM";
  document.querySelectorAll("[data-i18n]").forEach((e) => { e.textContent = t(e.dataset.i18n); });
  document.querySelectorAll("[data-i18n-ph]").forEach((e) => { e.placeholder = t(e.dataset.i18nPh); });
  document.querySelectorAll("[data-i18n-title]").forEach((e) => { e.title = t(e.dataset.i18nTitle); });
  $("#langSelect").value = LANG;
  syncLangKnob();
}

// ------------------------------------------------------------------ state
const S = { toneSource: null, live: null, livePrevOpen: null, liveEditTab: "nam",
  liveFiles: { models: [], irs: [] }, liveRef: { model: "", ir: "" }, liveTempLabel: { model: "", ir: "" },
  allCals: [], devices: [], engine: null, cals: { out: null, in: null }, take: null, demo: false, inputs: [], busyCall: false, meterOn: false, lastTakeStatus: null };
const opened = () => (S.engine && S.engine.opened) || null;
const op = () => (S.engine && S.engine.op) || null;
const isCalibrated = () => $("#calibrated").checked;
const gear = () => document.querySelector("input[name=gear]:checked").value;
const deviceById = (id) => S.devices.find((d) => d.id === id);

async function poll() {
  try {
    const s = await api("/api/capture/state");
    S.engine = s.engine; S.cals = s.calibrations; S.take = s.take; S.demo = s.demo;
    if (S.take && S.take.status !== S.lastTakeStatus) { S.lastTakeStatus = S.take.status; }
    // the whole calibration list (every device/channel, not just the one open right now): the monitor needs to know
    // whether the guitar channel is calibrated even while step 1 has the return channel open, or nothing at all
    S.allCals = (await api("/api/capture/calibration")).entries;
    render();
  } catch (e) { /* server restarting: try again */ }
  setTimeout(poll, op() ? 250 : 1000);
}

// ------------------------------------------------------------------ device
// The static list index.html ships (8, 16, 32, 48, 64, 96, 128, 192, 256, 512, 1024) is a reasonable default, but an
// ASIO device (Windows) has its own real min/max/granularity - offering a value the driver cannot actually use is
// exactly how "seleziono 16, esce 48" happens (the driver silently substitutes something else; sounddevice only
// ever echoes back whatever was requested, never what was really granted - see capture/backends.py's
// AsioBackend.buffer_range()). If the device reports its own range, only the values it can really honor are kept.
const STATIC_BUFFER_SIZES = [8, 16, 32, 48, 64, 96, 128, 192, 256, 512, 1024];
function fillBufferSizes(dev) {
  const sel = $("#bufSize"), keep = sel.value;
  const range = dev && dev.buffer_range;
  const values = range
    ? STATIC_BUFFER_SIZES.filter((v) => v >= range.min && v <= range.max &&
        (range.granularity > 0 ? (v - range.min) % range.granularity === 0 : true))
    : STATIC_BUFFER_SIZES;
  // "Automatico" = keep the size the interface has now (what its own control panel shows): the driver calls it "preferred"
  const autoOpt = el("option", { value: "0", textContent: t("cap.dev.buffer_auto") + (range && range.preferred ? ` (${range.preferred})` : "") });
  sel.replaceChildren(autoOpt, ...values.map((v) => el("option", { value: v, textContent: String(v) })));
  sel.value = values.some((v) => String(v) === keep) ? keep : "0";
}

function fillChannels(dev) {
  const fill = (sel, n) => { const keep = sel.value; sel.replaceChildren(...Array.from({ length: n }, (_, i) => el("option", { value: i, textContent: t("cap.dev.channel", { n: i + 1 }) }))); if (keep && +keep < n) sel.value = keep; };
  fillBufferSizes(dev);
  fill($("#outCh"), dev ? dev.out_channels : 0);
  fill($("#returnCh"), dev ? dev.in_channels : 0);
  fill($("#guitarCh"), dev ? dev.in_channels : 0);
  // uscita/e per l'ascolto (cuffie/monitor), separate da #outCh (che pilota l'ampli/pedale) - scelte subito qui
  // al passo 1 (Roman), default canali 1-2 (stereo), scelta libera
  const n = dev ? dev.out_channels : 0;
  const keepL = $("#liveMonL").value, keepR = $("#liveMonR").value;
  $("#liveMonL").replaceChildren(...Array.from({ length: n }, (_, i) => el("option", { value: i, textContent: t("cap.dev.channel", { n: i + 1 }) })));
  $("#liveMonR").replaceChildren(el("option", { value: "", textContent: t("cap.live.mon_none") }),
    ...Array.from({ length: n }, (_, i) => el("option", { value: i, textContent: t("cap.dev.channel", { n: i + 1 }) })));
  $("#liveMonL").value = (keepL !== "" && +keepL < n) ? keepL : (n > 0 ? 0 : "");
  $("#liveMonR").value = (keepR !== "" && +keepR < n) ? keepR : (n > 1 ? 1 : "");
}
// Remembers the last device/channels/buffer picked in step 1 (localStorage, same pattern as "lang" above): without
// this, every restart fell back to channel 0 on every menu, which rarely matches a saved calibration - and since the
// monitor and step 4 only unlock once the picked channels match one, that also silently kept the monitor from starting.
function saveDeviceSettings() {
  try { localStorage.setItem("devSettings", JSON.stringify({
    device: $("#devSelect").value, outCh: $("#outCh").value, returnCh: $("#returnCh").value,
    guitarCh: $("#guitarCh").value, monL: $("#liveMonL").value, monR: $("#liveMonR").value,
    bufSize: $("#bufSize").value })); } catch {}
}
function loadDeviceSettings() { try { return JSON.parse(localStorage.getItem("devSettings") || "null"); } catch { return null; } }

async function loadDevices() {
  setMsg("#devMsg", "");
  $("#devRefresh").disabled = true; $("#devRefresh").textContent = t("cap.dev.searching");
  try {
    const r = await api("/api/capture/devices");
    const real = r.devices.filter((d) => d.kind !== "sim");   // the simulated "demo" interface never shows in the picker
    S.devices = real;
    S.platform = r.platform;
    applyStatic();
    $("#devSelect").replaceChildren(...real.map((d) => el("option", { value: d.id, textContent: `${d.name}  (${d.in_channels} in / ${d.out_channels} out)` })));
    if (!real.length) setMsg("#devMsg", t(S.platform === "darwin" ? "cap.dev.none_coreaudio" : "cap.dev.none_asio") + (r.asio_error ? ` (${r.asio_error.detail})` : ""));
    const saved = loadDeviceSettings();
    const savedDevOk = saved && real.some((d) => d.id === saved.device);
    $("#devSelect").value = savedDevOk ? saved.device : (real[0] || {}).id || "";
    fillChannels(deviceById($("#devSelect").value));
    if (savedDevOk) {
      const dev = deviceById(saved.device);
      if (dev) {
        if (+saved.outCh < dev.out_channels) $("#outCh").value = saved.outCh;
        if (+saved.returnCh < dev.in_channels) $("#returnCh").value = saved.returnCh;
        if (+saved.guitarCh < dev.in_channels) $("#guitarCh").value = saved.guitarCh;
        if (saved.monL != null && +saved.monL < dev.out_channels) $("#liveMonL").value = saved.monL;
        if (saved.monR != null && (saved.monR === "" || +saved.monR < dev.out_channels)) $("#liveMonR").value = saved.monR;
      }
      if (saved.bufSize != null && [...$("#bufSize").options].some((o) => o.value === saved.bufSize)) $("#bufSize").value = saved.bufSize;
    }
  } catch (e) { setMsg("#devMsg", e.message); }
  $("#devRefresh").disabled = false; $("#devRefresh").textContent = t("cap.dev.refresh");
}
$("#devSelect").addEventListener("change", () => { fillChannels(deviceById($("#devSelect").value)); saveDeviceSettings(); });
["outCh", "returnCh", "guitarCh", "liveMonL", "liveMonR", "bufSize"].forEach((id) => $("#" + id).addEventListener("change", saveDeviceSettings));
$("#devRefresh").addEventListener("click", loadDevices);
$("#devOpen").addEventListener("click", async () => {
  setMsg("#devMsg", "");
  try {
    // opens on the input "Sto calibrando" shows (the guitar's Hi-Z by default): a capture reopens on the return by itself (ensureCaptureDevice)
    await post("/api/capture/open", { device: $("#devSelect").value, in_channels: [calibratingChannel()], out_channels: [+$("#outCh").value], blocksize: +$("#bufSize").value || 0 });
    await refreshInputs();
  } catch (e) { setMsg("#devMsg", e.message); }
});
$("#devClose").addEventListener("click", async () => { try { await post("/api/capture/close"); } catch (e) { setMsg("#devMsg", e.message); } });
// Roman: once the interface is open, the device picker disappears but the channels stay editable, always - moving
// one just reopens the interface on the new choice, instead of requiring "Chiudi" first.
const calibratingChannel = () => ($("#inWhich").value === "guitar" ? +$("#guitarCh").value : +$("#returnCh").value);
async function reopenDeviceChannels() {
  const o = opened();
  if (!o || op() || S.livePrevOpen) return;             // nothing open, something running, or the live tool owns the device
  const outCh = +$("#outCh").value, retCh = calibratingChannel();
  if (o.out_channels.length === 1 && o.out_channels[0] === outCh && o.in_channels.length === 1 && o.in_channels[0] === retCh) return;
  setMsg("#devMsg", "");
  try {
    await post("/api/capture/close");
    await post("/api/capture/open", { device: o.device, in_channels: [retCh], out_channels: [outCh], blocksize: +$("#bufSize").value || 0 });
    await refreshInputs();
  } catch (e) { setMsg("#devMsg", e.message); }
}
["outCh", "returnCh", "guitarCh"].forEach((id) => $("#" + id).addEventListener("change", reopenDeviceChannels));
$("#inWhich").addEventListener("change", async () => {
  const o = opened();
  if (!o || op()) { return; }                            // nothing to switch, or something is running
  const ch = $("#inWhich").value === "guitar" ? +$("#guitarCh").value : +$("#returnCh").value;
  setMsg("#inMsg", "");
  try {
    await post("/api/capture/close");
    await post("/api/capture/open", { device: o.device, in_channels: [ch], out_channels: o.out_channels });
  } catch (e) { setMsg("#inMsg", e.message); }
});
$("#meterToggle").addEventListener("click", async () => {
  try { op() === "meter" ? await post("/api/capture/stop") : await post("/api/capture/meter/start"); } catch (e) { setMsg("#devMsg", e.message); }
});

// The monitor and the live tool both take the interface for themselves and give it back as it was: whatever was open
// (if anything) is remembered, the tool's own channels are opened, and on stop - or on a failed start - it is put back.
const snapshotOpen = () => { const o = opened(); return o ? { device: o.device, in_channels: o.in_channels, out_channels: o.out_channels } : null; };
// "Sfoglia": one dialog per KIND of file, each opening where that kind was last picked (the NAM profiles, the IRs and the input.wav each keep their own
// folder, in the player and in the capture tab alike). The browser's file picker with an `id` does exactly that and remembers it between runs; where it
// does not exist (Safari / the Mac web view) the plain file input is used, which opens wherever the system last was.
const PICK_TYPES = {
  model: { description: "NAM", accept: { "application/octet-stream": [".nam"] } },
  ir: { description: "IR (WAV)", accept: { "audio/wav": [".wav"] } },
  input: { description: "WAV", accept: { "audio/wav": [".wav"] } },
};
async function pickFile(kind, fallbackInput, onFile) {
  let file = null;
  if (typeof window.showOpenFilePicker === "function") {
    try {
      const [handle] = await window.showOpenFilePicker({ id: "nam-" + kind, startIn: "documents", multiple: false, types: [PICK_TYPES[kind]] });
      file = await handle.getFile();
    } catch (e) {
      if (e && e.name === "AbortError") return;                        // cancelled: nothing to do
      file = undefined;                                                  // refused or unavailable here: the plain dialog below
    }
  } else file = undefined;
  if (file === undefined) { fallbackInput.click(); return; }
  await onFile(file);
}
// Every open of the interface is a driver session that sets its one buffer size (and between sessions the driver goes back to its own setting): the page
// opens it only when it has to. Here the interface is about to go to the native engine, so no test stream is made (probe false).
async function claimDevice(device, inChannels, outChannels) {
  if (opened()) await post("/api/capture/close");
  await post("/api/capture/open", { device, in_channels: inChannels, out_channels: outChannels, blocksize: +$("#bufSize").value || 0, probe: false });
}
// A capture and its level TEST always use the RETURN input chosen in step 1 (and the step-1 output): whatever else the interface was left
// open on - the guitar input after a monitor / live session, a calibration channel - is reopened on those two first, so a take can never
// be recorded from the wrong input (the server refuses anything else, see _single_input).
async function ensureCaptureDevice() {
  if (typeof TUNER !== "undefined" && TUNER.standalone) { TUNER.on = false; await tunerStopAlone(); }   // the tuner holds the interface only until a capture needs it
  const o = opened();
  if (!o) return;
  const want = +$("#returnCh").value, out = +$("#outCh").value;
  if (o.in_channels.length === 1 && o.in_channels[0] === want && o.out_channels.length === 1 && o.out_channels[0] === out) return;
  await post("/api/capture/close");
  await post("/api/capture/open", { device: o.device, in_channels: [want], out_channels: [out], blocksize: +$("#bufSize").value || 0 });
}
async function restoreDevice(prev, { closeFirst = true } = {}) {
  if (closeFirst) await post("/api/capture/close");
  if (prev) await post("/api/capture/open", { ...prev, blocksize: +$("#bufSize").value || 0 });     // the same buffer choice as the first open
}

// ------------------------------------------------------------------ rendering
// A meter row is built once and then only updated: the poll loops feed these 5 times a second while audio runs.
function meterRow(label, db) {
  const row = el("div", { className: "meter" }, el("span"), el("div", { className: "track" }, el("div")), el("span"));
  updateMeterRow(row, label, db);
  return row;
}
function updateMeterRow(row, label, db) {
  const pct = db == null ? 0 : Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
  const cls = db != null && db >= -0.3 ? "clip" : db != null && db >= -6 ? "hot" : "";
  const [name, track, value] = row.children, fill = track.firstChild;
  name.textContent = label;
  fill.className = "fill " + cls;
  fill.style.width = pct + "%";
  value.textContent = db == null ? "–" : `${f1(db)} dBFS`;
}
function setMeters(container, rows) {                  // rows: [[label, dB or null], ...]
  if (container.children.length !== rows.length) { container.replaceChildren(...rows.map(([l, d]) => meterRow(l, d))); return; }
  rows.forEach(([l, d], i) => updateMeterRow(container.children[i], l, d));
}
function lock(id, ok, reasonEl, reason) {
  $(id).classList.toggle("locked", !ok);
  if (reasonEl) $(reasonEl).textContent = ok ? "" : reason;
}
function render() {
  const o = opened(), running = op();
  $("#devOpen").hidden = !!o; $("#devClose").hidden = !o; $("#meterToggle").hidden = !o;
  $("#devPickRow").hidden = !!o;                    // once open, the picker disappears - only the channels stay (always editable, see reopenDeviceChannels)
  $("#devSelect").disabled = !!o; $("#outCh").disabled = !!running; $("#returnCh").disabled = !!running; $("#devRefresh").disabled = !!o; $("#bufSize").disabled = !!o;
  if (o && o.in_channels.length === 1 && S.devices.length && deviceById(o.device)) {
    // while the interface is open the menus show what is really open (it may have been opened from elsewhere or before a reload).
    // Exception: if the open channel exactly matches "chitarra" and differs from "ritorno", it is the calibration
    // toggle's own doing (#inWhich below) - leave #returnCh alone, or switching back to "Ritorno" would forget it
    if ($("#devSelect").value !== o.device) { $("#devSelect").value = o.device; fillChannels(deviceById(o.device)); }
    if (+$("#outCh").value !== o.out_channels[0]) $("#outCh").value = o.out_channels[0];
    const looksLikeGuitar = o.in_channels[0] === +$("#guitarCh").value && o.in_channels[0] !== +$("#returnCh").value;
    if (!looksLikeGuitar && +$("#returnCh").value !== o.in_channels[0]) $("#returnCh").value = o.in_channels[0];
    if (+$("#guitarCh").value !== +$("#returnCh").value) $("#inWhich").value = looksLikeGuitar ? "guitar" : "return";   // one channel for both: keep what was chosen
  }
  $("#inWhich").disabled = !o || !!running;
  $("#meterToggle").textContent = t(running === "meter" ? "cap.meter.stop" : "cap.meter.start");
  if (o) {
    const i = o.info || {}, ms = (n) => ((n / (i.rate || 48000)) * 1000).toFixed(1);
    const dev = deviceById(o.device);
    $("#devInfo").textContent = t("cap.dev.opened", { name: dev ? dev.name : o.device, rate: i.rate, ms: ms((i.latency_in || 0) + (i.latency_out || 0)),
      lin: i.latency_in || 0, lout: i.latency_out || 0, out: o.out_channels[0] + 1, inn: o.in_channels[0] + 1,
      buf: i.blocksize || "?", bufms: ms(i.blocksize || 0) });
  } else $("#devInfo").textContent = "";
  const live = S.engine && S.engine.state && S.engine.state.in_peak_dbfs;
  const meterRows = () => (o && running ? o.in_channels.map((c, k) => [t("cap.meter.input", { n: c + 1 }), live ? live[k] : null]) : []);
  setMeters($("#meters"), meterRows());
  setMeters($("#metersRec"), meterRows());
  $("#demoBanner").hidden = !S.demo; if (S.demo) $("#demoBanner").textContent = t("cap.demo.banner");
  $("#demoFill").hidden = !S.demo;
  $("#gearHint").textContent = t(gear() === "head" ? "cap.mode.head_hint" : "cap.mode.pedal_hint");
  renderAmpFields();

  const cal = isCalibrated(), pedal = gear() === "pedal";
  const liveRunning = !!(S.live && S.live.running);
  const playerOn = typeof playerRunning === "function" && playerRunning();      // player.js, the second tab
  const busy = liveRunning || playerOn;                  // the live NAM+IR tool and the NAM player own the audio device while running - like a capture
  lock("#cardMode", !!o);
  lock("#cardOut", !!o && cal);
  const inNeeded = pedal;
  $("#inNeeded").textContent = inNeeded ? t("cap.in.needed_pedal") : "";
  lock("#cardIn", !!o && cal && !!S.cals.out, "#inLock", !o || !cal ? t("cap.locked.calibrated") : t("cap.locked.need_out"));
  // the RETURN's calibration, not the one of whatever input is open now (the interface may be open on the guitar's, for its calibration)
  const returnDev = deviceById($("#devSelect").value), returnCal = returnDev ? findCal("in", returnDev.name, +$("#returnCh").value) : null;
  const recReady = !!o && (!cal || (!!S.cals.out && (!pedal || !!returnCal)));
  lock("#cardRec", recReady && !busy, "#recLock", !o ? t("cap.locked.open") : !S.cals.out ? t("cap.locked.need_out") : t("cap.locked.need_in"));
  // the live tool is its own tool, usable at any point in the wizard (see capture_api.py's /native/live routes):
  // never locked by "o" (whatever step 1 currently has open) - only by whether the channels it needs have a
  // calibration on file, checked against the whole calibration list
  lock("#cardLive", true);
  $("#devOpen").disabled = busy; $("#devClose").disabled = busy;
  renderLive(liveRunning);
  if (typeof renderPlayer === "function") renderPlayer();

  // output calibration
  const co = S.cals.out;
  $("#outStored").textContent = co ? t("cap.out.stored", { dbu: f2(co.co_dbu), date: fmtDate(co.date), note: co.note ? ` – ${co.note}` : "" }) : t("cap.out.none");
  $("#toneBtn").textContent = t(running === "tone" ? "cap.out.stop_tone" : "cap.out.play");
  const vr = parseVrmsLocal($("#outVrms").value);
  $("#outResult").textContent = vr ? t("cap.out.result", { v: vr, dbu: f2(20 * Math.log10(vr / 0.7746) - parseFloat($("#outLevel").value)) }) : "";
  const level = (id) => parseFloat($(id).value);
  const coNow = co ? co.co_dbu : null;
  $("#outExpected").textContent = coNow != null && isFinite(level("#outLevel")) ? t("cap.out.expected", { dbu: f1(coNow + level("#outLevel")) }) : "";
  $("#trimResult").textContent = coNow != null && isCalibrated() && isFinite(level("#trim")) ? t("cap.rec.trim_result", { dbu: f2(coNow + level("#trim")) }) : "";
  // input calibration: automatic or manual level
  const manualIn = $("#inMode").value === "manual";
  $("#inLevelField").hidden = !manualIn; $("#inToneField").hidden = !manualIn;
  $("#inTone").textContent = t(running === "tone" && S.toneSource === "in" ? "cap.in.tone_stop" : "cap.in.tone_play");
  $("#inExpected").textContent = manualIn && coNow != null && isFinite(level("#inLevel")) ? t("cap.in.expected", { dbu: f1(coNow + level("#inLevel")) }) : "";
  const stLive = S.engine && S.engine.state || {}, peakNow = stLive.in_peak_dbfs && stLive.in_peak_dbfs[0];
  const liveCi = coNow != null && running && peakNow != null && peakNow > -90 && stLive.level_dbfs != null ? coNow + stLive.level_dbfs - peakNow : null;
  $("#inLive").textContent = liveCi != null && running !== "record" && running !== "meter" ? t("cap.in.live", { peak: f1(peakNow), level: f1(stLive.level_dbfs), ci: f2(liveCi) }) : "";
  const ci = S.cals.in;
  $("#inStored").textContent = ci ? t("cap.in.stored", { dbu: f2(ci.ci_dbu), date: fmtDate(ci.date), level: f1(ci.level_dbfs), spread: f2(ci.spread_db) }) : t("cap.in.none");
  if (ci && ci.stale) $("#inStored").textContent += " " + t("cap.in.stale");
  const prog = S.engine && S.engine.progress || [];
  const last = prog.filter((p) => p.op === "return_cal").slice(-1)[0];
  $("#inProgress").textContent = running === "return_cal" ? t("cap.in.running", { level: last ? f1(last.level_dbfs) : "…", peak: last ? f1(last.peak_dbfs) : "…" }) : "";
  const anyOp = !!running;
  $("#inTone").disabled = anyOp && !(running === "tone" && S.toneSource === "in");
  $("#inRun").disabled = anyOp && !(running === "tone" && S.toneSource === "in"); $("#inVerify").disabled = anyOp || !ci; $("#latProbe").disabled = anyOp; $("#toneBtn").disabled = anyOp && !(running === "tone" && S.toneSource === "out");
  $("#outSave").disabled = running === "record";

  // capture
  const take = S.take, recording = running === "record" || (take && take.status === "recording");
  $("#recStart").hidden = recording; $("#recStop").hidden = !recording;
  $("#recStart").disabled = anyOp || !recReady || !S.inputs.length || (take && (take.status === "analyzing" || take.status === "ready"));
  $("#recBar").hidden = !recording && !(take && take.status === "analyzing");
  const st = S.engine && S.engine.state || {};
  const rate = opened() ? opened().info.rate : 48000;
  if (recording && st.op === "record" && st.start == null) {          // the warm-up: blips until the interface's timing has settled
    $("#recFill").style.width = "0%";
    $("#recText").textContent = t("cap.rec.warmup", { s: Math.floor((st.pos || 0) / rate) });
  } else if (recording && st.total) {
    const start = st.start || 0, frac = Math.min(1, Math.max(0, ((st.pos || 0) - start) / (st.total - start)));
    $("#recFill").style.width = (frac * 100).toFixed(1) + "%";
    const left = (st.total - Math.max(st.pos || 0, start)) / rate / (S.demo ? 60 : 1);
    $("#recText").textContent = t("cap.rec.progress", { pct: (frac * 100).toFixed(0), left: fmtTime(left) }) + (st.clip ? " — " + t("cap.warn.clip_live") : "") + (st.xruns ? " — " + t("cap.warn.xruns_live", { n: st.xruns }) : "");
  } else if (take && take.status === "analyzing") { $("#recFill").style.width = "100%"; $("#recText").textContent = t("cap.rec.analyzing"); }
  else $("#recText").textContent = take && take.status === "aborted" ? t("cap.rec.aborted") : "";
  if (take && take.status === "error") setMsg("#recMsg", errText(take.error));   // other messages are set (and cleared) by the actions

  renderLinearity(running, anyOp);
  $("#levelTest").disabled = anyOp || !o;
  renderResult();
}
const DOT = { ok: "ok", warn: "warn", bad: "bad", hot: "warn", clip: "bad", low: "warn", silent: "bad" };
let linShown = "";
function renderLinearity(running, anyOp) {
  const lin = S.cals.lin, prog = (S.engine && S.engine.progress || []).filter((p) => p.op === "linearity").slice(-1)[0];
  $("#linRun").disabled = anyOp || !S.cals.in;
  let dot = "", text = "";
  if (running === "linearity") text = t("cap.lin.running", { n: prog ? prog.step : 0, total: prog ? prog.total : 5 });
  else if (!lin) text = t("cap.lin.none");
  else if (lin.stale) text = t("cap.lin.stale");
  else { dot = DOT[lin.status] || ""; text = t(lin.reason === "clip" ? "cap.lin.clip" : lin.reason === "too_few" ? "cap.lin.too_few" : "cap.lin." + lin.status, { dev: f2(lin.max_dev_db), span: f1(lin.span_db) }); }
  $("#linDot").className = "dot " + dot; $("#linText").textContent = text;
  $("#linDetails").hidden = !lin;
  const key = lin ? lin.date + LANG : "";
  if (key !== linShown) {                                   // rebuild only when the data change, so that an open report stays open
    linShown = key;
    if (!lin) { $("#linReport").replaceChildren(); return; }
    const head = el("tr", {}, ...["sent", "read", "ci", "dev", "thd"].map((c) => el("th", { textContent: t("cap.lin.col." + c) })));
    const rows = lin.points.map((p) => el("tr", {}, el("td", { textContent: f1(p.level_dbfs) }), el("td", { textContent: f2(p.read_dbfs) }), el("td", { textContent: f2(p.ci_dbu) }),
      el("td", { textContent: p.dev_db == null ? "–" : (p.dev_db > 0 ? "+" : "") + f2(p.dev_db), className: p.dev_db == null ? "" : Math.abs(p.dev_db) > 0.3 ? "bad" : Math.abs(p.dev_db) > 0.1 ? "warn" : "" }),
      el("td", { textContent: f1(p.thdn_db) + (p.clipped ? " ⚠" : "") })));
    $("#linReport").replaceChildren(el("table", { className: "lin" }, head, ...rows),
      el("div", { className: "hint", textContent: t("cap.lin.report_note", { ci: f2(lin.median_ci), date: fmtDate(lin.date) }) }));
  }
}
function clearLevelTest() { S.levelTest = null; $("#testDot").className = "dot"; $("#testText").textContent = ""; $("#testSuggest").textContent = ""; }
function showLevelTest(r) {
  $("#testDot").className = "dot " + (DOT[r.judge] || "");
  $("#testText").textContent = t("cap.test.result", { peak: f1(r.peak_dbfs), head: f1(r.headroom_db), n: r.clipped }) + " — " + t("cap.test.judge." + r.judge);
  const co = S.cals.out;
  $("#testSuggest").textContent = r.suggest_trim_db != null && co ? t("cap.test.suggest", { trim: f1(r.suggest_trim_db), dbu: f2(co.co_dbu + r.suggest_trim_db) }) : "";
}
function parseVrmsLocal(text) {
  const s = String(text || "").trim().toLowerCase().replace(",", ".").replace(/\s+/g, "");
  let v = NaN;
  if (s.endsWith("mv")) v = parseFloat(s) / 1000; else v = parseFloat(s.replace(/v$/, ""));
  return v >= 0.0005 && v <= 200 ? v : null;
}
function errText(e) { return e ? (t("cap.err." + e.key, { detail: e.detail || "" }) === "cap.err." + e.key ? `${e.key}: ${e.detail || ""}` : t("cap.err." + e.key, { detail: e.detail || "" })) : ""; }

// ------------------------------------------------------------------ output calibration
async function demoRoute(route) { if (S.demo) { try { await post("/api/capture/demo/config", { route, gear: gear() === "pedal" ? "pedal" : "head" }); } catch {} } }
$("#toneBtn").addEventListener("click", async () => {
  setMsg("#outMsg", "");
  try {
    if (op() === "tone") { await post("/api/capture/stop"); return; }
    const level = parseFloat($("#outLevel").value);
    if (!confirm(t("cap.out.confirm", { level, out: opened().out_channels[0] + 1 }))) return;
    S.toneSource = "out";
    await post("/api/capture/tone/start", { level_dbfs: level, confirm: true });
  } catch (e) { setMsg("#outMsg", e.message); }
});
$("#outVrms").addEventListener("input", render);
document.querySelectorAll(".capBrowse").forEach((b) => b.addEventListener("click", async () => {
  setMsg("#capBrowseMsg", "");
  try { await post("/api/capture/captures/reveal"); } catch (e) { setMsg("#capBrowseMsg", e.message); }
}));

// ------------------------------------------------------------------ software level knobs (0.5 dB steps)
const snap = (v, lo, hi) => Math.min(hi, Math.max(lo, Math.round(v * 2) / 2));
let levelTimer = null;
function pushLevel(input) {
  // the tone that is playing follows the knob it belongs to (glides, no click); the server refuses a level outside the range
  const owner = { outLevel: "out", inLevel: "in" }[input.id];
  if (op() !== "tone" || S.toneSource !== owner) return;
  const v = parseFloat(input.value);
  if (!isFinite(v)) return;
  clearTimeout(levelTimer);
  levelTimer = setTimeout(() => post("/api/capture/tone/level", { level_dbfs: snap(v, -60, 0) }).catch((e) => setMsg(owner === "out" ? "#outMsg" : "#inMsg", e.message)), 60);
}
document.querySelectorAll(".stepper button").forEach((b) => b.addEventListener("click", () => {
  const input = $("#" + b.dataset.for);
  const lo = parseFloat(input.min), hi = parseFloat(input.max), cur = parseFloat(input.value);
  input.value = snap((isFinite(cur) ? cur : 0) + parseFloat(b.dataset.step), lo, hi);
  input.dispatchEvent(new Event("input", { bubbles: true }));
}));
["outLevel", "inLevel", "trim"].forEach((id) => $("#" + id).addEventListener("input", () => { pushLevel($("#" + id)); if (id === "trim") clearLevelTest(); render(); }));
$("#inMode").addEventListener("change", render);
$("#demoFill").addEventListener("click", async () => {
  try { const r = await api("/api/capture/demo/multimeter?level_dbfs=" + $("#outLevel").value); $("#outVrms").value = r.vrms.toFixed(3); render(); } catch (e) { setMsg("#outMsg", e.message); }
});
$("#outSave").addEventListener("click", async () => {
  setMsg("#outMsg", "");
  try {
    if (op() === "tone") await post("/api/capture/stop");
    const r = await post("/api/capture/calibrate/output", { vrms: $("#outVrms").value, sent_dbfs: parseFloat($("#outLevel").value) });
    S.cals = r.calibrations; $("#outVrms").value = ""; render();
  } catch (e) { setMsg("#outMsg", e.message); }
});

// ------------------------------------------------------------------ input calibration
async function longCall(button, msgId, fn) {
  setMsg(msgId, ""); button.disabled = true;
  try { return await fn(); } catch (e) { setMsg(msgId, e.message); } finally { button.disabled = false; }
}
async function stopToneAndWait() {
  if (op() !== "tone") return;
  await post("/api/capture/stop");
  for (let i = 0; i < 40 && op() === "tone"; i++) await new Promise((res) => setTimeout(res, 100));
}
$("#inTone").addEventListener("click", async () => {
  setMsg("#inMsg", "");
  try {
    if (op() === "tone") { await post("/api/capture/stop"); return; }
    const level = snap(parseFloat($("#inLevel").value), -60, 0);
    if (!confirm(t("cap.in.confirm_tone", { level, out: opened().out_channels[0] + 1 }))) return;
    await demoRoute("loopback");
    S.toneSource = "in";
    await post("/api/capture/tone/start", { level_dbfs: level, confirm: true });
  } catch (e) { setMsg("#inMsg", e.message); }
});
$("#inRun").addEventListener("click", () => longCall($("#inRun"), "#inMsg", async () => {
  const manual = $("#inMode").value === "manual";
  const level = snap(parseFloat($("#inLevel").value), -60, 0);
  if (op() !== "tone" && !confirm(t(manual ? "cap.in.confirm_manual" : "cap.in.confirm", { level, out: opened().out_channels[0] + 1 }))) return;
  await stopToneAndWait();
  await demoRoute("loopback");
  const r = await post("/api/capture/calibrate/input", { confirm: true, ...(manual ? { level_dbfs: level } : {}) });
  if (r.aborted) { $("#inResult").textContent = ""; return; }
  S.cals = r.calibrations;
  if (r.rejected) { $("#inResult").textContent = t("cap.in.rejected." + r.rejected, { sent: f1(r.result.level_dbfs) }); $("#inResult").className = "result-line bad"; return; }
  const e = r.entry;
  $("#inResult").textContent = t("cap.in.result", { dbu: f2(e.ci_dbu), sent: f1(e.level_dbfs), peak: f1(e.peak_dbfs), spread: f2(e.spread_db) }) + " — " + t("cap.in.judge." + e.judge);
  $("#inResult").className = "result-line " + (e.judge === "ok" ? "good" : "bad");
}));
$("#linRun").addEventListener("click", () => longCall($("#linRun"), "#inMsg", async () => {
  if (!confirm(t("cap.lin.confirm", { out: opened().out_channels[0] + 1 }))) return;
  await demoRoute("loopback");
  const r = await post("/api/capture/linearity", { confirm: true });
  if (r.aborted) return;
  S.cals = r.calibrations; render();
}));
$("#inVerify").addEventListener("click", () => longCall($("#inVerify"), "#inMsg", async () => {
  await demoRoute("loopback");
  const r = await post("/api/capture/verify/input");
  const bad = Math.abs(r.deviation_db) > 0.3;
  $("#inResult").textContent = t("cap.in.verify_result", { now: f2(r.ci_now), stored: f2(r.ci_stored), dev: (r.deviation_db > 0 ? "+" : "") + f2(r.deviation_db) }) + " — " + t(bad ? "cap.in.verify_bad" : "cap.in.verify_ok");
  $("#inResult").className = "result-line " + (bad ? "bad" : "good");
}));
$("#latProbe").addEventListener("click", () => longCall($("#latProbe"), "#inMsg", async () => {
  await demoRoute("loopback");
  $("#latResult").textContent = t("cap.in.latency_running");
  const r = await post("/api/capture/latency-probe");
  if (r.delay != null && typeof latencyKey === "function" && latencyKey()) {          // player.js: the model's delay offers it
    S.lastLatency = { key: latencyKey(), ms: r.ms };
    try { localStorage.setItem(latencyKey(), String(r.ms)); } catch {}
  }
  $("#latResult").textContent = r.delay == null ? t("cap.in.latency_none") : t("cap.in.latency_result", { samples: r.delay, ms: f2(r.ms), driver: r.driver_latency });
}));


// ------------------------------------------------------------------ shared with the live tool below
function findCal(kind, deviceName, channel) {
  return (S.allCals || []).find((e) => e.kind === kind && e.device === deviceName && e.channel === channel) || null;
}
// Recognizes srv.cap.native_unreachable by its translated prefix (the server already answered in our language,
// via the X-Lang header) so it is never mistaken for a real native-side error (bad device, bad channel).
function isNativeUnreachable(e) {
  const prefix = t("srv.cap.native_unreachable", { detail: "" }).trim();
  return typeof e.message === "string" && e.message.startsWith(prefix);
}

// ------------------------------------------------------------------ live NAM + IR (native, from the return input)
// A tool of its own, usable at any point in the wizard, on #returnCh: chitarra -> pedale/ampli reale -> ritorno ->
// [NAM] -> [IR] -> uscita. Testata (gear=head): solo IR. Pedale (gear=pedal): NAM + IR insieme, per sentire il
// pedale dentro un ampli+cassa di riferimento (vedi capture_api.py's native_live_start).
async function loadLiveFiles() {
  try {
    const r = await api("/api/capture/native/live/files");
    S.liveFiles = r;
    buildLiveMenu("model"); buildLiveMenu("ir");
    renderLivePicker("model"); renderLivePicker("ir");
  } catch (e) { /* the native engine (or its folders) may not exist yet - the tool just shows no files */ }
}
// Picking a file with "Sfoglia" is a plain browser file input: it opens the normal macOS Open panel and can point
// at any folder, but the browser only ever gets the file's bytes, never its path - the upload IS the "browse" (see
// capture_api.py's own note on native_live_upload). Without "Importa nell'app" ticked the file is never moved into
// Modelli NAM/IR: capture_api.py keeps a throwaway copy just long enough for the native engine to read it once.
async function uploadLiveFile(kind, file) {
  if (!file) return;
  const fd = new FormData();
  fd.append("kind", kind);
  fd.append("do_import", $("#liveImport").checked ? "true" : "false");
  fd.append("file", file);
  try {
    const r = await api("/api/capture/native/live/upload", { method: "POST", body: fd });
    if (!r.imported) S.liveTempLabel[kind] = `${file.name} — ${t("cap.live.not_saved")}`;
    S.liveRef[kind] = r.ref;
    if (r.imported) await loadLiveFiles();      // now behaves exactly like any other saved file
    else { buildLiveMenu(kind); renderLivePicker(kind); }
    if (S.live && S.live.running) {
      const swap = await post("/api/capture/native/live/swap", { confirm: true, model: S.liveRef.model || null, ir: S.liveRef.ir || null });
      S.live.pre_gain_db = swap.pre_gain_db; S.live.gain_db = swap.gain_db;
      resetLiveEditFields(swap);
    }
    render();
  } catch (e) { setMsg("#liveMsg", e.message); }
}

// A custom dropdown, not a native <select>: a native <select>'s open list is drawn by the OS and cannot carry a
// right-click "Elimina" on a single row in any browser (Safari/WebKit included, which is what the packaged app
// uses) - Roman asked for right-click to delete a saved file, so the list has to be our own markup.
function liveEntryLabel(kind, ref) {
  if (!ref) return t("cap.live.none");
  if (ref.startsWith("tmp:")) return S.liveTempLabel[kind] || ref;
  const list = (kind === "model" ? S.liveFiles.models : S.liveFiles.irs) || [];
  const e = list.find((x) => x.name === ref);
  if (!e) return ref;
  return e.name;
}
function renderLivePicker(kind) {
  $(kind === "model" ? "#liveModelTriggerText" : "#liveIrTriggerText").textContent = liveEntryLabel(kind, S.liveRef[kind]);
}
function buildLiveMenu(kind) {
  const menu = $(kind === "model" ? "#liveModelMenu" : "#liveIrMenu");
  const list = (kind === "model" ? S.liveFiles.models : S.liveFiles.irs) || [];
  const rows = [{ ref: "", label: t("cap.live.none") }];
  if (S.liveRef[kind] && S.liveRef[kind].startsWith("tmp:")) rows.push({ ref: S.liveRef[kind], label: S.liveTempLabel[kind], temp: true });
  for (const e of list) rows.push({ ref: e.name, label: e.name, uses: e.uses });
  menu.replaceChildren(...rows.map((r) => {
    const row = el("div", { className: "live-picker-row" + (r.ref === S.liveRef[kind] ? " active" : ""), role: "option" },
      el("span", { textContent: r.label }));
    row.addEventListener("click", () => selectLiveRef(kind, r.ref));
    if (r.ref && !r.temp) row.addEventListener("contextmenu", (ev) => { ev.preventDefault(); openLiveContextMenu(kind, r.ref, ev.clientX, ev.clientY); });
    return row;
  }));
}
// Picking a row calls the native engine right away if "Prova dal vivo" is already running (native_live_swap:
// hot-swaps the model/IR without stopping playback), not just when "Avvia" is next pressed - Roman: picking
// the wrong NAM for a pedal needs to be fixable while still listening, not only before starting.
async function selectLiveRef(kind, ref) {
  const prev = S.liveRef[kind];
  S.liveRef[kind] = ref;
  closeLiveMenus(); renderLivePicker(kind); render();
  if (!(S.live && S.live.running)) return;
  setMsg("#liveMsg", "");
  try {
    const r = await post("/api/capture/native/live/swap", { confirm: true, model: S.liveRef.model || null, ir: S.liveRef.ir || null });
    S.live.pre_gain_db = r.pre_gain_db; S.live.gain_db = r.gain_db;
    resetLiveEditFields(r);
  } catch (e) {
    S.liveRef[kind] = prev;   // the native engine refused it (e.g. both cleared): put the picker back
    renderLivePicker(kind);
    setMsg("#liveMsg", e.message);
  }
  render();
}
function closeLiveMenus() { document.querySelectorAll(".live-picker-menu").forEach((m) => (m.hidden = true)); }
function toggleLiveMenu(kind) {
  const menu = $(kind === "model" ? "#liveModelMenu" : "#liveIrMenu");
  const willOpen = menu.hidden;
  closeLiveMenus(); closeLiveContextMenu();
  if (willOpen) { buildLiveMenu(kind); menu.hidden = false; }
}
let liveContextTarget = null;
function openLiveContextMenu(kind, ref, x, y) {
  liveContextTarget = { kind, ref };
  const menu = $("#liveContextMenu");
  menu.style.left = Math.min(x, window.innerWidth - 160) + "px";
  menu.style.top = Math.min(y, window.innerHeight - 60) + "px";
  menu.hidden = false;
}
function closeLiveContextMenu() { $("#liveContextMenu").hidden = true; liveContextTarget = null; }
document.addEventListener("click", () => { closeLiveMenus(); closeLiveContextMenu(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closeLiveMenus(); closeLiveContextMenu(); } });
$("#liveContextDelete").addEventListener("click", async (e) => {
  e.stopPropagation();
  if (!liveContextTarget) return;
  const { kind, ref } = liveContextTarget;
  closeLiveContextMenu();
  if (!confirm(t("cap.live.delete_confirm", { name: ref }))) return;
  try {
    await api(`/api/capture/native/live/files?kind=${kind}&name=${encodeURIComponent(ref)}`, { method: "DELETE" });
    if (S.liveRef[kind] === ref) S.liveRef[kind] = "";
    await loadLiveFiles();
  } catch (e2) { setMsg("#liveMsg", e2.message); }
});
function isLiveTogether() { return $("#liveTogether").checked; }   // both head and pedal: the guitar feeds the gear while the return is heard
function liveTargets() {
  const dev = deviceById($("#devSelect").value);
  if (!dev) return null;
  return { device: $("#devSelect").value, deviceName: dev.name, outCh: +$("#outCh").value, returnCh: +$("#returnCh").value,
    guitarCh: +$("#guitarCh").value, monL: +$("#liveMonL").value, monR: $("#liveMonR").value === "" ? null : +$("#liveMonR").value };
}
function liveReady() {
  const t = liveTargets();
  if (!t) return { ok: false };
  if (isLiveTogether() && (t.outCh === t.monL || t.outCh === t.monR)) return { ok: false, missing: "clash", targets: t };   // the send and the monitor on one channel
  // "Calibrato" off: same always-startable 0 dB fallback as the guitar monitor (Roman) - skip needing any
  // calibration on file at all, not just tolerate a missing one
  if (!$("#liveCalibrated").checked) return { ok: true, calibrated: false, targets: t };
  const co = findCal("out", t.deviceName, t.outCh), ci = findCal("in", t.deviceName, t.returnCh);
  const together = isLiveTogether(), wantsModel = gear() === "pedal" && !!S.liveRef.model;
  const ciGuitar = together ? findCal("in", t.deviceName, t.guitarCh) : null;
  // strictly needed: the guitar leg is exact unity gain (Ci_guitar - Co) -> output + guitar input; the RETURN's calibration only
  // when a NAM profile is chosen (a head's return is a mic/DI: IR alone needs none)
  const needOut = together || wantsModel, needGuitar = together, needReturn = wantsModel;
  const missing = (needOut && !co) ? "out" : (needGuitar && !ciGuitar) ? "guitar" : (needReturn && !ci) ? "return" : null;
  return { ok: !missing, missing, calibrated: true, co, ci, ciGuitar, targets: t };
}
// The NAM controls (input/gate/output level, EQ) only make sense with a profile loaded - a head is IR-only (no
// model picker at all, see liveModelField above), so its edit panel shows just the IR tab, no tab strip to switch.
function renderLiveEditTabs(pedal) {
  $("#liveEditToggle").textContent = t(pedal ? "cap.live.edit" : "cap.live.edit_ir");
  $("#liveEditTabs").hidden = !pedal;
  const tab = pedal ? S.liveEditTab : "ir";
  $("#liveTabNam").classList.toggle("primary", tab === "nam");
  $("#liveTabIr").classList.toggle("primary", tab === "ir");
  $("#liveEditNam").hidden = tab !== "nam";
  $("#liveEditIr").hidden = tab !== "ir";
}
function renderLive(liveRunning) {
  renderLivePicker("model"); renderLivePicker("ir");
  const pedal = gear() === "pedal";
  $("#liveHintHead").hidden = pedal; $("#liveHintPedal").hidden = !pedal;
  $("#liveModelField").hidden = !pedal;
  const together = isLiveTogether();
  $("#liveTogether").disabled = liveRunning;
  $("#liveCalibrated").disabled = liveRunning;
  const ready = liveReady();
  const takeBusy = S.take && (S.take.status === "recording" || S.take.status === "analyzing");
  const haveChoice = pedal ? (S.liveRef.model || S.liveRef.ir) : !!S.liveRef.ir;
  $("#liveStart").hidden = liveRunning; $("#liveStop").hidden = !liveRunning;
  $("#liveEditWrap").hidden = !liveRunning;
  if (!liveRunning) $("#liveEditPanel").hidden = true;
  renderLiveEditTabs(pedal);
  // the pickers, Sfoglia and Importa all stay usable while running: picking a different NAM/IR while
  // listening hot-swaps it (selectLiveRef -> /native/live/swap), it doesn't need Avvia pressed again
  const playerBusy = typeof playerRunning === "function" && playerRunning();
  $("#liveStart").disabled = !!op() || takeBusy || playerBusy || !ready.ok || !haveChoice;
  $("#liveNeeded").textContent = playerBusy ? t("cap.player.stop_first") : takeBusy ? t("cap.live.need_take_done")
    : !ready.ok ? (ready.missing === "clash" ? t("cap.player.out_clash", { n: ready.targets.outCh + 1 }) : ready.missing === "out" ? t("cap.live.need_out_return") : ready.missing === "guitar" ? t("cap.live.need_in_guitar") : t("cap.live.need_in_return"))
    : !haveChoice ? t("srv.cap.live_need_model_or_ir")
    : (ready.calibrated === false) ? t("cap.live.uncalibrated") : "";
  if (!liveRunning) { $("#liveInfo").textContent = ""; setMeters($("#metersLive"), []); return; }
  const st = (S.live && S.live.state) || {};
  $("#liveInfo").textContent = t("cap.live.running", { pre: f2(S.live.pre_gain_db), gain: f2(S.live.gain_db) }) +
    (S.live.together ? " — " + t("cap.live.guitar_running", { gain: f2(S.live.guitar_gain_db) }) : "") +
    (st.model_loaded ? " — " + t("cap.live.model_on") : "") + (st.ir_loaded ? " — " + t("cap.live.ir_on") : "") +
    (st.xruns ? " — " + t("cap.warn.xruns_live", { n: st.xruns }) : "");
  const rows = [[t("cap.live.return_in"), st.in_peak_dbfs], [t("cap.live.monitor_out"), st.out_peak_dbfs]];
  if (S.live.together) rows.push([t("cap.live.guitar_in"), st.guitar_in_peak_dbfs],
    [t(gear() === "pedal" ? "cap.live.guitar_out_pedal" : "cap.live.guitar_out_head"), st.guitar_out_peak_dbfs]);
  setMeters($("#metersLive"), rows);
}
async function liveStatePoll() {
  while (S.live && S.live.running) {
    try { S.live.state = await api("/api/capture/native/monitor/state"); } catch { /* transient: keep the last reading */ }
    if (!document.hidden) render();
    await new Promise((res) => setTimeout(res, 200));
  }
}
$("#liveRefresh").addEventListener("click", loadLiveFiles);
$("#liveModelTrigger").addEventListener("click", (e) => { e.stopPropagation(); toggleLiveMenu("model"); });
$("#liveIrTrigger").addEventListener("click", (e) => { e.stopPropagation(); toggleLiveMenu("ir"); });
$("#liveModelBrowse").addEventListener("click", () => pickFile("model", $("#liveModelFile"), (f) => uploadLiveFile("model", f)));
$("#liveIrBrowse").addEventListener("click", () => pickFile("ir", $("#liveIrFile"), (f) => uploadLiveFile("ir", f)));
$("#liveModelOpenFolder").addEventListener("click", async () => { try { await post("/api/capture/native/live/files/reveal?kind=model"); } catch (e) { setMsg("#liveMsg", e.message); } });
$("#liveIrOpenFolder").addEventListener("click", async () => { try { await post("/api/capture/native/live/files/reveal?kind=ir"); } catch (e) { setMsg("#liveMsg", e.message); } });
$("#liveModelFile").addEventListener("change", async () => { await uploadLiveFile("model", $("#liveModelFile").files[0]); $("#liveModelFile").value = ""; });
$("#liveIrFile").addEventListener("change", async () => { await uploadLiveFile("ir", $("#liveIrFile").files[0]); $("#liveIrFile").value = ""; });
$("#liveStart").addEventListener("click", async () => {
  setMsg("#liveMsg", "");
  const ready = liveReady();
  if (!ready.ok) return;
  const model = S.liveRef.model || null, ir = S.liveRef.ir || null;
  if (!model && !ir) return;
  const together = isLiveTogether();
  if (!confirm(t(together ? "cap.live.confirm_together" : "cap.live.confirm"))) return;
  const prev = snapshotOpen();
  const inChannels = together ? [ready.targets.guitarCh, ready.targets.returnCh] : [ready.targets.returnCh];
  const monitorOutChannels = together ? [ready.targets.monL, ...(ready.targets.monR != null ? [ready.targets.monR] : [])] : undefined;
  try {
    await claimDevice(ready.targets.device, inChannels, [ready.targets.outCh]);
    S.livePrevOpen = prev;
    const body = { confirm: true, model, ir, calibrated: ready.calibrated, master_db: parseFloat($("#liveMasterDb").value) };
    if (together) body.monitor_out_channels = monitorOutChannels;
    else body.listen_out_channels = [ready.targets.monL, ...(ready.targets.monR != null ? [ready.targets.monR] : [])];   // heard on the monitor outputs
    const r = await post("/api/capture/native/live/start", body);
    S.live = { running: true, together, pre_gain_db: r.pre_gain_db, gain_db: r.gain_db, guitar_gain_db: r.guitar_gain_db, state: null };
    resetLiveEditFields(r);
    liveStatePoll();
    render();
  } catch (e) {
    // no Python fallback exists for NAM+IR (real DSP, only ever built into the native engine): say plainly that
    // the real app needs to be open, instead of surfacing the raw connection error
    setMsg("#liveMsg", isNativeUnreachable(e) ? t("cap.live.need_native_app") : e.message);
    try { await restoreDevice(prev); } catch { /* best effort */ }
    S.livePrevOpen = null;
    render();
  }
});
$("#liveTogether").addEventListener("change", render);
$("#liveMonL").addEventListener("change", render);
$("#liveMonR").addEventListener("change", render);
$("#liveStop").addEventListener("click", async () => {
  try {
    if (S.live) S.live.running = false;                 // stop the poll loop first
    await post("/api/capture/native/live/stop?restore=0");   // the device is restored just below, once: no Python session in between
    S.live = null;
    const prev = S.livePrevOpen; S.livePrevOpen = null;
    await restoreDevice(prev);                              // the native stop hands Python back the live session's own channels (guitar + return): close them, then reopen only what was open before - never leave that pair behind, a capture would read the guitar
    render();
  } catch (e) { setMsg("#liveMsg", e.message); }
});

// "Edit NAM": the controls a real amp-modeler plugin has (input/output level, noise gate, 3-band tone stack),
// on top of whatever native_live_start/swap just computed. Input/output show the REAL current value (not a
// trim starting at 0) and moving them sets that absolute value, same as turning a real plugin's own knob -
// resetLiveEditFields re-syncs them whenever a start/swap recomputes the calibrated gains, so a manual edit
// never silently survives a model change that no longer matches it.
// Profile calibration: a .nam that carries input_level_dbu / output_level_dbu metadata can be locked to the levels
// calibration computes from them - input needs the model's input level AND the return's Ci, output needs the model's
// output level AND Co, so each side is available (and locked) on its own. Unticked, both steppers are free again.
function setLiveLevelLock(locked) {
  const a = S.live && S.live.avail || { input: false, output: false };
  for (const [side, id] of [["input", "liveInputDb"], ["output", "liveOutputDb"]]) {
    const lock = locked && a[side];
    $("#" + id).disabled = lock;
    if (KNOB_SYNC[id]) KNOB_SYNC[id]();
  }
}
// ---- rotary knobs: each is a real <input type=number> (kept for its value, typing and the existing "input" listeners) with
// a knob drawn over it. Turning the knob writes the input and fires "input", exactly as a stepper click used to.
const KNOB_GROUPS_NAM = [
  ["cap.live.sec_nam", [["liveInputDb", "cap.live.input_level", -24, 24, 0.5, 0], ["liveGateDb", "cap.live.gate", -100, 0, 1, -100], ["liveOutputDb", "cap.live.output_level", -24, 24, 0.5, 0]]],
  ["cap.live.sec_eq", [["liveEqBassDb", "cap.live.eq_bass", -12, 12, 0.5, 0], ["liveEqMidDb", "cap.live.eq_mid", -12, 12, 0.5, 0], ["liveEqTrebleDb", "cap.live.eq_treble", -12, 12, 0.5, 0]], "sliders"],
];
const KNOB_GROUPS_IR = [
  ["cap.live.sec_ir", [["liveIrInDb", "cap.live.ir_in", -24, 24, 0.5, 0], ["liveIrOutDb", "cap.live.ir_out", -24, 24, 0.5, 0]]],
];
const KNOB_SYNC = {};
// A slide fader (the EQ, like the tone sliders of the old radio): same contract as a knob - a real <input type=number> holds the
// value, dragging/wheel/keys write it and fire "input", scripts that assign .value move the cap.
function makeSlider(id, labelKey, min, max, step, def) {
  const input = el("input", { type: "number", id, min, max, step, value: def, className: "knob-num" });
  const cap = el("div", { className: "cap-s" });
  const slot = el("div", { className: "slot", tabIndex: 0, role: "slider" }, cap);
  const label = el("label", { htmlFor: id, className: "sl-name" }); label.dataset.i18n = labelKey; label.textContent = t(labelKey);
  const marks = [max, (max * 2) / 3, max / 3, 0, max / 3, (max * 2) / 3, max].map((v) => el("span", { textContent: String(Math.round(v)) }));
  const scale = el("div", { className: "sl-scale" }, ...marks);
  const clamp = (v) => Math.min(max, Math.max(min, Math.round(v / step) * step));
  const sync = () => {
    const v = parseFloat(input.value), frac = isFinite(v) ? (Math.min(max, Math.max(min, v)) - min) / (max - min) : 0.5;
    cap.style.left = `calc(${frac * 100}% - ${frac * 26}px)`;
    slot.setAttribute("aria-valuemin", min); slot.setAttribute("aria-valuemax", max); slot.setAttribute("aria-valuenow", isFinite(v) ? v : 0);
    slot.setAttribute("aria-label", t(labelKey));
  };
  const set = (v) => { const next = clamp(v); if (String(next) === input.value) return; input.value = next; sync(); input.dispatchEvent(new Event("input", { bubbles: true })); };
  const proto = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value");
  Object.defineProperty(input, "value", { configurable: true, get() { return proto.get.call(this); }, set(v) { proto.set.call(this, v); sync(); } });
  const fromPointer = (e) => { const r = slot.getBoundingClientRect(); return min + Math.min(1, Math.max(0, (e.clientX - r.left - 13) / (r.width - 26))) * (max - min); };
  let dragging = false;
  slot.addEventListener("pointerdown", (e) => { slot.setPointerCapture(e.pointerId); slot.focus(); dragging = true; set(fromPointer(e)); e.preventDefault(); });
  slot.addEventListener("pointermove", (e) => { if (dragging) set(fromPointer(e)); });
  slot.addEventListener("pointerup", () => { dragging = false; }); slot.addEventListener("pointercancel", () => { dragging = false; });
  slot.addEventListener("wheel", (e) => { e.preventDefault(); set((parseFloat(input.value) || 0) + (e.deltaY < 0 ? 1 : -1) * step * (e.shiftKey ? 1 : 2)); }, { passive: false });
  slot.addEventListener("keydown", (e) => {
    const dir = { ArrowUp: 1, ArrowRight: 1, ArrowDown: -1, ArrowLeft: -1 }[e.key];
    if (dir) { e.preventDefault(); set((parseFloat(input.value) || 0) + dir * step * (e.shiftKey ? 1 : 2)); }
  });
  slot.addEventListener("dblclick", () => set(def));
  input.addEventListener("input", sync);
  KNOB_SYNC[id] = sync;
  sync();
  return el("div", { className: "sl-field" }, scale, el("div", { className: "sl-ruler" }), slot, label, input);
}
function makeKnob(id, labelKey, min, max, step, def) {
  const input = el("input", { type: "number", id, min, max, step, value: def, className: "knob-num" });
  const cap = el("div", { className: "knob-cap" });
  const dial = el("div", { className: "knob", tabIndex: 0, role: "slider" }, cap);
  const label = el("label", { htmlFor: id }); label.dataset.i18n = labelKey; label.textContent = t(labelKey);
  const clamp = (v) => Math.min(max, Math.max(min, Math.round(v / step) * step));
  const sync = () => {
    const v = parseFloat(input.value), frac = isFinite(v) ? (Math.min(max, Math.max(min, v)) - min) / (max - min) : 0;
    cap.style.transform = `rotate(${-135 + 270 * frac}deg)`;
    dial.classList.toggle("disabled", input.disabled);
    dial.setAttribute("aria-valuemin", min); dial.setAttribute("aria-valuemax", max); dial.setAttribute("aria-valuenow", isFinite(v) ? v : 0);
    dial.setAttribute("aria-label", t(labelKey));
  };
  const set = (v) => { const next = clamp(v); if (String(next) === input.value) return; input.value = next; sync(); input.dispatchEvent(new Event("input", { bubbles: true })); };
  // a script that assigns input.value (resetLiveEditFields, the profile lock) turns the knob too
  const proto = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value");
  Object.defineProperty(input, "value", { configurable: true, get() { return proto.get.call(this); }, set(v) { proto.set.call(this, v); sync(); } });
  let drag = null;
  dial.addEventListener("pointerdown", (e) => {
    if (input.disabled) return;
    dial.setPointerCapture(e.pointerId); dial.focus();
    drag = { y: e.clientY, v: parseFloat(input.value) || 0 }; e.preventDefault();
  });
  dial.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const span = (e.shiftKey ? 900 : 180);                  // pixels for the whole range: Shift = fine
    set(drag.v + ((drag.y - e.clientY) / span) * (max - min));
  });
  const end = () => { drag = null; };
  dial.addEventListener("pointerup", end); dial.addEventListener("pointercancel", end);
  dial.addEventListener("wheel", (e) => { if (input.disabled) return; e.preventDefault(); set((parseFloat(input.value) || 0) + (e.deltaY < 0 ? 1 : -1) * step * (e.shiftKey ? 1 : 2)); }, { passive: false });
  dial.addEventListener("keydown", (e) => {
    if (input.disabled) return;
    const dir = { ArrowUp: 1, ArrowRight: 1, ArrowDown: -1, ArrowLeft: -1 }[e.key];
    if (dir) { e.preventDefault(); set((parseFloat(input.value) || 0) + dir * step * (e.shiftKey ? 1 : 2)); }
  });
  dial.addEventListener("dblclick", () => { if (!input.disabled) set(def); });
  input.addEventListener("input", sync);
  KNOB_SYNC[id] = sync;
  sync();
  return el("div", { className: "knob-field" }, label, dial, input);
}
function buildKnobGroup(groups) {
  return groups.map(([titleKey, knobs, kind]) => {
    const title = el("div", { className: "knob-title" }); title.dataset.i18n = titleKey; title.textContent = t(titleKey);
    const make = kind === "sliders" ? makeSlider : makeKnob;
    return el("div", { className: "knob-group" }, title, el("div", { className: kind === "sliders" ? "sliders" : "knobs" }, ...knobs.map((k) => make(...k))));
  });
}
function buildKnobs() {
  $("#liveKnobsNam").replaceChildren(...buildKnobGroup(KNOB_GROUPS_NAM));
  $("#liveKnobsIr").replaceChildren(...buildKnobGroup(KNOB_GROUPS_IR));
}
buildKnobs();
function resetLiveEditFields(r) {
  const avail = { input: r.model_input_level_dbu != null && r.ci_dbu != null, output: r.model_output_level_dbu != null && r.co_dbu != null };
  if (S.live) S.live.avail = avail;
  const any = avail.input || avail.output;
  $("#liveInputDb").value = f2(r.pre_gain_db);
  $("#liveOutputDb").value = f2(r.gain_db);
  $("#liveGateDb").value = -100;
  $("#liveEqBassDb").value = 0; $("#liveEqMidDb").value = 0; $("#liveEqTrebleDb").value = 0;
  $("#liveIrInDb").value = 0; $("#liveIrOutDb").value = 0;
  if (S.live) S.live.calLevels = { pre_gain_db: r.pre_gain_db, gain_db: r.gain_db };
  $("#liveProfileCal").disabled = !any;
  $("#liveProfileCal").checked = any;                       // available -> starts locked on the calibrated levels
  $("#liveProfileCalHint").textContent = any ? t("cap.live.profile_cal_hint", { input: avail.input ? "✓" : "–", output: avail.output ? "✓" : "–" }) : t("cap.live.profile_cal_none");
  setLiveLevelLock(any);
}
$("#liveProfileCal").addEventListener("change", () => {
  const on = $("#liveProfileCal").checked, cal = S.live && S.live.calLevels, a = S.live && S.live.avail;
  if (!cal || !a) return;
  if (on) {                                                 // back to the calibrated values, then lock
    const patch = {};
    if (a.input) { $("#liveInputDb").value = f2(cal.pre_gain_db); patch.pre_gain_db = cal.pre_gain_db; }
    if (a.output) { $("#liveOutputDb").value = f2(cal.gain_db); patch.gain_db = cal.gain_db; }
    if (Object.keys(patch).length) pushLiveControls(patch);
  }
  setLiveLevelLock(on);
});
$("#liveEditToggle").addEventListener("click", () => { $("#liveEditPanel").hidden = !$("#liveEditPanel").hidden; });
$("#liveTabNam").addEventListener("click", () => { S.liveEditTab = "nam"; render(); });
$("#liveTabIr").addEventListener("click", () => { S.liveEditTab = "ir"; render(); });
let liveControlsTimer = null, livePending = {};
function pushLiveControls(patch) {                          // patches within the debounce window are merged, none is lost
  Object.assign(livePending, patch);
  clearTimeout(liveControlsTimer);
  liveControlsTimer = setTimeout(() => {
    const body = livePending; livePending = {};
    post("/api/capture/native/live/controls", body).catch((e) => setMsg("#liveMsg", e.message));
  }, 40);
}
// The listening volume of the live test: on the page before and while playing, always starting at -18 dB, added to the output gain before the engine's ceiling.
$("#liveMasterKnob").replaceChildren(...buildKnobGroup([["cap.player.sec_master", [["liveMasterDb", "cap.player.master", -48, 12, 0.5, -18]]]]));
$("#liveMasterDb").value = f2(-18);
$("#liveMasterDb").addEventListener("input", () => { if (S.live && S.live.running) pushLiveControls({ master_db: parseFloat($("#liveMasterDb").value) }); });
$("#liveInputDb").addEventListener("input", () => pushLiveControls({ pre_gain_db: parseFloat($("#liveInputDb").value) }));
$("#liveOutputDb").addEventListener("input", () => pushLiveControls({ gain_db: parseFloat($("#liveOutputDb").value) }));
$("#liveGateDb").addEventListener("input", () => pushLiveControls({ gate_threshold_db: parseFloat($("#liveGateDb").value) }));
$("#liveEqBassDb").addEventListener("input", () => pushLiveControls({ eq_bass_db: parseFloat($("#liveEqBassDb").value) }));
$("#liveEqMidDb").addEventListener("input", () => pushLiveControls({ eq_mid_db: parseFloat($("#liveEqMidDb").value) }));
$("#liveEqTrebleDb").addEventListener("input", () => pushLiveControls({ eq_treble_db: parseFloat($("#liveEqTrebleDb").value) }));
$("#liveIrInDb").addEventListener("input", () => pushLiveControls({ ir_in_db: parseFloat($("#liveIrInDb").value) }));
$("#liveIrOutDb").addEventListener("input", () => pushLiveControls({ ir_out_db: parseFloat($("#liveIrOutDb").value) }));

// ------------------------------------------------------------------ model name helper
// Builds "Nome del modello" from a few common amp knobs, abbreviated (Roman: "basse 3 alte 5 medie 7" -> "B3 - A5 - M7"),
// plus any custom control named by hand - an empty field is simply left out of the name, for every field alike. Purely a
// client-side convenience (the server only ever sees the final free-text name): per-amp custom control labels are kept in
// localStorage, so re-capturing the same amp later offers the same labels again ("richiamabili"), ready for a new value.
function loadAmpProfiles() { try { return JSON.parse(localStorage.getItem("ampProfiles") || "{}"); } catch { return {}; } }
function saveAmpProfiles(p) { try { localStorage.setItem("ampProfiles", JSON.stringify(p)); } catch {} }
const ampKey = (name) => name.trim().toLowerCase();

function refreshAmpNameList() {
  const names = Object.values(loadAmpProfiles()).map((p) => p.name).sort((a, b) => a.localeCompare(b));
  $("#ampNameList").replaceChildren(...names.map((n) => el("option", { value: n })));
}
function customEntries() {
  return [...$("#ampCustomRows").children].map((row) => ({ label: row.children[0].value.trim(), value: row.children[1].value.trim() }))
    .filter((c) => c.label && c.value);
}
function addCustomRow(label, value) {
  const labelInput = el("input", { type: "text", value: label || "" });
  labelInput.placeholder = t("cap.rec.custom_label_ph"); labelInput.style.width = "150px";
  const valueInput = el("input", { type: "text", value: value || "" });
  valueInput.placeholder = t("cap.rec.custom_value_ph"); valueInput.style.width = "110px";
  const remove = el("button", { type: "button", className: "small", textContent: "×" });
  const row = el("div", { className: "field-inline" }, labelInput, valueInput, remove);
  remove.addEventListener("click", () => { row.remove(); updateModelName(); });
  labelInput.addEventListener("input", updateModelName);
  valueInput.addEventListener("input", updateModelName);
  $("#ampCustomRows").appendChild(row);
  return labelInput;
}
// The knob set depends on what's being captured (Roman: head knobs are bass/mid/treble/presence, a pedal's are
// gain/volume/tone instead) - renderAmpFields() shows/hides the right fields, this just reads whichever are visible.
function buildGeneratedName() {
  const amp = $("#ampName").value.trim();
  const tokens = [];
  const knob = (id, letter) => { const v = $("#" + id).value.trim(); if (v !== "") tokens.push(letter + v); };
  if (gear() === "pedal") {
    knob("ampGain", "G"); knob("ampVolume", "V"); knob("ampTone", "T");
    knob("ampTreble", "A"); knob("ampBass", "B"); knob("ampMid", "M");
  } else {
    const ch = $("#ampChannel").value.trim().replace(/^ch\s*/i, "");      // "2" or "CH2" -> "CH2"
    if (ch) tokens.push("CH" + ch);
    knob("ampGain", "G"); knob("ampBass", "B"); knob("ampTreble", "A"); knob("ampMid", "M"); knob("ampPresence", "P");
  }
  for (const c of customEntries()) tokens.push(`${c.label}:${c.value}`);
  return [amp, tokens.join(" - ")].filter(Boolean).join(" - ");
}
function renderAmpFields() {
  const pedal = gear() === "pedal";
  $("#ampNameLabel").textContent = t(pedal ? "cap.rec.pedal_name" : "cap.rec.amp");
  $("#ampName").placeholder = t(pedal ? "cap.rec.pedal_name_ph" : "cap.rec.amp_ph");
  $("#ampChannelField").hidden = pedal;
  $("#ampPresenceField").hidden = pedal;
  $("#ampVolumeField").hidden = !pedal;
  $("#ampToneField").hidden = !pedal;
}
function saveCurrentAmpProfile() {
  const name = $("#ampName").value.trim();
  if (!name) return;
  const profiles = loadAmpProfiles(), key = ampKey(name);
  const existing = profiles[key] || { name, customLabels: [] };
  profiles[key] = { name, customLabels: Array.from(new Set([...existing.customLabels, ...customEntries().map((c) => c.label)])) };
  saveAmpProfiles(profiles);
  refreshAmpNameList();
}
function updateModelName() {
  const generated = buildGeneratedName();
  if (generated) $("#modelName").value = generated;
  saveCurrentAmpProfile();
  saveNameForm();
}
// The fields of the name stay as they were after a capture ("Nuova cattura" reloads the page) and after a restart: a series of captures
// usually changes one knob at a time. "Pulisci" empties them.
const NAME_FIELDS = ["ampName", "ampChannel", "ampGain", "ampVolume", "ampTone", "ampBass", "ampMid", "ampTreble", "ampPresence", "modelName"];
function saveNameForm() {
  const form = Object.fromEntries(NAME_FIELDS.map((id) => [id, $("#" + id).value]));
  form.custom = [...$("#ampCustomRows").children].map((row) => [row.children[0].value, row.children[1].value]);
  try { localStorage.setItem("nameForm", JSON.stringify(form)); } catch {}
}
function restoreNameForm() {
  let form = null;
  try { form = JSON.parse(localStorage.getItem("nameForm") || "null"); } catch {}
  if (!form) return;
  NAME_FIELDS.forEach((id) => { if (typeof form[id] === "string") $("#" + id).value = form[id]; });
  (form.custom || []).forEach(([label, value]) => addCustomRow(label, value));
}
$("#modelName").addEventListener("input", saveNameForm);
$("#ampClear").addEventListener("click", () => {
  NAME_FIELDS.forEach((id) => { $("#" + id).value = ""; });
  $("#ampCustomRows").replaceChildren();
  saveNameForm();
});
["ampChannel", "ampBass", "ampMid", "ampTreble", "ampPresence", "ampGain", "ampVolume", "ampTone"].forEach((id) => $("#" + id).addEventListener("input", updateModelName));
$("#ampName").addEventListener("input", updateModelName);
$("#ampName").addEventListener("change", () => {
  // recall: an amp already used before offers its custom control labels again, ready to fill in for this capture
  const profile = loadAmpProfiles()[ampKey($("#ampName").value)];
  if (profile && profile.customLabels.length && !$("#ampCustomRows").children.length) profile.customLabels.forEach((label) => addCustomRow(label, ""));
  updateModelName();
});
$("#ampAddCustom").addEventListener("click", () => addCustomRow("", "").focus());

// ------------------------------------------------------------------ capture
async function refreshInputs() {
  try {
    const r = await api("/api/capture/input");
    S.inputs = r.found;
    const sel = $("#inputRef"), keep = sel.value;
    sel.replaceChildren(...r.found.map((f) => el("option", { value: f.rel, textContent: f.rel })));
    if (keep && r.found.some((f) => f.rel === keep)) sel.value = keep;
    $("#inputMissing").hidden = r.found.length > 0;
  } catch (e) { /* the page still works without */ }
}
$("#inputBrowse").addEventListener("click", () => pickFile("input", $("#inputFile"), useInputFile));
$("#inputFolderOpen").addEventListener("click", async () => {
  try { await post("/api/capture/captures/reveal?folder=input"); } catch (e) { setMsg("#recMsg", e.message); }
});
async function useInputFile(file) {
  if (!file) return;
  setMsg("#recMsg", "");
  const fd = new FormData(); fd.append("file", file);
  try {
    const r = await api("/api/capture/input/upload", { method: "POST", body: fd });
    await refreshInputs();
    $("#inputRef").value = r.rel;                      // the file just chosen is the one to use
  } catch (err) { setMsg("#recMsg", err.message); }
}
$("#inputFile").addEventListener("change", async (e) => { await useInputFile(e.target.files[0]); e.target.value = ""; });
$("#levelTest").addEventListener("click", () => longCall($("#levelTest"), "#recMsg", async () => {
  const trim = parseFloat($("#trim").value) || 0;
  if (!confirm(t("cap.test.confirm", { level: trim, out: opened().out_channels[0] + 1 }))) return;
  clearLevelTest(); $("#testText").textContent = t("cap.test.running");
  await demoRoute("gear");
  await ensureCaptureDevice();
  const r = await post("/api/capture/level-test", { confirm: true, send_gain_db: trim });
  S.levelTest = { trim, judge: r.judge };
  showLevelTest(r);
}));
$("#recStart").addEventListener("click", async () => {
  setMsg("#recMsg", "");
  const trimNow = parseFloat($("#trim").value) || 0;
  if (S.levelTest && S.levelTest.trim === trimNow && S.levelTest.judge === "clip" && !confirm(t("cap.test.warn_clip"))) return;
  if (!S.levelTest && !confirm(t("cap.test.not_done"))) return;
  if (!confirm(t("cap.rec.confirm"))) return;
  const warmup = $("#recWarmup").checked;                  // read now: what the page shows is what the capture does
  try {
    await demoRoute("gear");
    await ensureCaptureDevice();
    await post("/api/capture/record/start", { gear: gear(), name: $("#modelName").value, input_ref: $("#inputRef").value, calibrated: isCalibrated(), send_gain_db: parseFloat($("#trim").value) || 0,
                                              warmup });
  } catch (e) { setMsg("#recMsg", e.message); }
});
// The warm-up (blips until the driver's one-sample step is over) is only for interfaces that do it: OFF every time the program starts. Within one run it is
// kept (sessionStorage: it survives "Nuova cattura", which reloads the page, but not closing the program). One value, NOT tied to the interface that is open: the
// capture re-opens the interface right before recording, and a value looked up per interface was reset in that moment (a capture went out without the warm-up).
function loadWarmup() { try { $("#recWarmup").checked = sessionStorage.getItem("warmup") === "1"; } catch { $("#recWarmup").checked = false; } }
$("#recWarmup").addEventListener("change", () => { try { sessionStorage.setItem("warmup", $("#recWarmup").checked ? "1" : "0"); } catch {} });
loadWarmup();
$("#recStop").addEventListener("click", async () => { try { await post("/api/capture/record/stop"); } catch (e) { setMsg("#recMsg", e.message); } });

// ------------------------------------------------------------------ result
const BLOCKING = new Set(["no_blip", "clip", "short_recording", "esr_high", "esr_glitch", "timing_step", "xruns"]);
// Once a take is saved the server forgets it (S.take goes back to null), but the page must not: this card is where the capture is sent to
// the trainer. S.saved keeps the take and what saving answered until a new recording starts or the page is reloaded ("Nuova cattura").
function slipText(n) { return n === 1 ? t("cap.res.one_sample") : t("cap.res.n_samples", { n }); }
function warnText(w, r) {
  const d = r.replicate || {}, str = d.stretch;
  const fix = d.repaired;
  const params = { esr: r.replicate_esr.toFixed(4), rest: d.esr_rest == null ? "–" : d.esr_rest.toFixed(4), from: str ? f1(str[0]) : "", to: str ? f1(str[1]) : "",
                   slip: d.slip ? slipText(Math.abs(d.slip)) : fix ? slipText(Math.abs(fix.slip)) : "", until: fix ? f1(fix.until_s) : "", before: fix ? fix.esr_before.toFixed(4) : "" };
  const text = t(w === "esr_glitch" && !d.slip ? "cap.warn.esr_stretch" : "cap.warn." + w, params);
  // the driver stepped and the warm-up was off: say where the switch is
  return (w === "esr_repaired" || w === "timing_step") && !(r.timing && r.timing.reason) ? text + " " + t("cap.warn.try_warmup") : text;
}
function timingText(tm) {
  // what the blips around the input said about the interface's timing (alignment.WarmupWatch / timing_check)
  if (!tm.checked) return t("cap.res.timing_unchecked", { w: tm.warmup_s == null ? "–" : f1(tm.warmup_s) });
  if (tm.moved_in_training) return t("cap.res.timing_moved", { b: f2(tm.between_blips), a: tm.after_input == null ? "–" : f2(tm.after_input) });
  if (!tm.reason) return t("cap.res.timing_steady_off");          // no warm-up: only the check after the input ran
  const steps = tm.steps || [];
  if (steps.length) return t("cap.res.timing_step_seen", { at: f1(steps[steps.length - 1][0]), slip: slipText(Math.abs(steps[steps.length - 1][1])), w: f1(tm.warmup_s) });
  if (tm.before_input != null && Math.abs(tm.before_input) >= 0.5) return t("cap.res.timing_step_first", { w: f1(tm.warmup_s) });
  return t("cap.res.timing_steady", { w: tm.warmup_s == null ? "–" : f1(tm.warmup_s) });
}
function applySaved() {
  const h = S.saved.handoff;
  $("#resActions").hidden = true;
  $("#resSaved").hidden = false;
  $("#resSaved").textContent = t("cap.res.saved_at", { path: h.output_rel });
  $("#resTrainerWrap").hidden = !h.folder_rel;
  $("#resSendTrainer").textContent = t(S.saved.sent ? "cap.trainer.sent" : "cap.res.send_trainer");
  $("#resSendTrainer").disabled = !!S.saved.sent;
  $("#resNewWrap").hidden = false;
  renderTrainerStatus();
}
function renderResult() {
  if (S.take && (S.take.status === "recording" || S.take.status === "analyzing")) S.saved = null;
  const take = S.take && S.take.status === "ready" ? S.take : (S.saved && S.saved.take), card = $("#cardRes");
  card.hidden = !take;
  if (!take) { card.dataset.id = ""; return; }
  if (card.dataset.id === take.id + LANG + (S.saved ? "s" : "")) return;
  card.dataset.id = take.id + LANG + (S.saved ? "s" : "");
  const r = take.report, md = take.metadata || {};
  const rows = [
    [t("cap.res.latency"), r.latency.delay == null ? "–" : t("cap.res.latency_v", { samples: r.latency.delay, ms: f2(r.latency.ms), cut: r.latency.cut })],
    [t("cap.res.length"), t("cap.res.length_v", { n: r.n_input.toLocaleString(locale()), s: (r.n_input / 48000).toFixed(1) })],
    [t("cap.res.residual"), r.residual ? t("cap.res.residual_v", { d: r.residual.delay, rec: r.residual.recommended }) : "–"],
    [t("cap.res.esr"), `${r.replicate_esr.toFixed(5)}  (${t("cap.res.limit")} 0.01)`],
    ...(r.replicate && r.replicate.stretch ? [[t("cap.res.stretch"), t("cap.res.stretch_v", { from: f1(r.replicate.stretch[0]), to: f1(r.replicate.stretch[1]),
      slip: r.replicate.slip ? slipText(Math.abs(r.replicate.slip)) : "–", rest: r.replicate.esr_rest == null ? "–" : r.replicate.esr_rest.toFixed(4) })]] : []),
    ...(r.timing ? [[t("cap.res.timing"), timingText(r.timing)]] : []),
    [t("cap.res.peak"), `${f1(r.peak_dbfs)} dBFS`],
    [t("cap.res.noise"), `${f1(r.noise_dbfs)} dBFS`],
    [t("cap.res.blip"), r.blip_over_noise_db == null ? "–" : `${f1(r.blip_over_noise_db)} dB`],
    [t("cap.res.levels"), md.input_level_dbu == null ? t("cap.res.levels_none") : md.output_level_dbu == null ? t("cap.res.levels_in_only", { i: f2(md.input_level_dbu) }) : t("cap.res.levels_v", { i: f2(md.input_level_dbu), o: f2(md.output_level_dbu) })],
  ];
  const table = el("table", { className: "report" }, ...rows.map(([k, v]) => el("tr", {}, el("td", { textContent: k }), el("td", { textContent: v }))));
  const warns = [...r.warnings, ...(take.warnings || [])];
  const list = el("ul", { className: "warn-list" }, ...warns.map((w) => el("li", { className: BLOCKING.has(w) ? "bad" : "warn", textContent: warnText(w, r) })));
  $("#resBody").replaceChildren(el("div", { className: "result-line " + (r.ok ? "good" : "bad"), textContent: t(r.ok ? "cap.res.ok" : "cap.res.flagged") }), table, list);
  $("#resSend").textContent = t(r.ok ? "cap.res.save" : "cap.res.save_anyway");
  if (S.saved) applySaved();
  else {                                                  // a fresh take after a saved one: back to "save / discard"
    $("#resActions").hidden = false; $("#resSaved").hidden = true; $("#resTrainerWrap").hidden = true; $("#resNewWrap").hidden = true;
    $("#resSendTrainer").disabled = false;
  }
}
$("#resSend").addEventListener("click", async () => {
  setMsg("#resMsg", "");
  try {
    const flagged = S.take && S.take.report && !S.take.report.ok;
    if (flagged && !confirm(t("cap.res.send_confirm"))) return;
    const take = S.take;
    const h = await post("/api/capture/take/accept", { name: $("#modelName").value, force: !!flagged });
    S.lastCaptureFolder = h.folder_rel;
    S.saved = { take, handoff: h, sent: false };
    applySaved();
  } catch (e) { setMsg("#resMsg", e.message); }
});
$("#resSendTrainer").addEventListener("click", async () => {
  setMsg("#resTrainerMsg", "");
  $("#resSendTrainer").disabled = true;
  try {
    await post("/api/capture/trainer/send", { folder_rel: S.lastCaptureFolder });
    if (S.saved) S.saved.sent = true;
    $("#resSendTrainer").textContent = t("cap.trainer.sent");
  } catch (e) { setMsg("#resTrainerMsg", e.message); $("#resSendTrainer").disabled = false; }
});
$("#resNew").addEventListener("click", () => location.reload());
$("#resDiscard").addEventListener("click", async () => { try { await post("/api/capture/take/discard"); $("#cardRes").hidden = true; } catch (e) { setMsg("#resMsg", e.message); } });

// ------------------------------------------------------------------ trainer hand-off (live network detection)
// A light, not a folder (Roman: "non voglio cartelle condivise, voglio una luce... se rileva il NAM Trainer in
// rete"): the page polls GET /trainer/status every few seconds (independent of the fast render() loop - a network
// probe has no business running 4x/s) and lights the dot in every place that cares. "Invia" only ever appears once
// the light is green. window.trainerSetHost / window.trainerBrowseFolder are called only from packaging/main.swift's
// File menu, after a native dialog gets a real folder path or the trainer's address - nothing on the page itself
// triggers these.
function setLamp(dotSel, textSel, online, host, local) {
  $(dotSel).className = "dot " + (online ? "ok" : "");
  $(textSel).textContent = t(online ? "cap.trainer.online" : "cap.trainer.offline") + (host ? ` (${host})` : "") + (!online && local ? " " + t("cap.trainer.local_hint") : "");
}
function renderTrainerStatus() {
  const s = S.trainerStatus;
  const online = !!(s && s.online);
  setLamp("#recTrainerDot", "#recTrainerText", online, s && s.host, s && s.local_online);
  setLamp("#resTrainerDot", "#resTrainerText", online, s && s.host, s && s.local_online);
  $("#resSendTrainer").hidden = !online;
  document.querySelectorAll("#trainerList [data-sendable]").forEach((btn) => { btn.disabled = !online || btn.dataset.sendable !== "1"; });
}
async function trainerStatusPoll() {
  try { S.trainerStatus = await api("/api/capture/trainer/status"); } catch { S.trainerStatus = null; }
  renderTrainerStatus();
  setTimeout(trainerStatusPoll, 4000);
}
window.trainerSetHost = async function (host) {
  try {
    const r = await post("/api/capture/trainer/host", { host });
    S.trainerStatus = r;
    renderTrainerStatus();
    alert(t("cap.trainer.host_set", { host: r.host }));
  } catch (e) { alert(e.message); }
};
function renderTrainerList(captures) {
  const list = $("#trainerList");
  if (!captures.length) { list.replaceChildren(el("div", { className: "hint", textContent: t("cap.trainer.empty") })); return; }
  list.replaceChildren(...captures.map((c) => {
    const label = (c.name || "?") + (c.created ? " — " + fmtDate(c.created) : "") + (c.exportable ? "" : ` (${t("cap.trainer.not_exportable")})`);
    const line = el("span", { textContent: label });
    const btn = el("button", { type: "button", textContent: t("cap.trainer.send") });
    btn.dataset.sendable = c.exportable ? "1" : "0";
    btn.disabled = !c.exportable || !(S.trainerStatus && S.trainerStatus.online);
    btn.addEventListener("click", async () => {
      btn.disabled = true; setMsg("#trainerMsg", "");
      try { await post("/api/capture/trainer/send", { dir: c.dir }); btn.textContent = t("cap.trainer.sent"); }
      catch (e) { setMsg("#trainerMsg", e.message); btn.disabled = false; }
    });
    return el("div", { className: "field-inline" }, line, btn);
  }));
}
window.trainerBrowseFolder = async function (path) {
  setMsg("#trainerMsg", "");
  $("#cardTrainerBrowse").hidden = false;
  try {
    const r = await post("/api/capture/trainer/scan", { path });
    renderTrainerList(r.captures);
    $("#cardTrainerBrowse").scrollIntoView({ behavior: "smooth" });
  } catch (e) { renderTrainerList([]); setMsg("#trainerMsg", e.message); }
};
$("#trainerClose").addEventListener("click", () => { $("#cardTrainerBrowse").hidden = true; });

// ------------------------------------------------------------------ language + boot
// The header knob is the visible language switch (the <select> stays as the state holder): a click turns it to the other language.
function syncLangKnob() {
  const cap = $("#langKnob .knob-cap");
  if (cap) cap.style.transform = `rotate(${LANG === "it" ? -42 : 42}deg)`;
  $("#langNow").textContent = LANG.toUpperCase();
}
$("#langKnob").addEventListener("click", () => {
  const next = LANG === "it" ? "en" : "it";
  $("#langSelect").value = next;
  $("#langSelect").dispatchEvent(new Event("change"));
});
$("#langSelect").addEventListener("change", (e) => {
  LANG = e.target.value in LANGS ? e.target.value : "en";
  try { localStorage.setItem("lang", LANG); } catch {}
  applyStatic(); fillChannels(deviceById($("#devSelect").value)); $("#cardRes").dataset.id = ""; render();
});
document.querySelectorAll("#calibrated, input[name=gear]").forEach((e) => e.addEventListener("change", render));
document.querySelectorAll("input[name=gear]").forEach((e) => e.addEventListener("change", updateModelName));   // the knob set just changed (bass/mid/treble/presence vs gain/volume/tone): refresh the generated name right away

(async function boot() {
  try { I18N = await (await fetch("/i18n.json", { cache: "no-cache" })).json(); } catch {}
  applyStatic();
  restoreNameForm();
  await poll();
  await loadDevices();
  await refreshInputs();
  await loadLiveFiles();
  refreshAmpNameList();
  trainerStatusPoll();
})();
