// Label-review app: table of keyframes + keyboard labeling over the 64 frames (4 fps) of each.
// Data from GET /api/items (items.json of tools/export_label_review.py); decisions go to
// POST /api/review, which appends them to revisions.jsonl.
"use strict";

const $ = (s) => document.querySelector(s);
const S = {
  meta: null, items: [], reviews: {}, rows: [], sortKey: "patient", sortDir: 1,
  queue: [], pos: 0, cls: 0, off: 0, busy: false,
};
const STATUS_IT = { todo: "da fare", confirmed: "confermato", modified: "modificato", skipped: "saltato" };

const pretty = (n) => n.replace(/_/g, " ");
const hms = (t) => {
  const s = Math.max(0, t), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  return `${h}:${String(m).padStart(2, "0")}:${r.toFixed(2).padStart(5, "0")}`;
};
const esc = (x) => String(x).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function statusOf(rev) {
  if (!rev) return "todo";
  if (rev.action === "skipped") return "skipped";
  return rev.action === "confirmed" ? "confirmed" : "modified";
}

function decorate(it) {
  const rev = S.reviews[it.id];
  it.trueName = S.meta.names[it.true];
  it.predName = S.meta.names[it.pred];
  it.status = statusOf(rev);
  it.newName = rev && rev.action !== "skipped" && rev.action !== "confirmed"
    ? `${pretty(S.meta.names[rev.new_class])}${rev.offset_frames_4fps ? ` (${rev.offset_frames_4fps > 0 ? "+" : ""}${rev.offset_frames_4fps} fr)` : ""}`
    : "";
}

async function load() {
  const r = await fetch("api/items", { cache: "no-store" });
  if (!r.ok) throw new Error(`api/items: ${r.status}`);
  const d = await r.json();
  S.meta = d; S.items = d.items; S.reviews = d.reviews || {};
  S.items.forEach(decorate);
  const opt = (sel, vals, label) => {
    $(sel).innerHTML = `<option value="">tutti</option>` +
      vals.map((v) => `<option value="${esc(v)}">${esc(label ? label(v) : v)}</option>`).join("");
  };
  const clsIdx = d.names.map((_, i) => i).sort((a, b) => d.names[a].localeCompare(d.names[b]));
  opt("#f-true", clsIdx, (i) => pretty(d.names[i]));
  opt("#f-pred", clsIdx, (i) => pretty(d.names[i]));
  opt("#f-patient", [...new Set(S.items.map((i) => i.patient))].sort());
  opt("#f-fold", [...new Set(S.items.map((i) => i.fold))].sort());
  $("#legend").innerHTML = d.names.map((n, i) =>
    `<li data-c="${i}"><kbd>${esc(d.keys[i])}</kbd> ${esc(pretty(n))}</li>`).join("");
  restoreFilters();
  renderTable();
}

// ----------------------------------------------------------------------------------- table
const FILTERS = ["#f-true", "#f-pred", "#f-correct", "#f-cmin", "#f-cmax", "#f-patient", "#f-fold", "#f-type", "#f-status"];

function saveFilters() {
  try {
    const f = Object.fromEntries(FILTERS.map((s) => [s, $(s).value]));
    localStorage.setItem("lr-filters", JSON.stringify({ f, k: S.sortKey, d: S.sortDir }));
  } catch (e) { /* storage unavailable */ }
}
function restoreFilters() {
  try {
    const x = JSON.parse(localStorage.getItem("lr-filters") || "null");
    if (!x) return;
    FILTERS.forEach((s) => { if (x.f[s] !== undefined) $(s).value = x.f[s]; });
    S.sortKey = x.k || S.sortKey; S.sortDir = x.d || 1;
  } catch (e) { /* ignore */ }
}

function filtered() {
  const v = (s) => $(s).value;
  const cmin = parseFloat(v("#f-cmin")), cmax = parseFloat(v("#f-cmax"));
  return S.items.filter((it) =>
    (v("#f-true") === "" || it.true === +v("#f-true")) &&
    (v("#f-pred") === "" || it.pred === +v("#f-pred")) &&
    (v("#f-correct") === "" || it.correct === (v("#f-correct") === "1")) &&
    (isNaN(cmin) || it.conf >= cmin) && (isNaN(cmax) || it.conf <= cmax) &&
    (v("#f-patient") === "" || it.patient === v("#f-patient")) &&
    (v("#f-fold") === "" || it.fold === +v("#f-fold")) &&
    (v("#f-type") === "" || it.type === v("#f-type")) &&
    (v("#f-status") === "" || it.status === v("#f-status")));
}

function renderTable() {
  const k = S.sortKey, d = S.sortDir;
  S.rows = filtered().sort((a, b) => {
    const x = a[k], y = b[k];
    const c = typeof x === "string" ? x.localeCompare(y) : (x > y) - (x < y);
    return c * d || a.patient.localeCompare(b.patient) || a.second - b.second;
  });
  document.querySelectorAll("#items th").forEach((th) => {
    th.classList.toggle("asc", th.dataset.k === k && d > 0);
    th.classList.toggle("desc", th.dataset.k === k && d < 0);
  });
  const nRev = Object.values(S.reviews).filter((r) => r.action !== "skipped").length;
  $("#summary").textContent = `${S.items.length} keyframe · ${nRev} revisionati`;
  $("#count").textContent = `${S.rows.length} righe filtrate`;
  $("#items tbody").innerHTML = S.rows.map((it, i) => `<tr data-i="${i}" class="${it.correct ? "" : "wrong"}">
    <td>${it.patient}</td><td>${it.fold}</td><td>${hms(it.second)}</td>
    <td>${it.type === "event" ? "evento" : "controllo"}</td>
    <td>${esc(pretty(it.trueName))}</td><td>${esc(pretty(it.predName))}</td>
    <td>${it.conf.toFixed(2)}</td><td>${it.p_true.toFixed(2)}</td><td>${it.correct ? "sì" : "no"}</td>
    <td class="st-${it.status}">${STATUS_IT[it.status]}</td><td>${esc(it.newName)}</td></tr>`).join("");
  saveFilters();
}

// -------------------------------------------------------------------------------- labeling
function startLabeling(rows, pos) {
  if (!rows.length) return;
  S.queue = rows.slice(); S.pos = pos || 0;
  $("#table-view").hidden = true; $("#label-view").hidden = false;
  openItem();
}

function backToTable(msg) {
  $("#label-view").hidden = true; $("#table-view").hidden = false;
  renderTable();
  if (msg) $("#count").textContent += ` · ${msg}`;
}

function frameUrl(it, i) {
  return `frames/${it.frames}/${String(i).padStart(2, "0")}.jpg`;
}

function openItem() {
  const it = S.queue[S.pos];
  const rev = S.reviews[it.id];
  const keep = rev && rev.action !== "skipped";
  S.cls = keep ? rev.new_class : it.true;
  S.off = keep ? rev.offset_frames_4fps : 0;
  const n = S.meta.n_frames;
  $("#grid").innerHTML = Array.from({ length: n }, (_, i) =>
    `<img data-i="${i}" src="${frameUrl(it, i)}" loading="eager" alt="" class="${i === S.meta.key_index ? "orig" : ""}">`).join("");
  $("#saved-msg").textContent = "";
  for (let k = 1; k <= 2 && S.pos + k < S.queue.length; k++) {  // preload the next items
    const nx = S.queue[S.pos + k];
    for (let i = 0; i < n; i += 1) new Image().src = frameUrl(nx, i);
  }
  render();
}

function render() {
  const it = S.queue[S.pos], m = S.meta, cur = m.key_index + S.off;
  $("#progress").textContent = `${S.pos + 1} / ${S.queue.length}`;
  $("#item-title").textContent = `${it.patient} · fold ${it.fold} · ${hms(it.second)} · ${it.type === "event" ? "evento" : "controllo null"}`;
  document.querySelectorAll("#grid img").forEach((im) => im.classList.toggle("cur", +im.dataset.i === cur));
  $("#preview").src = frameUrl(it, cur);
  const changed = S.cls !== it.true;
  const top3 = it.top3.map(([c, p]) => `${esc(pretty(m.names[c]))} ${p.toFixed(2)}`).join("<br>");
  const rev = S.reviews[it.id];
  $("#info").innerHTML = `
    <dt>Classe reale</dt><dd>${esc(pretty(m.names[it.true]))}</dd>
    <dt>Classe assegnata</dt><dd class="${changed ? "changed" : ""}">${esc(pretty(m.names[S.cls]))}${changed ? " (modificata)" : ""}</dd>
    <dt>Predetta</dt><dd class="${it.correct ? "" : "wrongtxt"}">${esc(pretty(m.names[it.pred]))} · conf ${it.conf.toFixed(2)}</dd>
    <dt>P(reale)</dt><dd>${it.p_true.toFixed(2)}</dd>
    <dt>Prime 3</dt><dd>${top3}</dd>
    <dt>Spostamento</dt><dd class="${S.off ? "changed" : ""}">${S.off > 0 ? "+" : ""}${S.off} frame (${S.off >= 0 ? "+" : ""}${(S.off / m.fps).toFixed(2)} s) → ${hms(it.second + S.off / m.fps)}</dd>
    <dt>Stato</dt><dd>${STATUS_IT[statusOf(rev)]}</dd>`;
  document.querySelectorAll("#legend li").forEach((li) => {
    li.classList.toggle("sel", +li.dataset.c === S.cls);
    li.classList.toggle("orig", +li.dataset.c === it.true);
  });
}

async function submit(skipped) {
  if (S.busy) return;
  S.busy = true;
  const it = S.queue[S.pos];
  try {
    const r = await fetch("api/review", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: it.id, new_class: S.cls, offset_frames_4fps: S.off, skipped }),
    });
    const rec = await r.json();
    if (!r.ok) throw new Error(rec.error || r.status);
    if (!(skipped && S.reviews[it.id] && S.reviews[it.id].action !== "skipped")) S.reviews[it.id] = rec;
    decorate(it);
    next(1);
  } catch (e) {
    $("#saved-msg").textContent = `Errore nel salvataggio: ${e.message}`;
  } finally {
    S.busy = false;
  }
}

function next(step) {
  const p = S.pos + step;
  if (p >= S.queue.length) return backToTable("fine della lista");
  if (p < 0) return;
  S.pos = p;
  openItem();
}

document.addEventListener("keydown", (e) => {
  if ($("#label-view").hidden || e.ctrlKey || e.metaKey || e.altKey) return;
  const m = S.meta, k = e.key;
  if (k === "ArrowLeft" || k === "ArrowRight") {
    const d = (k === "ArrowLeft" ? -1 : 1) * (e.shiftKey ? 4 : 1);
    S.off = Math.min(m.n_frames - 1 - m.key_index, Math.max(-m.key_index, S.off + d));
  } else if (k === "Enter") {
    submit(false);
  } else if (k === " ") {
    submit(true);
  } else if (k === "Escape") {
    backToTable();
  } else if (k === "Backspace") {
    next(-1);
  } else {
    const c = m.keys.indexOf(k.toLowerCase());
    if (c < 0) return;
    S.cls = c;
  }
  e.preventDefault();
  if (!$("#label-view").hidden) render();
});

document.querySelector("#items thead").addEventListener("click", (e) => {
  const k = e.target.dataset.k;
  if (!k) return;
  S.sortDir = S.sortKey === k ? -S.sortDir : 1;
  S.sortKey = k;
  renderTable();
});
document.querySelector("#items tbody").addEventListener("click", (e) => {
  const tr = e.target.closest("tr");
  if (tr) startLabeling(S.rows, +tr.dataset.i);
});
document.querySelector("#grid").addEventListener("click", (e) => {
  const i = e.target.dataset && e.target.dataset.i;
  if (i === undefined) return;
  const m = S.meta;
  S.off = +i - m.key_index;
  render();
});
FILTERS.forEach((s) => $(s).addEventListener("input", renderTable));
$("#f-reset").addEventListener("click", () => {
  FILTERS.forEach((s) => { $(s).value = s === "#f-cmin" ? "0" : s === "#f-cmax" ? "1" : ""; });
  renderTable();
});
$("#start").addEventListener("click", () => startLabeling(S.rows, 0));

load().catch((e) => { $("#count").textContent = `Errore nel caricamento: ${e.message}`; });
