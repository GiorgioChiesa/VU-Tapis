// Event view of the error viewer (written by tools/export_event_viewer.py).
// Tool boxes over each 4x4 clip grid (same look and keys as tools_overlay.js: B shows/hides them,
// L adds the boxes under 0.3), outcome filter, and montage chapters (seek, current clip).
(function () {
  "use strict";
  const CLASSES = ["grasping_retractor", "cadiere_forceps", "bipolar_forceps", "force_bipolar", "clip_applier",
                   "stapler", "permanent_cautery_hook_spatula", "monopolar_curved_scissor", "vessel_sealer",
                   "tip_up_fenestrated_grasper", "bipolar_dissector", "needle_driver", "prograsp_forceps",
                   "suction_irrigator"];
  const PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
                   "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896"];
  const color = (tool) => PALETTE[CLASSES.indexOf(tool)] || "#ffff00";
  const NS = "http://www.w3.org/2000/svg";
  const store = {
    get(key, dflt) {
      try { const v = localStorage.getItem(key); return v === null ? dflt : v === "1"; } catch (e) { return dflt; }
    },
    set(key, v) { try { localStorage.setItem(key, v ? "1" : "0"); } catch (e) { /* private mode */ } },
  };

  // ---- bbox toggle ---------------------------------------------------------------------------
  let show = store.get("toolOverlay", true);
  let low = store.get("toolOverlayLow", false);
  const bar = document.createElement("div");
  bar.className = "to-bar";
  bar.innerHTML = '<button type="button"></button> <label><input type="checkbox"> anche &lt; 0.3 (L)</label>';
  document.body.appendChild(bar);
  const btn = bar.querySelector("button");
  const lowBox = bar.querySelector("input");
  function apply() {
    document.body.classList.toggle("to-off", !show);
    document.body.classList.toggle("to-low", low);
    btn.textContent = "Bbox strumenti: " + (show ? "ON" : "OFF") + " (B)";
    lowBox.checked = low;
  }
  btn.addEventListener("click", () => { show = !show; store.set("toolOverlay", show); apply(); });
  lowBox.addEventListener("change", () => { low = lowBox.checked; store.set("toolOverlayLow", low); apply(); });
  document.addEventListener("keydown", (e) => {
    if ((e.target.matches && e.target.matches("input, textarea")) || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === "b" || e.key === "B") btn.click();
    if (e.key === "l" || e.key === "L") { lowBox.checked = !lowBox.checked; lowBox.dispatchEvent(new Event("change")); }
  });
  apply();

  const clips = document.querySelectorAll(".clip");
  if (!clips.length) return;  // index page: only the toggle

  function el(tag, attrs) {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    return node;
  }

  // One 224x224 tile (nested svg clips labels to the tile) at (ox, oy) of the parent svg.
  function drawTile(svg, boxes, ox, oy, font) {
    const tile = el("svg", { x: ox, y: oy, width: 224, height: 224, viewBox: "0 0 224 224" });
    for (const b of [...boxes].sort((a, c) => a.conf - c.conf)) {
      const [x1, y1, x2, y2] = b.xyxy;
      const c = color(b.tool);
      const g = el("g", { class: b.above ? "to-hi" : "to-lo" });
      const rect = { x: x1, y: y1, width: x2 - x1, height: y2 - y1, fill: "none", stroke: c,
                     "stroke-width": Math.max(1, font / 5) };
      if (!b.above) rect["stroke-dasharray"] = `${font / 2} ${font / 3}`;
      g.appendChild(el("rect", rect));
      const label = `${b.tool} ${b.conf.toFixed(2)}`;
      const ty = Math.max(y1, font + 2);
      g.appendChild(el("rect", { x: x1, y: ty - font - 2, width: label.length * font * 0.56 + 4, height: font + 3,
                                 fill: c, opacity: b.above ? 0.9 : 0.6 }));
      const text = el("text", { x: x1 + 2, y: ty - 2, fill: "#fff", "font-size": font, "font-family": "sans-serif" });
      text.textContent = label;
      g.appendChild(text);
      tile.appendChild(g);
    }
    svg.appendChild(tile);
  }

  function loadClip(clip) {
    fetch(clip.dataset.dir + "tools.json").then((r) => r.json()).then((data) => {
      const imgs = clip.querySelectorAll(".grid img");
      if (imgs[data.center_k]) imgs[data.center_k].classList.add("center");
      const svg = el("svg", { class: "to-svg", viewBox: "0 0 896 896", preserveAspectRatio: "none" });
      data.frames.forEach((f, k) => drawTile(svg, f.boxes, (k % 4) * 224, Math.floor(k / 4) * 224, 12));
      clip.querySelector(".grid-wrap").appendChild(svg);
      const center = data.frames[data.center_k].boxes.filter((b) => b.above)
        .map((b) => `${b.tool} ${b.conf.toFixed(2)}`).join(", ") || "nessuno";
      const inClip = Object.entries(data.in_clip).sort((a, b) => b[1] - a[1])
        .map(([t, n]) => `${t} ${n}/${data.frames.length}`).join(", ") || "nessuno";
      clip.querySelector(".to-txt").textContent =
        `strumenti al secondo centrale: ${center} · frame del clip con lo strumento: ${inClip}`;
    }).catch(() => { /* clip without tools.json */ });
  }
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) if (e.isIntersecting) { io.unobserve(e.target); loadClip(e.target); }
    }, { rootMargin: "800px" });
    clips.forEach((c) => io.observe(c));
  } else {
    clips.forEach(loadClip);
  }

  // ---- outcome filter ------------------------------------------------------------------------
  const boxes = document.querySelectorAll(".flt input");
  function filter() {
    const on = new Set([...boxes].filter((b) => b.checked).map((b) => b.value));
    clips.forEach((c) => c.classList.toggle("hidden", !on.has(c.dataset.kind)));
  }
  boxes.forEach((b) => b.addEventListener("change", filter));

  // ---- montage: speed, current clip, seek from a grid -----------------------------------------
  const video = document.getElementById("montage");
  const now = document.getElementById("now");
  const chapters = window.CHAPTERS || [];
  document.querySelectorAll(".pctl button").forEach((b) => b.addEventListener("click", () => {
    video.playbackRate = +b.dataset.rate;
    document.querySelectorAll(".pctl button").forEach((x) => x.classList.toggle("on", x === b));
  }));
  let current = -1;
  video.addEventListener("timeupdate", () => {
    let i = chapters.length - 1;
    while (i > 0 && chapters[i].start > video.currentTime + 1e-3) i--;
    if (i === current || i < 0) return;
    if (current >= 0) document.getElementById("c" + current).classList.remove("playing");
    current = i;
    document.getElementById("c" + i).classList.add("playing");
    now.textContent = `clip ${i + 1}/${chapters.length}: ${chapters[i].clip} (${chapters[i].kind.replace("_", " ")})`;
  });
  document.querySelectorAll(".seek").forEach((a) => a.addEventListener("click", (e) => {
    e.preventDefault();
    video.currentTime = +a.dataset.start + 0.01;
    video.play();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }));
})();
