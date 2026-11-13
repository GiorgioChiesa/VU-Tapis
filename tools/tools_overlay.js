// Tool boxes of the terzetto detector (SurgVU26 cat1) over the error-viewer clips.
// Written by tools/add_tool_overlay.py: each item folder has a tools.json with the boxes of its
// 16 frames in 224x224 crop coordinates. Button (or key B) shows/hides them on every page;
// the checkbox (or key L) also draws the boxes under the 0.3 threshold, dashed.
(function () {
  "use strict";
  // Same colours as the service overlays (tab20-like, one per canonical class).
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
  let show = store.get("toolOverlay", true);
  let low = store.get("toolOverlayLow", false);

  const style = document.createElement("style");
  style.textContent = `
    .to-bar{position:fixed;top:8px;right:8px;z-index:10;background:#fff;border:1px solid #bbb;
            border-radius:6px;padding:6px 10px;font:13px sans-serif;box-shadow:0 1px 4px #0003}
    .to-bar button{font:inherit;cursor:pointer}
    .to-wrap{position:relative;display:inline-block;line-height:0}
    .to-svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
    body.to-off .to-svg, body.to-off .to-txt{display:none}
    .to-lo{display:none} body.to-low .to-lo{display:inline}
    .to-txt{font-size:11px;color:#333;margin:2px 0}`;
  document.head.appendChild(style);

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
    if (e.target.matches && e.target.matches("input, textarea") || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === "b" || e.key === "B") btn.click();
    if (e.key === "l" || e.key === "L") { lowBox.checked = !lowBox.checked; lowBox.dispatchEvent(new Event("change")); }
  });
  apply();

  function el(tag, attrs) {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    return node;
  }

  // One 224x224 tile (nested svg clips labels to the tile) at (ox, oy) of the parent svg.
  function drawTile(svg, boxes, ox, oy, font) {
    const tile = el("svg", { x: ox, y: oy, width: 224, height: 224, viewBox: "0 0 224 224" });
    // low-confidence first, so the boxes above threshold stay on top
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

  function overlay(img, viewBox) {
    const wrap = document.createElement("span");
    wrap.className = "to-wrap";
    img.parentNode.insertBefore(wrap, img);
    wrap.appendChild(img);
    const svg = el("svg", { class: "to-svg", viewBox: viewBox, preserveAspectRatio: "none" });
    wrap.appendChild(svg);
    return svg;
  }

  const summary = (tools) => tools.map((b) => `${b.tool} ${b.conf.toFixed(2)}`).join(", ") || "nessuno strumento";

  // Item page: the 16 frames, each a 224 tile shown at 2x.
  const frames = document.querySelectorAll(".fr[data-k]");
  if (frames.length) {
    fetch("tools.json").then((r) => r.json()).then((data) => {
      for (const fr of frames) {
        const img = fr.querySelector("img");
        const svg = el("svg", { class: "to-svg", viewBox: "0 0 224 224", preserveAspectRatio: "none" });
        img.parentNode.appendChild(svg);
        drawTile(svg, data.frames[+fr.dataset.k].boxes, 0, 0, 7);
      }
    });
    return;
  }

  // Pair page: one 4x4 strip of 224 tiles per card, loaded when it scrolls into view.
  function loadCard(card) {
    const link = card.querySelector("a");
    const img = link && link.querySelector("img");
    if (!img) return;
    fetch(link.getAttribute("href") + "tools.json").then((r) => r.json()).then((data) => {
      const side = Math.ceil(Math.sqrt(data.frames.length));
      const svg = overlay(img, `0 0 ${side * 224} ${side * 224}`);
      data.frames.forEach((f, k) => drawTile(svg, f.boxes, (k % side) * 224, Math.floor(k / side) * 224, 16));
      const center = data.frames[data.center_k].boxes.filter((b) => b.above);
      const inClip = Object.entries(data.in_clip).sort((a, b) => b[1] - a[1])
        .map(([t, n]) => `${t} ${n}/${data.frames.length}`).join(", ") || "nessuno";
      const txt = document.createElement("div");
      txt.className = "to-txt";
      txt.textContent = `strumenti al secondo centrale: ${summary(center)} · frame del clip con lo strumento: ${inClip}`;
      link.insertAdjacentElement("afterend", txt);
    }).catch(() => { /* item without tools.json */ });
  }
  const cards = document.querySelectorAll(".c");
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) if (e.isIntersecting) { io.unobserve(e.target); loadCard(e.target); }
    }, { rootMargin: "600px" });
    cards.forEach((c) => io.observe(c));
  } else {
    cards.forEach(loadCard);
  }
})();
