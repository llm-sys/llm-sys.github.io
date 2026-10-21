// Interactive figures: grouped bars, lines, and the DoSA screening explainer.
// Plain SVG, no dependencies. Every chart also has a <table> in the page, so
// tooltips add detail but never hold the only copy of a value.

const NS = "http://www.w3.org/2000/svg";
const el = (name, attrs = {}, parent) => {
  const n = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
};
const text = (parent, x, y, str, attrs = {}) => {
  const t = el("text", { x, y, ...attrs }, parent);
  t.textContent = str;
  return t;
};
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
// Slot 1 is the recessive baseline gray, 2 a comparison blue, 3 the brand green,
// 4 a second comparison orange.
// Series pick a slot with "slot"; otherwise they take blue, green, gray in order.
const palette = () => [css("--viz-1"), css("--viz-2"), css("--viz-3"), css("--viz-4")];
const colorsFor = (spec) => {
  const p = palette();
  const order = [1, 2, 0];
  return spec.series.map((s, j) => p[s.slot ? s.slot - 1 : order[j % 3]]);
};

function niceTicks(min, max, count = 4) {
  const span = max - min;
  const raw = span / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) || raw;
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) ticks.push(+v.toFixed(6));
  return ticks;
}

const fmt = (v, unit = "") => {
  if (Number.isInteger(v)) return v + unit;
  const s = Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1).replace(/\.0$/, "") : v.toFixed(2).replace(/0$/, "");
  return s + unit;
};

// One tooltip element shared by all charts on the page.
let tip;
function tooltip() {
  if (tip) return tip;
  tip = document.createElement("div");
  tip.className = "viz-tip";
  tip.setAttribute("role", "status");
  document.body.appendChild(tip);
  return tip;
}
function showTip(rows, title, x, y) {
  const t = tooltip();
  t.replaceChildren();
  const h = document.createElement("p");
  h.className = "viz-tip__title";
  h.textContent = title;
  t.appendChild(h);
  for (const r of rows) {
    const row = document.createElement("p");
    row.className = "viz-tip__row";
    const key = document.createElement("i");
    key.style.background = r.color;
    const v = document.createElement("strong");
    v.textContent = r.value;
    const n = document.createElement("span");
    n.textContent = r.name;
    row.append(key, v, n);
    t.appendChild(row);
  }
  t.classList.add("is-on");
  const w = t.offsetWidth, hgt = t.offsetHeight;
  const left = Math.min(window.innerWidth - w - 12, Math.max(12, x + 14));
  const top = y - hgt - 14 < 8 ? y + 18 : y - hgt - 14;
  t.style.transform = `translate(${left}px, ${top + window.scrollY}px)`;
}
const hideTip = () => tip && tip.classList.remove("is-on");

function legend(host, spec, colors, shape) {
  if (spec.series.length < 2) return;
  const ul = document.createElement("ul");
  ul.className = "viz-legend";
  spec.series.forEach((s, i) => {
    const li = document.createElement("li");
    const k = document.createElement("i");
    k.className = `viz-key viz-key--${shape}`;
    k.style.setProperty("--c", colors[i]);
    li.append(k, document.createTextNode(s.name));
    ul.appendChild(li);
  });
  host.appendChild(ul);
}

// -- grouped columns -----------------------------------------------------------

function bars(host, spec) {
  const colors = colorsFor(spec);
  const p = palette();
  // One series with an emphasized category: that bar in green, the rest gray.
  const barColor = (j, cat) => spec.emphasis ? (cat === spec.emphasis ? p[2] : p[0]) : colors[j];
  const W = Math.max(320, host.clientWidth);
  const H = Math.round(Math.min(360, Math.max(240, W * 0.42)));
  const m = { t: (spec.groups ? 34 : 14) + (spec.y ? 22 : 0), r: 8, b: 30, l: 48 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const all = spec.series.flatMap((s) => s.values);
  const yMin = spec.yMin ?? 0, yMax = spec.yMax ?? Math.max(...all) * 1.08;
  const y = (v) => m.t + ph - ((v - yMin) / (yMax - yMin)) * ph;

  legend(host, spec, colors, "rect");
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "viz", width: W, height: H }, host);

  for (const t of niceTicks(yMin, yMax)) {
    el("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), class: t === yMin ? "viz-base" : "viz-grid" }, svg);
    text(svg, m.l - 8, y(t) + 4, fmt(t, spec.unit), { class: "viz-tick", "text-anchor": "end" });
  }
  if (spec.y) text(svg, 0, 12, spec.y, { class: "viz-axis-title" });

  const n = spec.categories.length, k = spec.series.length;
  const groupGap = spec.groups ? 18 : 0;
  const breaks = (spec.groups || []).slice(1).map((g) => g.from);
  const slots = pw - groupGap * breaks.length;
  const band = slots / n;
  const gap = 2;
  const bw = Math.min(22, (band * 0.72 - gap * (k - 1)) / k);
  const xOf = (i) => m.l + i * band + groupGap * breaks.filter((b) => b <= i).length;

  (spec.groups || []).forEach((g) => {
    const x0 = xOf(g.from), x1 = xOf(g.to) + band;
    const gy = m.t - 20;
    text(svg, (x0 + x1) / 2, gy, g.label, { class: "viz-group", "text-anchor": "middle" });
    el("line", { x1: x0 + 4, x2: x1 - 4, y1: gy + 8, y2: gy + 8, class: "viz-grid" }, svg);
  });

  spec.categories.forEach((cat, i) => {
    const cx = xOf(i) + band / 2;
    const total = k * bw + (k - 1) * gap;
    const g = el("g", { class: "viz-hit", tabindex: 0, role: "button", "aria-label": `${cat}: ` + spec.series.map((s) => `${s.name} ${fmt(s.values[i], spec.unit)}`).join(", ") }, svg);
    el("rect", { x: xOf(i), y: m.t, width: band, height: ph, class: "viz-band" }, g);
    spec.series.forEach((s, j) => {
      const x = cx - total / 2 + j * (bw + gap);
      const top = y(s.values[i]), base = y(yMin);
      const r = Math.min(4, bw / 2, base - top);
      el("path", {
        d: `M${x},${base} V${top + r} Q${x},${top} ${x + r},${top} H${x + bw - r} Q${x + bw},${top} ${x + bw},${top + r} V${base} Z`,
        fill: barColor(j, cat), class: "viz-bar", style: `--d:${i * 40 + j * 70}ms`,
      }, g);
    });
    text(svg, cx, H - m.b + 18, cat, { class: "viz-tick", "text-anchor": "middle" });
    const show = (ev) => {
      const r = ev.currentTarget.getBoundingClientRect();
      const px = ev.clientX ?? r.left + r.width / 2, py = ev.clientY ?? r.top;
      showTip(spec.series.map((s, j) => ({ name: s.name, value: fmt(s.values[i], spec.unit), color: barColor(j, cat) })), cat, px, py);
    };
    g.addEventListener("pointermove", show);
    g.addEventListener("focus", show);
    g.addEventListener("pointerleave", hideTip);
    g.addEventListener("blur", hideTip);
  });

  // Selective value labels at bar tips: "all", "last" category, or "max" per series.
  if (spec.label) {
    spec.series.forEach((s, j) => {
      const idxs = spec.label === "all" ? s.values.map((_, i) => i)
        : spec.label === "last" ? [s.values.length - 1]
        : [s.values.indexOf(Math.max(...s.values))];
      idxs.forEach((i) => {
        const cx = xOf(i) + band / 2 - (k * bw + (k - 1) * gap) / 2 + j * (bw + gap) + bw / 2;
        text(svg, cx, y(s.values[i]) - 7, fmt(s.values[i], spec.unit), { class: "viz-label", "text-anchor": "middle" });
      });
    });
  }
}

// -- lines -----------------------------------------------------------------------

function lines(host, spec) {
  const colors = colorsFor(spec);
  const W = Math.max(320, host.clientWidth);
  const H = Math.round(Math.min(320, Math.max(220, W * 0.38)));
  const m = { t: 14 + (spec.y ? 22 : 0), r: 60, b: 30, l: 48 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const xs = spec.categories.map(Number);
  const x = (v) => m.l + ((v - xs[0]) / (xs[xs.length - 1] - xs[0])) * pw;
  const all = spec.series.flatMap((s) => s.values);
  const yMin = spec.yMin ?? Math.min(...all), yMax = spec.yMax ?? Math.max(...all);
  const y = (v) => m.t + ph - ((v - yMin) / (yMax - yMin)) * ph;

  legend(host, spec, colors, "line");
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "viz", width: W, height: H }, host);
  for (const t of niceTicks(yMin, yMax)) {
    el("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), class: t === yMin ? "viz-base" : "viz-grid" }, svg);
    text(svg, m.l - 8, y(t) + 4, fmt(t, spec.unit), { class: "viz-tick", "text-anchor": "end" });
  }
  (spec.xTicks || [xs[0], xs[xs.length - 1]]).forEach((t) => text(svg, x(t), H - m.b + 18, String(t), { class: "viz-tick", "text-anchor": "middle" }));
  if (spec.y) text(svg, 0, 12, spec.y, { class: "viz-axis-title" });

  spec.series.forEach((s, j) => {
    const d = s.values.map((v, i) => `${i ? "L" : "M"}${x(xs[i]).toFixed(1)},${y(v).toFixed(1)}`).join("");
    el("path", { d, fill: "none", stroke: colors[j], class: "viz-line", pathLength: 1 }, svg);
    const last = s.values.length - 1;
    text(svg, x(xs[last]) + 8, y(s.values[last]) + 4 + (j - (spec.series.length - 1) / 2) * 13, s.name, { class: "viz-end" });
  });

  const cross = el("line", { y1: m.t, y2: m.t + ph, class: "viz-cross" }, svg);
  const dots = spec.series.map((s, j) => el("circle", { r: 4, fill: colors[j], class: "viz-dot" }, svg));
  const hit = el("rect", { x: m.l, y: m.t, width: pw, height: ph, class: "viz-overlay", tabindex: 0 }, svg);
  let idx = -1;
  const at = (i, px, py) => {
    idx = i;
    const cx = x(xs[i]);
    cross.setAttribute("x1", cx); cross.setAttribute("x2", cx);
    svg.classList.add("is-hover");
    dots.forEach((dt, j) => { dt.setAttribute("cx", cx); dt.setAttribute("cy", y(spec.series[j].values[i])); });
    showTip(spec.series.map((s, j) => ({ name: s.name, value: fmt(s.values[i], spec.unit), color: colors[j] })), `${spec.xLabel || "x"} ${xs[i]}`, px, py);
  };
  hit.addEventListener("pointermove", (ev) => {
    const r = svg.getBoundingClientRect();
    const vx = ((ev.clientX - r.left) / r.width) * W;
    let best = 0;
    xs.forEach((v, i) => { if (Math.abs(x(v) - vx) < Math.abs(x(xs[best]) - vx)) best = i; });
    at(best, ev.clientX, ev.clientY);
  });
  hit.addEventListener("keydown", (ev) => {
    if (ev.key !== "ArrowRight" && ev.key !== "ArrowLeft") return;
    ev.preventDefault();
    const i = Math.max(0, Math.min(xs.length - 1, (idx < 0 ? 0 : idx) + (ev.key === "ArrowRight" ? 1 : -1)));
    const r = svg.getBoundingClientRect();
    at(i, r.left + (x(xs[i]) / W) * r.width, r.top + m.t);
  });
  const off = () => { svg.classList.remove("is-hover"); hideTip(); };
  hit.addEventListener("pointerleave", off);
  hit.addEventListener("blur", off);
}

// -- screening explainer -----------------------------------------------------------

function screening(root) {
  const svg = root.querySelector("svg");
  const input = root.querySelector("[data-tau]");
  const keptEl = root.querySelector("[data-kept]");
  const totalEl = root.querySelector("[data-total]");
  // Deterministic keys: most score low, a few are real candidates.
  let seed = 11;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const N = 28;
  const keys = Array.from({ length: N }, (_, i) => {
    const hot = rnd() < 0.16;
    const s = hot ? 55 + rnd() * 40 : 4 + rnd() * 42;
    const slack = 4 + rnd() * 16;
    return { i, s, u: Math.min(100, s + slack) };
  });
  totalEl.textContent = N;

  const draw = () => {
    const W = Math.max(300, svg.parentElement.clientWidth);
    const rowH = W < 520 ? 7 : 9, gap = 3;
    const H = N * (rowH + gap) + 8;
    const m = { l: 4, r: 4 };
    const pw = W - m.l - m.r;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("height", H);
    svg.replaceChildren();
    const tau = +input.value;
    const xv = (v) => m.l + (v / 100) * pw;
    let kept = 0;
    keys.forEach((k, r) => {
      const y = 4 + r * (rowH + gap);
      const live = k.u >= tau;
      if (live) kept++;
      el("rect", { x: m.l, y, width: pw, height: rowH, rx: 2, class: "sc-track" }, svg);
      el("rect", { x: m.l, y, width: Math.max(rowH, xv(k.u) - m.l), height: rowH, rx: 2, class: live ? "sc-bound is-live" : "sc-bound" }, svg);
      if (live) el("rect", { x: xv(k.s) - 1.5, y: y - 1, width: 3, height: rowH + 2, rx: 1.5, class: "sc-score" }, svg);
    });
    el("line", { x1: xv(tau), x2: xv(tau), y1: 0, y2: H, class: "sc-tau" }, svg);
    keptEl.textContent = kept;
  };
  input.addEventListener("input", draw);
  new ResizeObserver(draw).observe(svg.parentElement);
  draw();
}

// -- boot -----------------------------------------------------------------------

const build = (host) => {
  const spec = JSON.parse(host.dataset.chart);
  host.replaceChildren();
  (spec.type === "lines" ? lines : bars)(host, spec);
};

document.querySelectorAll("[data-chart]").forEach((host) => {
  build(host);
  let w = host.clientWidth;
  new ResizeObserver(() => {
    if (Math.abs(host.clientWidth - w) > 8) { w = host.clientWidth; build(host); }
  }).observe(host);
});
document.querySelectorAll("[data-screening]").forEach(screening);
