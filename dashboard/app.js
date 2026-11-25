/* AMS 560 results dashboard. Reads data/<attempt>.js built by build_data.py.
   Data is loaded with a <script> tag, so index.html also works from file://. */
"use strict";

const ATTEMPT = new URLSearchParams(location.search).get("attempt") || "attempt_12";
const CTRL = {
  naive_slo:     { label: "naive_slo",     color: "#e07b24" },
  robust:        { label: "robust",        color: "#0e9f80" },
  robust_v1:     { label: "robust_v1",     color: "#7c6fd6" },
  robust_no_age: { label: "robust_no_age", color: "#d1477a" },
  static_high:   { label: "static_high",   color: "#7b8494" },
};
const PRIMARY = ["naive_slo", "robust"];
const ABLATION = ["naive_slo", "robust_v1", "robust_no_age", "robust"];
const WORKLOADS = ["sharp_burst", "ramp", "steady"];
const METRICS = {
  j_per_token:     { label: "Energy per token (J)", fmt: v => v.toFixed(3), lowerBetter: true },
  burst_miss_rate: { label: "Burst-phase miss rate (%)", fmt: v => v.toFixed(1), pct: true, lowerBetter: true },
  miss_rate:       { label: "Whole-run miss rate (%)", fmt: v => v.toFixed(1), pct: true, lowerBetter: true },
  p95:             { label: "P95 latency (s)", fmt: v => v.toFixed(2), lowerBetter: true },
  mean_power_w:    { label: "Mean GPU power (W)", fmt: v => v.toFixed(1), lowerBetter: true },
  top_clock_frac:  { label: "Time at 2122 MHz (%)", fmt: v => v.toFixed(0), pct: true, lowerBetter: true },
  n_clock_changes: { label: "Clock changes per run", fmt: v => v.toFixed(0), lowerBetter: true },
};

let D = null;
const $ = s => document.querySelector(s);
const dark = () => matchMedia("(prefers-color-scheme: dark)").matches;
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

function baseLayout(extra = {}) {
  const ink = css("--ink-2"), grid = css("--line");
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Segoe UI, system-ui, sans-serif", size: 12.5, color: ink },
    margin: { l: 56, r: 16, t: 10, b: 46 },
    xaxis: { gridcolor: grid, zerolinecolor: grid, linecolor: grid },
    yaxis: { gridcolor: grid, zerolinecolor: grid, linecolor: grid },
    legend: { orientation: "h", y: -0.2, x: 0 },
    hoverlabel: { font: { family: "Segoe UI, sans-serif" } },
  }, extra);
}
const CFG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] };
const plot = (id, data, layout) => Plotly.react(id, data, baseLayout(layout), CFG);

const runsOf = (ctrl, wl) => D.runs.filter(r => r.controller === ctrl && r.workload === wl)
  .sort((a, b) => a.delay_ms - b.delay_ms);
const run = (ctrl, wl, d) => D.runs.find(r => r.controller === ctrl && r.workload === wl && r.delay_ms === d);
const delays = () => [...new Set(D.runs.map(r => r.delay_ms))].sort((a, b) => a - b);
const val = (r, m) => r == null || r[m] == null ? null : (METRICS[m] && METRICS[m].pct ? r[m] * 100 : r[m]);
const pctDiff = (a, b) => (b / a - 1) * 100;
const sgn = (v, d = 1) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(d);

function seg(el, options, value, onChange, labels = {}) {
  el.innerHTML = "";
  options.forEach(o => {
    const b = document.createElement("button");
    b.textContent = labels[o] ?? o;
    if (o === value) b.classList.add("on");
    b.onclick = () => { [...el.children].forEach(c => c.classList.remove("on")); b.classList.add("on"); onChange(o); };
    el.appendChild(b);
  });
}

function table(el, cols, rows, { sortable = true, onRow = null } = {}) {
  let sortKey = null, asc = true;
  const draw = () => {
    const rs = sortKey == null ? rows : [...rows].sort((a, b) => {
      const x = a[sortKey], y = b[sortKey];
      return (x > y ? 1 : x < y ? -1 : 0) * (asc ? 1 : -1);
    });
    el.innerHTML = "<thead><tr>" + cols.map(c => `<th class="${c.l ? "l" : ""}" data-k="${c.k}">${c.h}</th>`).join("") +
      "</tr></thead><tbody>" + rs.map((r, i) => `<tr data-i="${rows.indexOf(r)}">` + cols.map(c => {
        const v = r[c.k];
        const cls = (c.l ? "l " : "") + (c.cls ? c.cls(v, r) : "");
        return `<td class="${cls}">${c.f ? c.f(v, r) : v ?? "–"}</td>`;
      }).join("") + "</tr>").join("") + "</tbody>";
    if (sortable) el.querySelectorAll("th").forEach(th => th.onclick = () => {
      const k = th.dataset.k; asc = sortKey === k ? !asc : true; sortKey = k; draw();
    });
    if (onRow) el.querySelectorAll("tbody tr").forEach(tr => tr.onclick = () => onRow(rows[+tr.dataset.i]));
  };
  draw();
}

/* ───────────── Introduction ───────────── */
function renderIntro() {
  const DL = window.DELIVERABLES, badge = s => s === "done"
    ? '<span class="badge b-yes">Done</span>' : '<span class="badge b-trend">Partial</span>';
  const tbl = rows => rows.map(([id, name, st, note]) =>
    `<tr><td class="id">${id}</td><td><b>${name}</b><div class="note">${note}</div></td><td>${badge(st)}</td></tr>`).join("");
  $("#deliv-prop").innerHTML = tbl(DL.proposal);
  $("#deliv-readme").innerHTML = tbl(DL.readme);
  const n = rows => rows.filter(r => r[2] === "done").length;
  $("#deliv-sum").innerHTML =
    `<div><span class="big">${n(DL.proposal)} / ${DL.proposal.length}</span> proposal deliverables done, the rest partial</div>` +
    `<div><span class="big">${n(DL.readme)} / ${DL.readme.length}</span> README checklist items done, the rest partial</div>` +
    `<div class="muted">Remaining: a 3-seed GPU repeat for significance, and a burst-size axis for the failure map.</div>`;
}

/* ───────────── Overview ───────────── */
function renderOverview() {
  const ds = delays(), dmax = ds[ds.length - 1];
  const nB = run("naive_slo", "sharp_burst", dmax), rB = run("robust", "sharp_burst", dmax);
  const sb = ds.map(d => pctDiff(run("naive_slo", "sharp_burst", d).j_per_token, run("robust", "sharp_burst", d).j_per_token));
  const wins = sb.filter(v => v < 0).length;
  const st = run("static_high", "sharp_burst", 0);
  const rob = runsOf("robust", "sharp_burst");
  const save = rob.map(r => -pctDiff(st.j_per_token, r.j_per_token));
  $("#kpis").innerHTML = [
    ["Burst-phase misses at " + dmax + " ms", `${(nB.burst_miss_rate * 100).toFixed(1)}% → <span class="good">${(rB.burst_miss_rate * 100).toFixed(1)}%</span>`,
      `naive → robust (${nB.burst_misses} → ${rB.burst_misses} of ${rB.burst_n} burst requests)`],
    ["Energy on sharp_burst", `<span class="good">${sgn(sb.reduce((a, b) => a + b) / sb.length)}%</span>`,
      `robust vs naive, mean of ${ds.length} delays (robust lower in ${wins}/${ds.length})`],
    ["Saved vs static_high", `${Math.min(...save).toFixed(0)}–${Math.max(...save).toFixed(0)}<small>%</small>`,
      "robust on sharp_burst, same session"],
    ["GPU runs", `${D.meta.n_runs}<small> × ${D.meta.seeds.length} seed</small>`,
      `SLO ${D.meta.slo_target_s} s, budget ${(D.meta.budget_ratio * 100).toFixed(0)}%, all runs complete`],
  ].map(([l, v, s]) => `<div class="kpi"><div class="lab">${l}</div><div class="val">${v}</div><div class="sub">${s}</div></div>`).join("");

  const traces = (m, mult = 1) => PRIMARY.concat(["robust_v1", "robust_no_age"]).map(c => {
    const rs = runsOf(c, "sharp_burst");
    const main = PRIMARY.includes(c);
    return { x: rs.map(r => r.delay_ms), y: rs.map(r => r[m] * mult), name: c, mode: "lines+markers",
      line: { color: CTRL[c].color, width: main ? 3 : 1.5, dash: main ? "solid" : "dot" },
      marker: { size: main ? 9 : 6 }, visible: main ? true : "legendonly",
      hovertemplate: `${c}<br>%{x} ms: %{y:.3f}<extra></extra>` };
  });
  plot("ov-miss", traces("burst_miss_rate", 100), {
    xaxis: { title: "telemetry delay (ms)", tickvals: ds, gridcolor: css("--line") },
    yaxis: { title: "burst-phase miss (%)", gridcolor: css("--line"), rangemode: "tozero" },
    shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 5, y1: 5, line: { color: css("--bad"), dash: "dash", width: 1 } }],
    annotations: [{ xref: "paper", x: 0.01, y: 5, text: "5% budget", showarrow: false, yshift: 9, font: { color: css("--bad"), size: 11 } }],
  });
  plot("ov-energy", traces("j_per_token"), {
    xaxis: { title: "telemetry delay (ms)", tickvals: ds, gridcolor: css("--line") },
    yaxis: { title: "J / output token", gridcolor: css("--line") },
    shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: st.j_per_token, y1: st.j_per_token, line: { color: CTRL.static_high.color, dash: "dash", width: 1 } }],
    annotations: [{ xref: "paper", x: 0.01, y: st.j_per_token, text: "static_high", showarrow: false, yshift: -9, font: { size: 11 } }],
  });

  const C = [
    ["yes", "Concluded", "<b>Stale telemetry hurts a reactive controller.</b> naive's burst-phase misses rise 0 → 0 → 2.9 → 11.7 % as the delay goes 0 → 2500 ms. attempt_8–11 (8 seeds, ≈0.8 → 12 %) and the 20-seed simulator show the same."],
    ["yes", "Concluded", "<b>DVFS pays off on this testbed.</b> Every dynamic controller used 11–27 % less energy than static_high. There were no SLO violations at 0–500 ms. This held in every attempt from 8 to 12."],
    ["yes", "Concluded", "<b>Robust's lag-aware rule works as designed.</b> After the burst, naive keeps 2122 MHz to the end of the run because its 60 s P95 window still remembers the burst. Robust sees an empty queue and returns to 667 MHz (CAPACITY_DOWN). Its time at the top clock falls from ≈56 % to ≈26 %."],
    ["trend", "Strong trend", "<b>Robust cuts burst misses under stale telemetry.</b> At 2500 ms it had 3 misses against 16 for naive, out of 137 burst requests. Fisher's exact test gives p ≈ 0.003 at the request level. That is one burst per controller, so it is not yet a seed-level result."],
    ["trend", "Trend", "<b>Robust uses less energy on bursty traffic.</b> It was 5–6 % lower in 3 of 4 delays and +1.8 % at 500 ms. Each cell is n = 1, and run-to-run noise is about ±2 %."],
    ["trend", "Trend", "<b>Each robust mechanism contributes (ablation).</b> At 2500 ms the burst misses were 11.7 % for naive, 9.5 % for robust_v1, 13.1 % for robust_no_age (which also oscillated, with 16 clock changes) and 2.2 % for robust."],
    ["no", "Not shown", "<b>No difference on steady or ramp.</b> The gap is ±1.5 % on steady and −2.6 to +4.7 % on ramp, with no misses for either controller. This is within noise for one seed."],
    ["no", "Needs data", "<b>Seed-level significance.</b> It needs the 3-seed repeat (<code>python run_demo.py --repeat-3seed 1</code>), which gives 4 paired seeds per cell."],
  ];
  $("#conclusions").innerHTML = C.map(([k, b, t]) =>
    `<div class="cl"><span class="badge b-${k}">${b}</span><p>${t}</p></div>`).join("");

  const p = D.meta.probe || {}, cap = D.meta.capacity || {};
  $("#setup").innerHTML = [
    ["GPU", "RTX 5060 Laptop, clocks locked with nvidia-smi"],
    ["Model / batch", "Qwen2.5-0.5B-Instruct, long-context prompts, batch 16"],
    ["Clocks (MHz)", (cap.clocks_mhz || []).join(", ")],
    ["Top-clock capacity", `${p.capacity_rps} req/s (${p.batch_time_s} s per full batch)`],
    ["Base rate / burst", `${p.base_rate} req/s, ×${p.burst_multiplier} for 20 s (peak = 0.9 × top capacity)`],
    ["SLO / budget", `${D.meta.slo_target_s} s fixed (calibration suggested ${D.meta.calibrated_slo_s} s), ${(D.meta.budget_ratio * 100).toFixed(0)}% error budget`],
    ["Injected delays", delays().join(", ") + " ms on every controller input"],
    ["Run length", "10 s warm-up + 110 s measured (40 s steady, 20 s burst, 50 s recovery)"],
    ["Seeds", D.meta.seeds.join(", ")],
  ].map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
}

/* ───────────── Results ───────────── */
const RS = { wl: "sharp_burst", metric: "j_per_token", abl: false };
function renderResults() {
  seg($("#res-workload"), WORKLOADS, RS.wl, v => { RS.wl = v; drawResults(); });
  const sel = $("#res-metric");
  sel.innerHTML = Object.entries(METRICS).map(([k, m]) => `<option value="${k}">${m.label}</option>`).join("");
  sel.value = RS.metric; sel.onchange = () => { RS.metric = sel.value; drawResults(); };
  $("#res-ablation").onchange = e => { RS.abl = e.target.checked; drawResults(); };
  drawResults();

  const cols = [
    { k: "controller", h: "controller", l: true, f: v => `<span class="dot" style="background:${CTRL[v].color}"></span>${v}` },
    { k: "workload", h: "workload", l: true }, { k: "delay_ms", h: "delay ms" },
    { k: "j_per_token", h: "J/token", f: v => v.toFixed(3) },
    { k: "miss_rate", h: "miss %", f: v => (v * 100).toFixed(1) },
    { k: "burst_miss_rate", h: "burst miss %", f: v => v == null ? "–" : (v * 100).toFixed(1) },
    { k: "p95", h: "P95 s", f: v => v.toFixed(2) }, { k: "mean_power_w", h: "W", f: v => v.toFixed(1) },
    { k: "n_clock_changes", h: "clock chg" },
    { k: "top_clock_frac", h: "at 2122 %", f: v => (v * 100).toFixed(0) },
    { k: "n_requests", h: "requests" },
  ];
  table($("#res-all"), cols, [...D.runs].sort((a, b) => a.workload.localeCompare(b.workload) || a.delay_ms - b.delay_ms),
    { onRow: r => { EX.wl = r.workload; EX.d = r.delay_ms; EX.on = new Set([r.controller]); location.hash = "explorer"; } });
}
function drawResults() {
  const m = RS.metric, M = METRICS[m], ds = delays();
  const ctrls = RS.wl === "sharp_burst" && RS.abl ? ABLATION : PRIMARY;
  const data = ctrls.map(c => {
    const rs = runsOf(c, RS.wl);
    return { type: "bar", name: c, x: rs.map(r => r.delay_ms + " ms"), y: rs.map(r => val(r, m)),
      marker: { color: CTRL[c].color }, text: rs.map(r => val(r, m) == null ? "" : M.fmt(val(r, m))),
      textposition: "outside", cliponaxis: false, hovertemplate: `${c} %{x}: %{y:.3f}<extra></extra>` };
  });
  const st = run("static_high", RS.wl, 0);
  const shapes = [], ann = [];
  if (st && val(st, m) != null) {
    shapes.push({ type: "line", xref: "paper", x0: 0, x1: 1, y0: val(st, m), y1: val(st, m), line: { color: CTRL.static_high.color, dash: "dash" } });
    ann.push({ xref: "paper", x: 1, xanchor: "right", y: val(st, m), text: `static_high ${M.fmt(val(st, m))}`, showarrow: false, yshift: 9, font: { size: 11 } });
  }
  plot("res-bars", data, { barmode: "group", bargap: 0.25, shapes, annotations: ann,
    yaxis: { title: M.label, gridcolor: css("--line"), rangemode: "tozero" }, margin: { l: 60, r: 16, t: 24, b: 50 } });
  $("#res-note").textContent = m === "burst_miss_rate" && RS.wl !== "sharp_burst"
    ? "Burst-phase miss only exists for sharp_burst." : (M.lowerBetter ? "Lower is better." : "");

  const rows = ds.map(d => {
    const n = run("naive_slo", RS.wl, d), r = run("robust", RS.wl, d);
    return { d, jn: n.j_per_token, jr: r.j_per_token, dj: pctDiff(n.j_per_token, r.j_per_token),
      mn: n.miss_rate * 100, mr: r.miss_rate * 100, bn: n.burst_miss_rate, br: r.burst_miss_rate,
      pn: n.p95, pr: r.p95, tn: n.top_clock_frac * 100, tr: r.top_clock_frac * 100,
      sn: st ? -pctDiff(st.j_per_token, n.j_per_token) : null, sr: st ? -pctDiff(st.j_per_token, r.j_per_token) : null };
  });
  const f3 = v => v.toFixed(3), f1 = v => v == null ? "–" : v.toFixed(1), fb = v => v == null ? "–" : (v * 100).toFixed(1);
  table($("#res-paired"), [
    { k: "d", h: "delay ms" }, { k: "jn", h: "J/tok naive", f: f3 }, { k: "jr", h: "J/tok robust", f: f3 },
    { k: "dj", h: "Δ energy %", f: v => sgn(v), cls: v => v > 0.5 ? "pos" : v < -0.5 ? "neg" : "" },
    { k: "mn", h: "miss % n", f: f1 }, { k: "mr", h: "miss % r", f: f1 },
    { k: "bn", h: "burst miss % n", f: fb }, { k: "br", h: "burst miss % r", f: fb },
    { k: "pn", h: "P95 n", f: v => v.toFixed(2) }, { k: "pr", h: "P95 r", f: v => v.toFixed(2) },
    { k: "tn", h: "at 2122 % n", f: v => v.toFixed(0) }, { k: "tr", h: "at 2122 % r", f: v => v.toFixed(0) },
    { k: "sn", h: "saved vs static n", f: v => f1(v) + "%" }, { k: "sr", h: "saved vs static r", f: v => f1(v) + "%" },
  ], rows);
}

/* ───────────── Explorer ───────────── */
const EX = { wl: "sharp_burst", d: 2500, on: new Set(["naive_slo", "robust"]) };
function renderExplorer() {
  seg($("#ex-workload"), WORKLOADS, EX.wl, v => { EX.wl = v; renderExplorer(); });
  seg($("#ex-delay"), delays(), EX.d, v => { EX.d = v; drawExplorer(); }, Object.fromEntries(delays().map(d => [d, d + " ms"])));
  const avail = Object.keys(CTRL).filter(c => D.runs.some(r => r.controller === c && r.workload === EX.wl));
  $("#ex-ctrls").innerHTML = avail.map(c =>
    `<span class="chip ${EX.on.has(c) ? "" : "off"}" data-c="${c}"><i style="background:${CTRL[c].color}"></i>${c}</span>`).join("");
  $("#ex-ctrls").querySelectorAll(".chip").forEach(ch => ch.onclick = () => {
    const c = ch.dataset.c; EX.on.has(c) ? EX.on.delete(c) : EX.on.add(c); ch.classList.toggle("off"); drawExplorer();
  });
  drawExplorer();
}
function drawExplorer() {
  const ctrls = [...EX.on].filter(c => Object.keys(CTRL).includes(c));
  const runs = ctrls.map(c => run(c, EX.wl, c === "static_high" ? 0 : EX.d)).filter(Boolean);
  const grid = css("--line");
  const dom = [[0.82, 1], [0.62, 0.79], [0.42, 0.59], [0.22, 0.39], [0, 0.19]];
  const titles = ["arrival rate (req/s)", "queue length", "selected clock (MHz)", "GPU power (W)", "request latency (s)"];
  const L = { showlegend: true, margin: { l: 64, r: 16, t: 8, b: 40 }, hovermode: "x unified",
    legend: { orientation: "h", y: -0.05 } };
  dom.forEach((d, i) => {
    L["yaxis" + (i ? i + 1 : "")] = { domain: d, title: { text: titles[i], font: { size: 11.5 } }, gridcolor: grid, zerolinecolor: grid,
      ...(i === 2 ? { tickvals: (D.meta.capacity || {}).clocks_mhz } : {}) };
    L["xaxis" + (i ? i + 1 : "")] = { anchor: "y" + (i ? i + 1 : ""), matches: i ? "x" : undefined, gridcolor: grid,
      showticklabels: i === 4, title: i === 4 ? "seconds after warm-up" : undefined };
  });
  const data = [];
  runs.forEach((r, k) => {
    const t = D.traces[r.run_id], col = CTRL[r.controller].color, c = r.controller;
    const tc = t.control;
    if (k === 0) data.push({ x: tc.t_rel, y: tc.arrival_rate, name: "true arrival rate", line: { color: css("--ink-3"), width: 2 }, legendgroup: "true" });
    data.push({ x: tc.t_rel, y: tc.obs_arrival_rate, name: c + " · observed rate", line: { color: col, dash: "dot", width: 1.6 }, legendgroup: c, showlegend: false });
    data.push({ x: tc.t_rel, y: tc.queue_len, name: c + " · queue", line: { color: col, width: 1.8 }, xaxis: "x2", yaxis: "y2", legendgroup: c, showlegend: false });
    data.push({ x: tc.t_rel, y: tc.obs_queue_len, name: c + " · observed queue", line: { color: col, width: 1.2, dash: "dot" }, xaxis: "x2", yaxis: "y2", legendgroup: c, showlegend: false });
    data.push({ x: tc.t_rel, y: tc.selected_mhz, name: c, line: { color: col, width: 2.4, shape: "hv" }, xaxis: "x3", yaxis: "y3", legendgroup: c,
      customdata: tc.reason, hovertemplate: `${c}: %{y} MHz · %{customdata}<extra></extra>` });
    data.push({ x: t.gpu.t_rel, y: t.gpu.power_w, name: c + " · power", line: { color: col, width: 1.3 }, xaxis: "x4", yaxis: "y4", legendgroup: c, showlegend: false });
    const rq = t.requests;
    data.push({ x: rq.arr_rel, y: rq.latency_s, name: c + " · latency", mode: "markers", xaxis: "x5", yaxis: "y5", legendgroup: c, showlegend: false,
      marker: { color: t.miss.map(m => m ? css("--bad") : col), size: t.miss.map(m => m ? 7 : 4.5), opacity: 0.75,
        line: { width: t.miss.map(m => m ? 1 : 0), color: col } },
      hovertemplate: `${c}: %{y:.2f} s<extra></extra>` });
  });
  const shapes = [];
  const ph = runs.length ? D.traces[runs[0].run_id].phases : [];
  ph.forEach((p, i) => {
    if (p.phase === "burst" || p.phase === "recovery") {
      const end = i + 1 < ph.length ? ph[i + 1].start : 112;
      shapes.push({ type: "rect", xref: "x", yref: "paper", x0: p.start, x1: end, y0: 0, y1: 1, layer: "below", line: { width: 0 },
        fillcolor: p.phase === "burst" ? "rgba(224,123,36,0.10)" : "rgba(14,159,128,0.05)" });
    }
  });
  shapes.push({ type: "line", xref: "paper", yref: "y5", x0: 0, x1: 1, y0: D.meta.slo_target_s, y1: D.meta.slo_target_s,
    line: { color: css("--bad"), dash: "dash", width: 1 } });
  L.shapes = shapes;
  plot("ex-plot", data, L);

  table($("#ex-summary"), [
    { k: "controller", h: "controller", l: true, f: v => `<span class="dot" style="background:${CTRL[v].color}"></span>${v}` },
    { k: "j_per_token", h: "J/token", f: v => v.toFixed(3) }, { k: "miss_rate", h: "miss %", f: v => (v * 100).toFixed(1) },
    { k: "burst_miss_rate", h: "burst miss", f: (v, r) => v == null ? "–" : `${(v * 100).toFixed(1)}% (${r.burst_misses}/${r.burst_n})` },
    { k: "p95", h: "P95 s", f: v => v.toFixed(2) }, { k: "n_clock_changes", h: "clock chg" },
    { k: "top_clock_frac", h: "at 2122", f: v => (v * 100).toFixed(0) + "%" }, { k: "end_mhz", h: "end clock", f: v => v.toFixed(0) },
  ], runs, { sortable: false });

  const reasons = [...new Set(runs.flatMap(r => Object.keys(r.reasons)))].sort();
  plot("ex-reasons", runs.map(r => ({ type: "bar", orientation: "h", name: r.controller, y: reasons,
    x: reasons.map(k => r.reasons[k] || 0), marker: { color: CTRL[r.controller].color } })),
    { barmode: "group", margin: { l: 16, r: 16, t: 6, b: 36 }, yaxis: { automargin: true }, xaxis: { title: "decisions", gridcolor: css("--line") }, legend: { orientation: "h", y: -0.25 } });
}

/* ───────────── Ablation ───────────── */
function renderAblation() {
  const ds = delays(), dmax = ds[ds.length - 1];
  const desc = {
    naive_slo: ["Baseline", "Reacts to the observed P95 and queue. Does not know the telemetry age.", ["age-aware: no", "forecast: no", "lag-aware P95: no"]],
    robust_v1: ["Old robust (attempt_8–11)", "Risk score with the telemetry age and error-budget burn, and age-scaled down-step guard.", ["age-aware: yes", "forecast: no", "lag-aware P95: no"]],
    robust_no_age: ["robust without age", "Full v2 logic, but told its data is fresh (age = 0).", ["age-aware: no", "forecast: settle-only horizon", "lag-aware P95: yes"]],
    robust: ["Full robust v2", "Forecasts over the age horizon, adds backlog to demand, ignores a lagging P95 when there is no backlog.", ["age-aware: yes", "forecast: age + settle", "lag-aware P95: yes"]],
  };
  $("#ab-ladder").innerHTML = ABLATION.map(c => {
    const r = run(c, "sharp_burst", dmax), [t, p, f] = desc[c];
    return `<div class="rung" style="border-top-color:${CTRL[c].color}"><h4>${c}</h4><p><b>${t}.</b> ${p}</p>
      <div class="feat">${f.map(x => `<span>${x}</span>`).join("")}</div>
      <div style="margin-top:10px"><span class="big">${(r.burst_miss_rate * 100).toFixed(1)}%</span>
      <span class="muted"> burst miss at ${dmax} ms</span></div></div>`;
  }).join("");
  const line = (m, mult = 1) => ABLATION.map(c => {
    const rs = runsOf(c, "sharp_burst");
    return { x: rs.map(r => r.delay_ms), y: rs.map(r => r[m] * mult), name: c, mode: "lines+markers",
      line: { color: CTRL[c].color, width: c === "robust" ? 3 : 2 }, marker: { size: 8 } };
  });
  const xa = { title: "telemetry delay (ms)", tickvals: ds, gridcolor: css("--line") };
  plot("ab-miss", line("burst_miss_rate", 100), { xaxis: xa, yaxis: { title: "%", gridcolor: css("--line"), rangemode: "tozero" } });
  plot("ab-energy", line("j_per_token"), { xaxis: xa, yaxis: { title: "J / token", gridcolor: css("--line") } });
  plot("ab-chg", line("n_clock_changes"), { xaxis: xa, yaxis: { title: "changes", gridcolor: css("--line"), rangemode: "tozero" } });
  plot("ab-top", line("top_clock_frac", 100), { xaxis: xa, yaxis: { title: "%", gridcolor: css("--line"), rangemode: "tozero" } });
  $("#ab-text").innerHTML = `<h3>Reading the ablation</h3><ul>
    <li><b>robust_v1 ≈ naive.</b> The old robust controller made almost the same clock choices as naive in attempt_8–11. Both took every
      up-step from the shared capacity floor and both stayed pinned at 2122 MHz after the burst. This is why v2 was built.</li>
    <li><b>The lag-aware P95 rule drives the energy result.</b> Both v2 variants (robust, robust_no_age) spend about 25 % of the run at the top clock,
      against about 56 % for naive and robust_v1. They are also the two cheapest controllers in J/token.</li>
    <li><b>Telemetry age drives the reliability result.</b> At ${dmax} ms, robust_no_age forecasts over only the 0.7 s settle time, reacts late,
      then over-corrects (${run("robust_no_age", "sharp_burst", dmax).n_clock_changes} clock changes). Its misses are as bad as naive's.
      The full robust controller forecasts over age + settle and holds misses at ${(run("robust", "sharp_burst", dmax).burst_miss_rate * 100).toFixed(1)} %.</li>
    <li><b>Caveat.</b> This is one seed per cell. The 20-seed simulator gives the same order at 2500 ms (whole-run miss): naive 13.5 %, robust_no_age 12.2 %, robust 6.6 %.</li></ul>`;
}

/* ───────────── Testbed ───────────── */
function renderTestbed() {
  const sw = [...D.sweep].sort((a, b) => a.clock_mhz - b.clock_mhz);
  const x = sw.map(r => r.clock_mhz + " MHz"), g = css("--line");
  const bar = (y, col, fmt) => [{ type: "bar", x, y, marker: { color: col }, text: y.map(fmt), textposition: "outside", cliponaxis: false }];
  plot("tb-j", bar(sw.map(r => r.j_per_token), "#3b82c4", v => v.toFixed(3)), { yaxis: { title: "J / token", gridcolor: g }, margin: { l: 56, r: 16, t: 24, b: 40 } });
  plot("tb-p95", bar(sw.map(r => r.p95), "#e07b24", v => v.toFixed(1)), {
    yaxis: { title: "P95 (s, log)", type: "log", gridcolor: g }, margin: { l: 56, r: 16, t: 24, b: 40 },
    shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: D.meta.slo_target_s, y1: D.meta.slo_target_s, line: { color: css("--bad"), dash: "dash" } }],
    annotations: [{ xref: "paper", x: 1, xanchor: "right", y: Math.log10(D.meta.slo_target_s), text: "SLO 5 s", showarrow: false, yshift: 9, font: { color: css("--bad"), size: 11 } }] });
  const cap = D.meta.capacity || {};
  plot("tb-cap", [{ type: "bar", x: (cap.clocks_mhz || []).map(c => c + " MHz"), y: cap.capacity_rps, marker: { color: "#0e9f80" },
    text: (cap.capacity_rps || []).map(v => v.toFixed(2)), textposition: "outside", cliponaxis: false, name: "capacity",
    customdata: cap.batch_time_s, hovertemplate: "%{x}: %{y:.2f} req/s<br>batch of 16 in %{customdata:.2f} s<extra></extra>" }],
    { yaxis: { title: "requests / s", gridcolor: g }, margin: { l: 56, r: 16, t: 24, b: 40 },
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 7.2, y1: 7.2, line: { color: css("--warn"), dash: "dot" } }],
      annotations: [{ xref: "paper", x: 0.01, y: 7.2, text: "burst peak 7.2 req/s", showarrow: false, yshift: 9, font: { size: 11, color: css("--warn") } }] });
  const p = D.meta.probe || {};
  $("#tb-kv").innerHTML = [
    ["Probe clock", p.clock_mhz + " MHz"], ["Full batches measured", p.n_full_batches],
    ["Batch time (in stack / bench)", `${p.batch_time_s} s / ${(+p.bench_batch_time_s).toFixed(2)} s`],
    ["Capacity at top clock", p.capacity_rps + " req/s"], ["Base rate chosen", `${p.base_rate} req/s (burst peak = ${p.burst_peak_frac} × capacity)`],
    ["Capacity model", (cap.model || "")], ["SLO used", `${D.meta.slo_target_s} s (fixed across attempts)`],
    ["Calibrated SLO (not used)", `${D.meta.calibrated_slo_s} s = ceil(1.5 × worst P95 at top clock)`],
  ].map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
}

/* ───────────── Simulation ───────────── */
const SM = { nw: 1, wl: "sharp_burst" };
function renderSimulation() {
  if (!D.sim.length) { $("#page-simulation").insertAdjacentHTML("beforeend", "<p>No sim_summary.csv in this attempt.</p>"); return; }
  const nws = [...new Set(D.sim.map(r => r.n_workers))].sort((a, b) => a - b);
  seg($("#sim-workers"), nws, SM.nw, v => { SM.nw = v; drawSim(); }, Object.fromEntries(nws.map(n => [n, n + (n > 1 ? " workers" : " worker")])));
  seg($("#sim-workload"), WORKLOADS, SM.wl, v => { SM.wl = v; drawSim(); });
  drawSim();
}
function drawSim() {
  const rows = D.sim.filter(r => r.n_workers === SM.nw && r.workload === SM.wl);
  const ctrls = ABLATION.filter(c => rows.some(r => r.controller === c));
  const xa = { title: "telemetry delay (ms)", tickvals: delays(), gridcolor: css("--line") };
  const line = (m, mult = 1) => ctrls.map(c => {
    const rs = rows.filter(r => r.controller === c).sort((a, b) => a.delay_ms - b.delay_ms);
    return { x: rs.map(r => r.delay_ms), y: rs.map(r => r[m] * mult), name: c, mode: "lines+markers",
      line: { color: CTRL[c].color, width: c === "robust" || c === "naive_slo" ? 3 : 1.6 }, marker: { size: 7 } };
  });
  plot("sim-miss", line("miss_rate", 100), { xaxis: xa, yaxis: { title: "miss rate (%)", gridcolor: css("--line"), rangemode: "tozero" } });
  plot("sim-energy", line("j_per_token"), { xaxis: xa, yaxis: { title: "J / token", gridcolor: css("--line") } });
  const p = D.sim.filter(r => r.controller === "robust_vs_naive" && r.n_workers === SM.nw).sort((a, b) =>
    WORKLOADS.indexOf(a.workload) - WORKLOADS.indexOf(b.workload) || a.delay_ms - b.delay_ms);
  table($("#sim-paired"), [
    { k: "workload", h: "workload", l: true }, { k: "delay_ms", h: "delay ms", f: v => v.toFixed(0) },
    { k: "d_j_per_token_pct", h: "Δ energy %", f: v => sgn(v, 2), cls: v => v > 0.5 ? "pos" : v < -0.5 ? "neg" : "" },
    { k: "d_miss_pp", h: "Δ miss (pp)", f: v => sgn(v, 2), cls: v => v > 0.2 ? "pos" : v < -0.2 ? "neg" : "" },
    { k: "robust_fewer_misses", h: "seeds robust better", f: v => v.toFixed(0) },
    { k: "robust_more_misses", h: "seeds robust worse", f: v => v.toFixed(0) },
  ], p);
}

/* ───────────── Figures ───────────── */
function renderFigures() {
  const cap = {
    fig1: "Telemetry delay vs J/token", fig2: "Telemetry delay vs SLO miss rate", fig3: "Burst time series (robust, 0 ms)",
    fig4: "Cumulative energy", fig5: "Error-budget burn over time", fig7: "Failure map (naive_slo)",
    fig8: "Ablation", fig_budget: "Error budget vs P95 (non-redundancy)",
  };
  $("#gallery").innerHTML = D.meta.figures.map(f => {
    const n = f.split("/").pop(), key = Object.keys(cap).find(k => n.startsWith(k + "_")) || "";
    return `<figure data-src="${f}"><img src="${f}" alt="${n}"><figcaption>${cap[key] ? cap[key] + " · " : ""}${n}</figcaption></figure>`;
  }).join("");
  $("#gallery").querySelectorAll("figure").forEach(fg => fg.onclick = () => {
    $("#lightbox img").src = fg.dataset.src; $("#lightbox").classList.add("on");
  });
  $("#lightbox").onclick = () => $("#lightbox").classList.remove("on");
}

/* ───────────── Team & files (content in team.js) ───────────── */
function renderTeam() {
  const T = window.TEAM;
  $("#team").innerHTML = T.members.map(m => `<div class="member"><div class="top">
      <div class="avatar" style="background:${m.color}">${m.initials}</div>
      <div><div class="name">${m.name}</div><div class="role">Member ${m.n} · ${m.role}</div></div></div>
      <ul>${m.did.map(x => `<li>${x}</li>`).join("")}</ul>
      <div class="files">${m.files.map(f => `<code data-f="${f}">${f}</code>`).join("")}</div></div>`).join("");
  $("#team").querySelectorAll("code[data-f]").forEach(c => c.onclick = () => {
    $("#file-q").value = c.dataset.f; FL.layer = "all"; drawFiles(); seg($("#file-layer"), ["all", ...T.layers.map(l => l.id)], "all", setLayer, layerLabels());
    const el = document.querySelector(`details.file[data-f="${CSS.escape(c.dataset.f)}"]`);
    if (el) { el.open = true; el.scrollIntoView({ behavior: "smooth", block: "center" }); }
  });
  $("#flow").innerHTML = T.flow.map((n, i) => (i ? '<div class="arrow">→</div>' : "") +
    `<div class="node"><b>${n[0]}</b><span>${n[1]}</span></div>`).join("");
  const layerLabels = () => Object.fromEntries([["all", "All"], ...T.layers.map(l => [l.id, l.short])]);
  const setLayer = v => { FL.layer = v; drawFiles(); };
  seg($("#file-layer"), ["all", ...T.layers.map(l => l.id)], FL.layer, setLayer, layerLabels());
  $("#file-q").oninput = drawFiles;
  drawFiles();
  $("#talk").innerHTML = T.talk.map(([w, t]) => `<li><b>${w}</b> ${t}</li>`).join("");
  $("#gloss").innerHTML = T.glossary.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  $("#qa").innerHTML = T.qa.map(([q, a]) => `<details><summary>${q}</summary><p>${a}</p></details>`).join("");
}
const FL = { layer: "all" };
function drawFiles() {
  const T = window.TEAM, q = $("#file-q").value.trim().toLowerCase();
  const own = n => (T.members.find(m => m.n === n) || {}).name || "";
  $("#files").innerHTML = T.layers.filter(l => FL.layer === "all" || l.id === FL.layer).map(l => {
    const fs = T.files.filter(f => f.layer === l.id).filter(f => !q ||
      [f.path, f.one, f.what.join(" "), f.ideas.join(" "), f.present, own(f.owner), f.owner_note || ""].join(" ").toLowerCase().includes(q));
    if (!fs.length) return "";
    return `<div class="layer"><h4>${l.name}</h4>` + fs.map(f => `<details class="file ${q ? "hit" : ""}" data-f="${f.path}" ${q && fs.length < 4 ? "open" : ""}>
      <summary><span class="fn">${f.path}</span><span class="one">${f.one}</span>
      <span class="own">${f.owner ? "M" + f.owner + " · " + own(f.owner) : "shared"}</span></summary>
      <div class="body"><h5>What it does</h5><ul>${f.what.map(x => `<li>${x}</li>`).join("")}</ul>
      <h5>Key ideas to understand</h5><ul>${f.ideas.map(x => `<li>${x}</li>`).join("")}</ul>
      <h5>How to present it</h5><p>${f.present}</p>
      ${f.owner_note ? `<h5>History</h5><p>${f.owner_note}</p>` : ""}</div></details>`).join("") + "</div>";
  }).join("") || '<p class="muted">No file matches.</p>';
}

/* ───────────── routing ───────────── */
const RENDER = { intro: renderIntro, overview: renderOverview, results: renderResults, explorer: renderExplorer, ablation: renderAblation,
  testbed: renderTestbed, simulation: renderSimulation, figures: renderFigures, team: renderTeam };
const done = new Set();
function route() {
  const page = (location.hash || "#intro").slice(1);
  const id = RENDER[page] ? page : "intro";
  document.querySelectorAll(".page").forEach(p => p.classList.toggle("on", p.id === "page-" + id));
  document.querySelectorAll("#nav a").forEach(a => a.classList.toggle("on", a.dataset.page === id));
  if (!done.has(id) || id === "explorer") { RENDER[id](); done.add(id); }
  window.scrollTo(0, 0);
  setTimeout(() => document.querySelectorAll(`#page-${id} .js-plotly-plot`).forEach(p => Plotly.Plots.resize(p)), 30);
}

function loadData(attempt) {
  return new Promise((resolve, reject) => {
    const have = () => (window.DASHBOARD_DATA || {})[attempt];
    if (have()) return resolve(have());
    const s = document.createElement("script");
    s.src = `data/${attempt}.js`;
    s.onload = () => have() ? resolve(have()) : reject(new Error(`data/${attempt}.js has no data for ${attempt}`));
    s.onerror = () => reject(new Error(`data/${attempt}.js not found. Run: python dashboard/build_data.py --attempt results/${attempt}`));
    document.head.appendChild(s);
  });
}

loadData(ATTEMPT).then(d => {
  D = d;
  document.querySelectorAll(".attempt-name").forEach(e => e.textContent = d.attempt);
  $("#side-foot").innerHTML = `Data: results/${d.attempt}<br>${d.meta.n_runs} GPU runs · seed ${d.meta.seeds.join(", ")}`;
  addEventListener("hashchange", route);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { done.clear(); route(); });
  route();
}).catch(e => { $("#main").innerHTML = `<div class="card"><h3>Could not load data</h3><p>${e.message}</p></div>`; });
