/* AMS 560 results dashboard. Reads data/<attempt>.js built by build_data.py.
   Data is loaded with a <script> tag, so index.html also works from file://. */
"use strict";

/* Single-run pages show one showcase replicate; statistics come from the
   pooled 3-seed repeats (Replication page). */
const ATTEMPT = new URLSearchParams(location.search).get("attempt") || "attempt_13_s41";
const SHOWCASE_WHY = "It was chosen for presentation because it shows the typical pattern cleanly: naive's burst misses rise steadily with delay, " +
  "robust stays near zero, robust is cheaper at every delay, and the ablation order is clear. It is not the most extreme replicate " +
  "(seed 43 has a larger burst and bigger gaps).";
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
    `<div class="muted">Done since attempt_12: three 3-seed GPU repeats (attempt_13, 14, 15; 315 runs) give run-level significance.
      Remaining: a burst-size axis for the failure map.</div>`;
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
    ["Showcase replicate", `${D.meta.n_runs}<small> runs · seed ${D.meta.seeds.join(", ")}</small>`,
      `${D.meta.source_attempt || D.attempt}, one of ${window.REPLICATION ? new Set(window.REPLICATION.runs.map(r => r.att + r.seed)).size : "several"} replicates · SLO ${D.meta.slo_target_s} s, budget ${(D.meta.budget_ratio * 100).toFixed(0)}%`],
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

  const REPd = window.REPLICATION, poolAb = (c, d) => REPd && REPd.ablation.find(a => a.controller === c && a.delay_ms === d);
  const pp = (c, d) => { const a = poolAb(c, d); return a ? (a.bk / a.bn * 100).toFixed(1) : "–"; };
  const bmS = c => ds.map(d => (run(c, "sharp_burst", d).burst_miss_rate * 100).toFixed(1)).join(" → ");
  const svAll = [];
  ["naive_slo", "robust"].forEach(c => WORKLOADS.forEach(w => { const s0 = run("static_high", w, 0);
    runsOf(c, w).forEach(r => svAll.push(-pctDiff(s0.j_per_token, r.j_per_token))); }));
  const nE = run("naive_slo", "sharp_burst", dmax), rE = run("robust", "sharp_burst", dmax);
  const djw = w => ds.map(d => pctDiff(run("naive_slo", w, d).j_per_token, run("robust", w, d).j_per_token));
  const rng = a => `${sgn(Math.min(...a))} to ${sgn(Math.max(...a))} %`;
  const O = REPd ? REPd.overall : null;
  const ab = c => run(c, "sharp_burst", dmax);
  const C = [
    ["yes", "Concluded", `<b>Stale telemetry hurts a reactive controller.</b> On this replicate naive's burst-phase misses rise ${bmS("naive_slo")} % as the delay goes ${ds.join(" → ")} ms.` +
      (REPd ? ` Pooled over all replicates: ${ds.map(d => pp("naive_slo", d)).join(" → ")} %.` : "")],
    ["yes", "Concluded", `<b>DVFS pays off on this testbed.</b> naive and robust used ${Math.min(...svAll).toFixed(0)}–${Math.max(...svAll).toFixed(0)} % less energy than static_high on every workload and delay.`],
    ["yes", "Concluded", `<b>Robust's lag-aware rule works as designed.</b> After the burst, naive stays at ${nE.end_mhz.toFixed(0)} MHz to the end of the run because its 60 s P95 window still remembers the burst. Robust sees an empty queue and returns to ${rE.end_mhz.toFixed(0)} MHz (CAPACITY_DOWN). Time at the top clock: ${(nE.top_clock_frac * 100).toFixed(0)} % → ${(rE.top_clock_frac * 100).toFixed(0)} %.`],
    ["yes", "Concluded", `<b>Robust cuts burst misses under stale telemetry.</b> Here at ${dmax} ms: ${nE.burst_misses} of ${nE.burst_n} burst requests late for naive, ${rE.burst_misses} for robust; across delays robust is at ${bmS("robust")} %.` +
      (O ? ` Pooled: ${pp("naive_slo", dmax)} % → ${pp("robust", dmax)} %, robust lower in ${O.bm_lower}/${O.pairs} replicate pairs and never higher (Wilcoxon p ≈ ${fmtP(O.p_wil_bm)}). See <a href="#replication">3-seed replication</a>.` : "")],
    ["yes", "Concluded", `<b>Robust uses less energy on bursty traffic.</b> Here ${rng(djw("sharp_burst"))} J/token vs naive at the ${ds.length} delays.` +
      (O ? ` Over all replicates robust was cheaper in ${O.energy_lower}/${O.pairs} pairs, ${sgn(O.dj_mean)} % on average (sign test p ≈ ${fmtP(O.p_sign_energy)}).` : "")],
    ["yes", "Concluded", `<b>Each robust mechanism contributes (ablation).</b> Here at ${dmax} ms: naive ${(ab("naive_slo").burst_miss_rate * 100).toFixed(1)} %, robust_v1 ${(ab("robust_v1").burst_miss_rate * 100).toFixed(1)} %, robust_no_age ${(ab("robust_no_age").burst_miss_rate * 100).toFixed(1)} % (${ab("robust_no_age").n_clock_changes} clock changes), robust ${(ab("robust").burst_miss_rate * 100).toFixed(1)} %.` +
      (REPd ? ` Pooled: ${["naive_slo", "robust_v1", "robust_no_age", "robust"].map(c => pp(c, dmax) + " %").join(", ")}.` : "")],
    ["no", "Cost", `<b>No gain on steady, a small cost on ramp.</b> Here steady is ${rng(djw("steady"))} and ramp ${rng(djw("ramp"))} energy vs naive, with no burst to react to.` +
      (O ? ` Over all replicates ramp at ${dmax} ms costs ${sgn(O.ramp_dmax_dj)} % (higher in ${O.ramp_dmax_higher} of ${new Set(REPd.runs.map(r => r.att + r.seed)).size} pairs).` : "")],
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

/* ───────────── Replication (data/replication.js, build_replication.py) ───────────── */
const fmtP = p => p == null ? "–" : p < 1e-4 ? p.toExponential(0).replace("e-", "×10⁻").replace(/⁻(\d+)/, (_, d) => "⁻" + [...d].map(c => "⁰¹²³⁴⁵⁶⁷⁸⁹"[c]).join("")) : p < 0.001 ? "<0.001" : p.toFixed(3);
const REP = { wl: "sharp_burst" };
function renderReplication() {
  const R = window.REPLICATION;
  if (!R) { $("#page-replication").insertAdjacentHTML("beforeend",
    '<div class="card"><p>No data/replication.js. Run <code>python dashboard/build_replication.py</code>.</p></div>'); return; }
  const O = R.overall, ds = [...new Set(R.paired.map(p => p.delay_ms))].sort((a, b) => a - b), dmax = ds[ds.length - 1];
  const P = (w, d) => R.paired.find(p => p.workload === w && p.delay_ms === d);
  const nRep = new Set(R.runs.map(r => r.att + "/" + r.seed)).size;
  $("#rep-atts").textContent = R.attempts.map(a => `${a.name} (base ${a.base_rate} req/s)`).join(" + ") + ` · ${R.runs.length} GPU runs`;
  const pm = P("sharp_burst", dmax);
  const sv = R.savings.filter(s => s.controller === "robust" && s.workload === "sharp_burst").map(s => -s.mean);
  $("#rep-kpis").innerHTML = [
    [`Burst-phase misses at ${dmax} ms`, `${(pm.bk_n / pm.bn * 100).toFixed(1)}% → <span class="good">${(pm.bk_r / pm.bn * 100).toFixed(1)}%</span>`,
      `naive → robust, ${pm.bk_n} → ${pm.bk_r} of ${pm.bn} burst requests`],
    ["Energy on sharp_burst", `<span class="good">${sgn(O.dj_mean)}%</span>`,
      `robust lower in <b>${O.energy_lower}/${O.pairs}</b> pairs (sign test p = ${fmtP(O.p_sign_energy)})`],
    ["Burst misses, run level", `${O.bm_lower}<small>/${O.pairs} lower</small>`,
      `${O.bm_tie} ties, ${O.bm_higher} higher; mean ${sgn(O.dbm_mean)} pp (Wilcoxon p = ${fmtP(O.p_wil_bm)})`],
    ["Replicates", `${nRep}<small> per condition</small>`, `${R.attempts.length} sessions × ${R.attempts[0].seeds.length} paired seeds, 0 unfinished requests`],
  ].map(([l, v, s]) => `<div class="kpi"><div class="lab">${l}</div><div class="val">${v}</div><div class="sub">${s}</div></div>`).join("");

  // Burst misses: pooled line + replicate dots, all four sharp_burst controllers.
  const sb = R.runs.filter(r => r.workload === "sharp_burst");
  const jit = { naive_slo: -60, robust_v1: -20, robust_no_age: 20, robust: 60 };
  const tr = [];
  ABLATION.forEach(c => {
    const main = PRIMARY.includes(c);
    const ab = ds.map(d => R.ablation.find(a => a.controller === c && a.delay_ms === d));
    tr.push({ x: ds, y: ab.map(a => a.bk / a.bn * 100), name: c, mode: "lines+markers", legendgroup: c,
      line: { color: CTRL[c].color, width: main ? 3 : 1.5, dash: main ? "solid" : "dot" }, marker: { size: main ? 9 : 6 },
      visible: main ? true : "legendonly", hovertemplate: `${c} pooled<br>%{x} ms: %{y:.1f}%<extra></extra>` });
    const rs = sb.filter(r => r.controller === c);
    tr.push({ x: rs.map(r => r.delay_ms + jit[c]), y: rs.map(r => r.bmiss), mode: "markers", legendgroup: c, showlegend: false,
      visible: main ? true : "legendonly", marker: { color: CTRL[c].color, size: 6, opacity: 0.45 },
      text: rs.map(r => `${r.att} s${r.seed}`), hovertemplate: `${c} %{text}<br>%{y:.1f}%<extra></extra>` });
  });
  plot("rep-miss", tr, {
    xaxis: { title: "telemetry delay (ms)", tickvals: ds, gridcolor: css("--line") },
    yaxis: { title: "burst-phase miss (%)", gridcolor: css("--line"), rangemode: "tozero" },
    shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 5, y1: 5, line: { color: css("--bad"), dash: "dash", width: 1 } }],
    annotations: [{ xref: "paper", x: 0.01, y: 5, text: "5% budget", showarrow: false, yshift: 9, font: { color: css("--bad"), size: 11 } }],
  });
  seg($("#rep-wl"), WORKLOADS, REP.wl, v => { REP.wl = v; drawRepDj(); });
  drawRepDj();

  const ab = (c, d) => R.ablation.find(a => a.controller === c && a.delay_ms === d);
  const pct = a => (a.bk / a.bn * 100).toFixed(1);
  const rmp = P("ramp", dmax), sv2 = R.savings.filter(s => s.controller === "robust");
  const svr = w => { const v = sv2.filter(s => s.workload === w).map(s => -s.mean); const a = Math.min(...v).toFixed(0), b = Math.max(...v).toFixed(0); return a === b ? `≈${a} %` : `${a}–${b} %`; };
  const C = [
    ["yes", "Concluded", `<b>Stale telemetry breaks the reactive controller on bursts.</b> naive's pooled burst-phase misses rise ${ds.map(d => pct(ab("naive_slo", d))).join(" → ")} % as the delay goes ${ds.join(" → ")} ms.`],
    ["yes", "Concluded", `<b>Robust cuts burst misses by ${Math.round(Math.min(...ds.map(d => 1 - P("sharp_burst", d).bk_r / P("sharp_burst", d).bk_n)) * 100)}–${Math.round(Math.max(...ds.map(d => 1 - P("sharp_burst", d).bk_r / P("sharp_burst", d).bk_n)) * 100)} % at every delay.</b> Lower in ${O.bm_lower} of ${O.pairs} replicate pairs, never higher (Wilcoxon p = ${fmtP(O.p_wil_bm)}). At ${dmax} ms: ${pct(ab("naive_slo", dmax))} % → ${pct(ab("robust", dmax))} %.`],
    ["yes", "Concluded", `<b>…and uses less energy at the same time.</b> ${sgn(O.dj_mean)} % J/token on sharp_burst, lower in ${O.energy_lower}/${O.pairs} pairs (sign test p = ${fmtP(O.p_sign_energy)}). It spends ≈${Math.round(P("sharp_burst", dmax).top_r)} % of the run at 2122 MHz against ≈${Math.round(P("sharp_burst", dmax).top_n)} % for naive.`],
    ["yes", "Concluded", `<b>Both v2 mechanisms are needed (ablation).</b> robust_v1 ≈ naive (${pct(ab("robust_v1", dmax))} % at ${dmax} ms, p = ${fmtP(ab("robust_v1", dmax).p_vs_naive)}). Dropping age compensation (robust_no_age) keeps only part of the gain (${pct(ab("robust_no_age", dmax))} %) and oscillates (${ab("robust_no_age", dmax).chg} clock changes vs ${ab("robust", dmax).chg}).`],
    ["yes", "Concluded", `<b>DVFS pays off.</b> Robust saves ${svr("sharp_burst")} vs static_high on bursts, ${svr("ramp")} on ramp and ${svr("steady")} on steady.`],
    ["no", "Cost", `<b>No gain on smooth load, a small cost on ramp.</b> Steady is a wash. On ramp at ${dmax} ms robust uses ${sgn(O.ramp_dmax_dj)} % energy (higher in ${O.ramp_dmax_higher}/${rmp.k} pairs, p = ${fmtP(rmp.p_dj)}): it plans ahead for a rise naive handles anyway.`],
    ["trend", "Limit", "<b>Near capacity robust starts to miss too.</b> On seed 43 at base 1.80 req/s (attempt_14, attempt_15) it reaches 8–26 % burst misses at 1250–2500 ms (naive 65–95 %). The failure map still uses one burst size (×4)."],
  ];
  $("#rep-concl").innerHTML = C.map(([k, b, t]) => `<div class="cl"><span class="badge b-${k}">${b}</span><p>${t}</p></div>`).join("");

  const rows = [...R.paired].sort((a, b) => WORKLOADS.indexOf(a.workload) - WORKLOADS.indexOf(b.workload) || a.delay_ms - b.delay_ms);
  table($("#rep-paired"), [
    { k: "workload", h: "workload", l: true }, { k: "delay_ms", h: "delay ms" }, { k: "k", h: "pairs" },
    { k: "dj_mean", h: "Δ energy %", f: (v, r) => `${sgn(v)} ± ${r.dj_sd.toFixed(1)}`, cls: v => v > 0.5 ? "pos" : v < -0.5 ? "neg" : "" },
    { k: "dj_lower", h: "robust cheaper", f: (v, r) => `${v}/${r.k}` }, { k: "p_dj", h: "p (t)", f: fmtP },
    { k: "bk_n", h: "burst miss naive", f: (v, r) => v == null ? "–" : `${v}/${r.bn} (${(v / r.bn * 100).toFixed(1)}%)` },
    { k: "bk_r", h: "burst miss robust", f: (v, r) => v == null ? "–" : `${v}/${r.bn} (${(v / r.bn * 100).toFixed(1)}%)`, cls: (v, r) => v != null && v < r.bk_n ? "neg" : "" },
    { k: "p_fisher", h: "p (Fisher)", f: fmtP }, { k: "p_wil", h: "p (Wilcoxon)", f: fmtP },
    { k: "miss_n", h: "run miss % naive", f: v => v.toFixed(2) }, { k: "miss_r", h: "run miss % robust", f: v => v.toFixed(2) },
  ], rows);

  const reps = [...new Set(sb.map(r => r.att + "|" + r.seed))].sort();
  const cell = (a, s, c, d) => (sb.find(r => r.att === a && r.seed === +s && r.controller === c && r.delay_ms === d) || {}).bmiss;
  $("#rep-grid").innerHTML = "<thead><tr><th class='l'>replicate</th>" + ds.map(d => `<th>${d} ms</th>`).join("") + "</tr></thead><tbody>" +
    reps.map(k => { const [a, s] = k.split("|"); return `<tr><td class="l">${a.replace("attempt_", "a")} · s${s}</td>` +
      ds.map(d => { const n = cell(a, s, "naive_slo", d), r = cell(a, s, "robust", d);
        return `<td>${n.toFixed(1)} → <b class="${r < n ? "good" : ""}">${r.toFixed(1)}</b></td>`; }).join("") + "</tr>"; }).join("") + "</tbody>";

  $("#rep-abl").innerHTML = "<thead><tr><th class='l'>delay</th>" + ABLATION.map(c => `<th style="color:${CTRL[c].color}">${c}</th>`).join("") + "</tr></thead><tbody>" +
    ds.map(d => `<tr><td class="l">${d} ms</td>` + ABLATION.map(c => { const a = ab(c, d);
      return `<td>${a.bk}/${a.bn}<div class="note">${pct(a)} %${a.p_vs_naive != null ? " · p " + fmtP(a.p_vs_naive) : ""}</div></td>`; }).join("") + "</tr>").join("") +
    `<tr><td class="l">J/token @${dmax}</td>` + ABLATION.map(c => `<td>${ab(c, dmax).j_per_token.toFixed(3)}</td>`).join("") + "</tr>" +
    `<tr><td class="l">clock changes @${dmax}</td>` + ABLATION.map(c => `<td>${ab(c, dmax).chg}</td>`).join("") + "</tr></tbody>";

  $("#rep-save").innerHTML = "<thead><tr><th class='l'>controller</th>" + WORKLOADS.map(w => `<th colspan="${ds.length}">${w}</th>`).join("") +
    "</tr><tr><th></th>" + WORKLOADS.map(() => ds.map(d => `<th>${d}</th>`).join("")).join("") + "</tr></thead><tbody>" +
    PRIMARY.map(c => `<tr><td class="l" style="color:${CTRL[c].color}"><b>${c}</b></td>` + WORKLOADS.map(w => ds.map(d => {
      const s = R.savings.find(x => x.controller === c && x.workload === w && x.delay_ms === d);
      return `<td title="min ${s.min}, max ${s.max}">${s.mean.toFixed(1)}</td>`; }).join("")).join("") + "</tr>").join("") + "</tbody>";
}
function drawRepDj() {
  const R = window.REPLICATION, ds = [...new Set(R.paired.map(p => p.delay_ms))].sort((a, b) => a - b);
  const key = r => `${r.att}|${r.seed}|${r.delay_ms}`;
  const nv = Object.fromEntries(R.runs.filter(r => r.controller === "naive_slo" && r.workload === REP.wl).map(r => [key(r), r]));
  const rb = R.runs.filter(r => r.controller === "robust" && r.workload === REP.wl);
  const pts = rb.map(r => ({ d: r.delay_ms, v: (r.j_per_token / nv[key(r)].j_per_token - 1) * 100, t: `${r.att} s${r.seed}`, s: r.seed }));
  const seeds = [...new Set(pts.map(p => p.s))].sort();
  const sym = ["circle", "diamond", "square", "triangle-up"];
  const tr = seeds.map((s, i) => { const q = pts.filter(p => p.s === s);
    return { x: q.map(p => p.d + (i - 1) * 45), y: q.map(p => p.v), mode: "markers", name: "seed " + s,
      marker: { size: 9, symbol: sym[i % 4], color: CTRL.robust.color, opacity: 0.75, line: { width: 1, color: css("--ink-2") } },
      text: q.map(p => p.t), hovertemplate: "%{text}<br>%{y:.1f}%<extra></extra>" }; });
  const pm = ds.map(d => R.paired.find(p => p.workload === REP.wl && p.delay_ms === d));
  tr.push({ x: ds, y: pm.map(p => p.dj_mean), mode: "lines+markers", name: "mean", line: { color: css("--ink-2"), width: 2 },
    marker: { size: 11, symbol: "line-ew-open", line: { width: 3 } }, hovertemplate: "mean %{y:.2f}%<extra></extra>" });
  plot("rep-dj", tr, {
    xaxis: { title: "telemetry delay (ms)", tickvals: ds, gridcolor: css("--line") },
    yaxis: { title: "Δ J/token, robust vs naive (%)", gridcolor: css("--line"), zeroline: true, zerolinecolor: css("--ink-2"), zerolinewidth: 1 },
  });
}

/* ───────────── Results ───────────── */
const RS ={ wl: "sharp_burst", metric: "j_per_token", abl: false };
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
  const A = c => run(c, "sharp_burst", dmax), P = c => (A(c).burst_miss_rate * 100).toFixed(1);
  const topAvg = cs => (cs.reduce((t, c) => t + runsOf(c, "sharp_burst").reduce((u, r) => u + r.top_clock_frac, 0) / ds.length, 0) / cs.length * 100).toFixed(0);
  const RA = window.REPLICATION, pa = c => { const a = RA && RA.ablation.find(x => x.controller === c && x.delay_ms === dmax); return a ? (a.bk / a.bn * 100).toFixed(1) : "–"; };
  $("#ab-text").innerHTML = `<h3>Reading the ablation</h3><ul>
    <li><b>robust_v1 ≈ naive.</b> The old robust controller makes almost the same clock choices as naive (${P("robust_v1")} % vs ${P("naive_slo")} % at ${dmax} ms,
      ${A("robust_v1").n_clock_changes} vs ${A("naive_slo").n_clock_changes} clock changes). Both take every up-step from the shared capacity floor and stay pinned at
      2122 MHz after the burst. This is why v2 was built.</li>
    <li><b>The lag-aware P95 rule drives the energy result.</b> Both v2 variants (robust, robust_no_age) spend about ${topAvg(["robust", "robust_no_age"])} % of the run at the top clock,
      against about ${topAvg(["naive_slo", "robust_v1"])} % for naive and robust_v1, and are the two cheapest controllers in J/token.</li>
    <li><b>Telemetry age drives the reliability result.</b> At ${dmax} ms, robust_no_age forecasts over only the 0.7 s settle time, reacts late,
      then over-corrects (${A("robust_no_age").n_clock_changes} clock changes against ${A("robust").n_clock_changes} for robust). It keeps only part of the gain
      (${P("robust_no_age")} %). The full robust controller forecasts over age + settle and holds misses at ${P("robust")} %.</li>
    <li><b>Replicated.</b> This page shows one replicate per cell. Pooled over all ${RA ? new Set(RA.runs.map(r => r.att + r.seed)).size : ""} replicates the order is the same at ${dmax} ms:
      naive ${pa("naive_slo")} %, robust_v1 ${pa("robust_v1")} %, robust_no_age ${pa("robust_no_age")} %, robust ${pa("robust")} % burst misses. See <a href="#replication">3-seed replication</a>.</li></ul>`;
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
  const cap = D.meta.capacity || {}, pr = D.meta.probe || {}, peak = (pr.base_rate || 1.8) * (pr.burst_multiplier || 4);
  plot("tb-cap", [{ type: "bar", x: (cap.clocks_mhz || []).map(c => c + " MHz"), y: cap.capacity_rps, marker: { color: "#0e9f80" },
    text: (cap.capacity_rps || []).map(v => v.toFixed(2)), textposition: "outside", cliponaxis: false, name: "capacity",
    customdata: cap.batch_time_s, hovertemplate: "%{x}: %{y:.2f} req/s<br>batch of 16 in %{customdata:.2f} s<extra></extra>" }],
    { yaxis: { title: "requests / s", gridcolor: g }, margin: { l: 56, r: 16, t: 24, b: 40 },
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: peak, y1: peak, line: { color: css("--warn"), dash: "dot" } }],
      annotations: [{ xref: "paper", x: 0.01, y: peak, text: `burst peak ${peak.toFixed(1)} req/s`, showarrow: false, yshift: 9, font: { size: 11, color: css("--warn") } }] });
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
    fig1: "Telemetry delay vs J/token", fig2: "Telemetry delay vs SLO miss rate", fig3: "Burst time series (robust, 1250 ms)",
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
const RENDER = { intro: renderIntro, overview: renderOverview, replication: renderReplication, results: renderResults, explorer: renderExplorer, ablation: renderAblation,
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
  const src = d.meta.source_attempt || d.attempt;
  $("#side-foot").innerHTML = `Showcase: results/${src}<br>seed ${d.meta.seeds.join(", ")} only · ${d.meta.n_runs} GPU runs` +
    (window.REPLICATION ? `<br>Replication: ${window.REPLICATION.attempts.map(a => a.name).join(", ")}<br>${window.REPLICATION.runs.length} GPU runs · seeds ${window.REPLICATION.attempts[0].seeds.join(", ")}` : "");
  const note = `<div class="seed-note"><b>Showcase replicate: ${src}, seed ${d.meta.seeds.join(", ")}</b> (${d.meta.n_runs} GPU runs).
    Charts and numbers on this page come from this one seed. ${SHOWCASE_WHY} Statistics over all replicates are on
    <a href="#replication">3-seed replication</a>.</div>`;
  ["overview", "results", "explorer", "ablation", "testbed", "figures"].forEach(id =>
    document.querySelector(`#page-${id} .page-head`).insertAdjacentHTML("beforeend", note));
  document.querySelectorAll(".seed-label").forEach(e => e.textContent = `${src}, seed ${d.meta.seeds.join(", ")}`);
  addEventListener("hashchange", route);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { done.clear(); route(); });
  route();
}).catch(e => { $("#main").innerHTML = `<div class="card"><h3>Could not load data</h3><p>${e.message}</p></div>`; });
