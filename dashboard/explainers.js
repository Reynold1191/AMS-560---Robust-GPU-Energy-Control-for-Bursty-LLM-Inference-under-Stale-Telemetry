/* Two explainer animations that replay real GPU runs (data/explainers.js):
   - kitchen: naive vs robust manager side by side, reading a status board that lags 2.5 s
   - machine: prompts -> requests -> queue -> batch on the GPU -> responses, plus the delayed control loop
   Every frame is a pure function of time t (seconds into the run), so the same code
   draws the dashboard player and the README GIFs (window.XP.render(kind, t)). */
(function () {
  const X = window.EXPLAINERS;
  if (!X) return;
  const RUN_END = 110, DELAY = X.delay_ms / 1000, SLO = X.slo_s;
  const COL = { naive_slo: "#e07b24", robust: "#0e9f80" };
  const NAME = { naive_slo: "Naive manager", robust: "Robust manager" };
  const TAG = { naive_slo: "trusts the board as if it were live", robust: "knows the board is 2.5 s old and plans ahead" };
  const LVL = { 180: 0, 667: 1, 1147: 2, 1635: 3, 2122: 4 };
  const esc = s => String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const ease = u => 1 - Math.pow(1 - clamp(u, 0, 1), 3);

  // ── per-run state at time t ──
  function state(c, t) {
    const R = X.runs[c], ctl = R.ctl;
    let k = 0; while (k + 1 < ctl.length && ctl[k + 1].t <= t) k++;
    const C = ctl[k], prev = ctl[Math.max(0, k - 1)];
    const arrived = R.reqs.filter(r => r[0] <= t);
    const waiting = arrived.filter(r => r[1] > t);
    const cooking = arrived.filter(r => r[1] <= t && r[2] > t);
    const done = arrived.filter(r => r[2] <= t);
    const burstDone = done.filter(r => r[7] === 2);
    const i = clamp(Math.floor(t * 2), 0, R.power.length - 1);
    return { R, C, prev, arrived, waiting, cooking, done, burstLate: burstDone.filter(r => r[3]).length, burstDone: burstDone.length,
      lateAll: done.filter(r => r[3]).length, power: R.power[i], energy: R.energy[i] };
  }
  const phaseAt = t => (X.phases.find(p => t >= p[0] && t < p[1]) || X.phases[X.phases.length - 1])[2];
  const PH = { steady: "normal traffic", burst: "RUSH ×4", recovery: "after the rush" };
  const eMax = Math.max(...Object.values(X.runs).map(r => r.energy[r.energy.length - 1]));

  // ── shared pieces ──
  function timeline(t, x0, x1, y) {
    const sx = s => x0 + (x1 - x0) * s / RUN_END;
    let s = `<g class="xp-tl">`;
    X.phases.forEach(([a, b, n]) => {
      s += `<rect x="${sx(a)}" y="${y}" width="${sx(b) - sx(a)}" height="10" rx="3" class="ph-${n}"/>`;
      s += `<text x="${(sx(a) + sx(b)) / 2}" y="${y + 26}" class="xp-tiny" text-anchor="middle">${PH[n]}</text>`;
    });
    s += `<line x1="${sx(t)}" x2="${sx(t)}" y1="${y - 6}" y2="${y + 16}" class="xp-head"/>`;
    s += `<circle cx="${sx(t)}" cy="${y + 5}" r="5" class="xp-headdot"/></g>`;
    return s;
  }
  const machineWords = s => s.replace("Full flame", "Max clock").replace(/the stove|the flame/g, "the clock").replace(/stove|flame/g, "clock")
    .replace(/dishes/g, "answers").replace(/the rail/gi, m => m[0] === "T" ? "The queue" : "the queue").replace("orders", "requests");
  const fo = (x, y, w, h, html, cls = "") => `<foreignObject x="${x}" y="${y}" width="${w}" height="${h}"><div xmlns="http://www.w3.org/1999/xhtml" class="xp-fo ${cls}">${html}</div></foreignObject>`;

  // ══════════ KITCHEN ══════════
  function kitchenLane(c, t, y0) {
    const S = state(c, t), C = S.C, col = COL[c];
    let s = `<g transform="translate(0 ${y0})">`;
    s += `<rect x="8" y="0" width="1184" height="236" rx="14" class="xp-lane"/>`;
    s += `<circle cx="30" cy="24" r="6" fill="${col}"/><text x="44" y="29" class="xp-h">${NAME[c]}</text>`;
    s += `<text x="${44 + NAME[c].length * 9.2}" y="29" class="xp-sub"> · ${TAG[c]}</text>`;

    // status board (what the manager sees: 2.5 s old)
    s += `<rect x="24" y="44" width="236" height="96" rx="9" class="xp-board"/>`;
    s += `<text x="38" y="64" class="xp-board-t">BOARD · ${DELAY} S OLD</text><text x="246" y="64" class="xp-board-t" text-anchor="end">t = ${Math.max(0, t - DELAY).toFixed(1)} s</text>`;
    s += `<text x="38" y="94" class="xp-board-k">orders waiting</text><text x="246" y="94" class="xp-board-v" text-anchor="end">${C.oq}</text>`;
    s += `<text x="38" y="124" class="xp-board-k">recent delivery (P95)</text><text x="246" y="124" class="xp-board-v ${C.op95 > SLO ? "hot" : ""}" text-anchor="end">${C.op95.toFixed(1)} s</text>`;
    // manager's call
    const changed = C.mhz !== S.prev.mhz;
    s += fo(24, 150, 236, 76, `<b>🧑‍💼 ${esc(C.say)}</b>${changed ? `<span class="xp-chg" style="color:${col}">stove ${S.prev.mhz} → ${C.mhz} MHz</span>` : ""}`, "xp-bubble");

    // ticket rail (reality)
    const q = S.waiting.length, gap = q > 27 ? 300 / q : 11;
    s += `<text x="290" y="64" class="xp-k">ticket rail · really waiting</text><text x="640" y="64" class="xp-v" text-anchor="end">${q}</text>`;
    if (Math.abs(q - C.oq) >= 5) s += `<text x="640" y="84" class="xp-warn" text-anchor="end">board says ${C.oq}</text>`;
    s += `<line x1="290" x2="640" y1="104" y2="104" class="xp-rail"/>`;
    S.waiting.forEach((r, k) => {
      const fly = ease((t - r[0]) / 0.6), tx = 294 + k * gap, x = 272 + (tx - 272) * fly, y = 106 - Math.sin(fly * Math.PI) * 14;
      const late = t - r[0] > SLO;
      s += `<rect x="${x}" y="${y}" width="10" height="22" rx="2" opacity="${0.25 + 0.75 * fly}" class="xp-ticket ${late ? "late" : r[7] === 2 ? "rush" : ""}"/>`;
    });

    // stove
    const lv = LVL[C.mhz] ?? 2, fs = 18 + lv * 11;
    s += `<text x="680" y="64" class="xp-k">stove</text><text x="850" y="64" class="xp-v" text-anchor="end">${C.mhz} MHz</text>`;
    for (let k = 0; k < 5; k++) s += `<rect x="${680 + k * 34}" y="74" width="30" height="6" rx="3" class="${k <= lv ? "xp-lvl on" : "xp-lvl"}" ${k <= lv ? `style="fill:${col}"` : ""}/>`;
    s += `<ellipse cx="765" cy="138" rx="78" ry="22" class="xp-pan"/>`;
    S.cooking.forEach((r, k) => { s += `<circle cx="${715 + (k % 8) * 14.5}" cy="${132 + Math.floor(k / 8) * 13}" r="5.5" class="xp-dish ${r[7] === 2 ? "rush" : ""}"/>`; });
    s += `<text x="765" y="${200 + lv * 2}" font-size="${fs}" text-anchor="middle" class="xp-emoji">🔥</text>`;
    s += `<text x="765" y="226" class="xp-tiny" text-anchor="middle">${S.cooking.length ? `cooking ${S.cooking.length} orders` : "idle"} · ${S.power.toFixed(0)} W</text>`;

    // the pass: dishes leaving
    s += `<text x="880" y="64" class="xp-k">dishes out</text>`;
    S.done.filter(r => t - r[2] < 2.4).forEach((r, k) => {
      const u = ease((t - r[2]) / 2.4);
      s += `<circle cx="${880 + u * 150 + (k % 2) * 9}" cy="${96 + (k % 8) * 14}" r="6" class="xp-plate ${r[3] ? "late" : ""}" opacity="${1 - u * 0.65}"/>`;
    });
    // scoreboard
    s += `<text x="1180" y="64" class="xp-k" text-anchor="end">late in the rush</text>`;
    s += `<text x="1180" y="104" class="xp-big ${S.burstLate ? "bad" : "good"}" text-anchor="end">${S.burstLate}<tspan class="xp-of"> / ${S.burstDone}</tspan></text>`;
    s += `<text x="1180" y="146" class="xp-k" text-anchor="end">gas used</text>`;
    s += `<text x="1180" y="176" class="xp-v" text-anchor="end">${(S.energy / 1000).toFixed(2)} kJ</text>`;
    s += `<rect x="1060" y="190" width="120" height="7" rx="3.5" class="xp-track"/><rect x="1060" y="190" width="${120 * S.energy / eMax}" height="7" rx="3.5" fill="${col}"/>`;
    return s + `</g>`;
  }
  function kitchen(t) {
    let s = `<svg viewBox="0 0 1200 560" class="xp-svg" role="img" aria-label="Kitchen replay: naive and robust managers during a rush with a 2.5 s old status board">`;
    s += `<text x="20" y="27" class="xp-clock">t = ${t.toFixed(1)} s</text><text x="132" y="27" class="xp-phase ph-t-${phaseAt(t)}">${PH[phaseAt(t)]}</text>`;
    s += timeline(t, 290, 1180, 14);
    s += kitchenLane("naive_slo", t, 58) + kitchenLane("robust", t, 312);
    return s + `<text x="1192" y="556" class="xp-tiny" text-anchor="end">Real GPU runs: ${esc(X.source)}. Same orders for both managers.</text></svg>`;
  }

  // ══════════ MACHINE ══════════
  function machine(t, c) {
    const S = state(c, t), C = S.C, col = COL[c], R = S.R;
    let s = `<svg viewBox="0 0 1200 640" class="xp-svg" role="img" aria-label="Machine view: prompts become requests, wait in a queue, run in batches on the GPU and come back as responses">`;
    s += `<text x="20" y="27" class="xp-clock">t = ${t.toFixed(1)} s</text><text x="132" y="27" class="xp-phase ph-t-${phaseAt(t)}">${PH[phaseAt(t)]}</text>`;
    s += timeline(t, 290, 1180, 14);

    // 1. prompt card: latest arrival
    const last = S.arrived[S.arrived.length - 1];
    const P = last ? X.prompts[last[6]] : X.prompts[0];
    s += `<text x="20" y="80" class="xp-step">1</text><text x="44" y="80" class="xp-h">A user sends a prompt</text>`;
    s += fo(20, 92, 270, 214, `<div class="xp-pr-top"><span class="xp-kind">${P.kind}</span><span>request #${S.arrived.length}</span></div>
      <div class="xp-pr-body">${esc(P.head)}<br>${P.body.map(l => esc(l.slice(0, 46)) + "…").join("<br>")}<br><i>… ${P.n_lines - 6} more lines …</i><br><b>${esc(P.question)}</b></div>
      <div class="xp-pr-foot">${last ? last[4] : P.tokens} tokens in · up to ${P.max_new} tokens out</div>`, "xp-prompt");
    // arrival-rate sparkline (real)
    const ctl = R.ctl, sx = v => 20 + 270 * v / RUN_END, sy = v => 384 - v * 6.5;
    let path = ""; ctl.forEach((k, i) => { if (k.t <= t) path += (i ? "L" : "M") + sx(k.t).toFixed(1) + " " + sy(k.rate).toFixed(1); });
    s += `<text x="20" y="328" class="xp-k">arrivals per second</text><text x="290" y="328" class="xp-v" text-anchor="end">${C.rate.toFixed(1)}</text>`;
    s += `<rect x="${sx(40)}" y="336" width="${sx(60) - sx(40)}" height="50" class="xp-burstband"/>`;
    s += `<line x1="20" x2="290" y1="386" y2="386" class="xp-axis"/><path d="${path}" class="xp-spark"/>`;

    // 2. queue
    s += `<text x="318" y="80" class="xp-step">2</text><text x="342" y="80" class="xp-h">Requests wait in a queue</text>`;
    s += `<path d="M292 190 H318" class="xp-arrow"/>`;
    s += `<rect x="330" y="96" width="170" height="290" rx="10" class="xp-box"/>`;
    const q = S.waiting.length;
    S.waiting.slice(0, 22).forEach((r, k) => {
      const u = ease((t - r[0]) / 0.5), y = 372 - k * 12.6;
      s += `<rect x="${346 - (1 - u) * 10}" y="${y}" width="138" height="9.5" rx="3" class="xp-req ${t - r[0] > SLO ? "late" : r[7] === 2 ? "rush" : ""}" opacity="${0.3 + 0.7 * u}"/>`;
    });
    s += `<text x="415" y="408" class="xp-k" text-anchor="middle"><tspan class="xp-v">${q}</tspan> waiting${q > 22 ? " (22 shown)" : ""}</text>`;

    // 3. GPU
    s += `<text x="532" y="80" class="xp-step">3</text><text x="556" y="80" class="xp-h">The GPU runs a batch of up to 16</text>`;
    s += `<path d="M502 240 H528" class="xp-arrow"/>`;
    s += `<rect x="540" y="96" width="300" height="290" rx="12" class="xp-gpu"/>`;
    s += `<text x="560" y="122" class="xp-gpu-t">RTX 5060 Laptop · Qwen2.5-0.5B</text>`;
    const cook = S.cooking, b0 = cook.length ? cook[0][1] : 0, b1 = cook.length ? Math.max(...cook.map(r => r[2])) : 1;
    for (let k = 0; k < 16; k++) {
      const r = cook[k], x = 562 + (k % 8) * 33, y = 140 + Math.floor(k / 8) * 36;
      s += `<rect x="${x}" y="${y}" width="27" height="28" rx="5" class="xp-slot ${r ? (r[7] === 2 ? "on rush" : "on") : ""}"/>`;
    }
    const prog = cook.length ? clamp((t - b0) / (b1 - b0), 0, 1) : 0;
    s += `<text x="560" y="234" class="xp-gpu-k">${cook.length ? `batch of ${cook.length}: read prompts, then write answers` : "idle, waiting for work"}</text>`;
    s += `<rect x="560" y="244" width="260" height="8" rx="4" class="xp-gpu-track"/><rect x="560" y="244" width="${260 * prog}" height="8" rx="4" fill="${col}"/>`;
    const lv = LVL[C.mhz] ?? 2;
    s += `<text x="560" y="286" class="xp-gpu-k">clock</text><text x="820" y="286" class="xp-gpu-v" text-anchor="end">${C.mhz} MHz</text>`;
    for (let k = 0; k < 5; k++) s += `<rect x="${560 + k * 52.5}" y="294" width="48" height="10" rx="3" class="xp-gpu-lvl ${k <= lv ? "on" : ""}" ${k <= lv ? `style="fill:${col}"` : ""}/>`;
    s += `<text x="560" y="336" class="xp-gpu-k">power</text><text x="820" y="336" class="xp-gpu-v" text-anchor="end">${S.power.toFixed(0)} W</text>`;
    s += `<rect x="560" y="344" width="260" height="8" rx="4" class="xp-gpu-track"/><rect x="560" y="344" width="${260 * clamp(S.power / 80, 0, 1)}" height="8" rx="4" class="xp-pw"/>`;
    s += `<text x="560" y="374" class="xp-gpu-k">energy so far</text><text x="820" y="374" class="xp-gpu-v" text-anchor="end">${(S.energy / 1000).toFixed(2)} kJ</text>`;

    // 4. responses + latency strip
    s += `<text x="872" y="80" class="xp-step">4</text><text x="896" y="80" class="xp-h">Answers go back</text>`;
    s += `<path d="M842 240 H868" class="xp-arrow"/>`;
    const fresh = S.done.slice(-6);
    fresh.forEach((r, k) => {
      const u = ease((t - r[2]) / 1.2), lat = r[2] - r[0];
      s += `<g opacity="${clamp(1.15 - (t - r[2]) / 8, 0.35, 1)}" transform="translate(${880 + u * 196} ${100 + k * 22})"><rect width="92" height="18" rx="9" class="xp-resp ${r[3] ? "late" : ""}"/>
        <text x="46" y="13" class="xp-resp-t" text-anchor="middle">${lat.toFixed(1)} s ${r[3] ? "✗" : "✓"}</text></g>`;
    });
    // latency of each finished request vs SLO
    const ly = v => 386 - clamp(v, 0, 12) * 12, lx = v => 880 + 300 * v / RUN_END;
    s += `<text x="880" y="250" class="xp-k">time from prompt to answer</text>`;
    s += `<line x1="880" x2="1180" y1="${ly(SLO)}" y2="${ly(SLO)}" class="xp-slo"/><text x="1180" y="${ly(SLO) - 5}" class="xp-tiny bad" text-anchor="end">5 s promise (SLO)</text>`;
    S.done.forEach(r => { s += `<circle cx="${lx(r[2])}" cy="${ly(r[2] - r[0])}" r="2.6" class="xp-pt ${r[3] ? "late" : ""}"/>`; });
    s += `<line x1="880" x2="1180" y1="386" y2="386" class="xp-axis"/>`;
    s += `<text x="880" y="408" class="xp-k">late: <tspan class="${S.lateAll ? "bad" : "good"}" font-weight="700">${S.lateAll}</tspan> of ${S.done.length} answered</text>`;

    // 5. control loop with delay buffer
    const y = 450;
    s += `<rect x="20" y="${y - 22}" width="1160" height="200" rx="14" class="xp-lane"/>`;
    s += `<text x="40" y="${y + 4}" class="xp-step">5</text><text x="64" y="${y + 4}" class="xp-h">Every second the controller resets the clock, using telemetry that is ${DELAY} s old</text>`;
    // sampler (right) -> delay pipe -> controller (left) -> clock lock back up to GPU
    s += `<rect x="930" y="${y + 24}" width="230" height="120" rx="10" class="xp-box"/><text x="946" y="${y + 46}" class="xp-k">live readings now</text>`;
    s += `<text x="946" y="${y + 76}" class="xp-k2">queue</text><text x="1144" y="${y + 76}" class="xp-v" text-anchor="end">${q}</text>`;
    s += `<text x="946" y="${y + 102}" class="xp-k2">P95 latency</text><text x="1144" y="${y + 102}" class="xp-v" text-anchor="end">${C.p95.toFixed(1)} s</text>`;
    s += `<text x="946" y="${y + 128}" class="xp-k2">power</text><text x="1144" y="${y + 128}" class="xp-v" text-anchor="end">${S.power.toFixed(0)} W</text>`;
    s += `<path d="M1045 ${y + 22} V392" class="xp-arrow-up"/>`;
    // the pipe
    s += `<rect x="370" y="${y + 70}" width="550" height="26" rx="13" class="xp-pipe"/>`;
    s += `<text x="645" y="${y + 60}" class="xp-k" text-anchor="middle">delay buffer: readings arrive ${DELAY} s late</text>`;
    for (let k = Math.ceil(t - DELAY); k <= Math.floor(t); k++) {
      if (k < 0) continue;
      const u = (t - k) / DELAY, x = 908 - u * 526;
      s += `<g transform="translate(${x} ${y + 83})"><circle r="9" class="xp-pkt"/><text y="4" class="xp-pkt-t" text-anchor="middle">${k}</text></g>`;
    }
    s += `<text x="645" y="${y + 118}" class="xp-tiny" text-anchor="middle">each dot is one snapshot, labelled with the second it was taken</text>`;
    // controller
    s += `<rect x="40" y="${y + 24}" width="320" height="120" rx="10" class="xp-ctl" style="stroke:${col}"/>`;
    s += `<text x="56" y="${y + 46}" class="xp-k">controller sees (from ${Math.max(0, t - DELAY).toFixed(0)} s)</text><text x="344" y="${y + 46}" class="xp-k" text-anchor="end" style="fill:${col};font-weight:700">${c === "robust" ? "robust" : "naive"}</text>`;
    s += `<text x="56" y="${y + 70}" class="xp-k2">queue ${C.oq} · P95 ${C.op95.toFixed(1)} s</text>`;
    s += fo(52, y + 78, 300, 60, `${esc(machineWords(C.say))}<br><span style="color:${col}">→ lock GPU clock at ${C.mhz} MHz</span>`, "xp-say");
    s += `<path d="M200 ${y + 22} V392 H536" class="xp-arrow-up" fill="none"/>`;
    s += `<text x="210" y="${y - 30}" class="xp-tiny">nvidia-smi --lock-gpu-clocks</text>`;
    return s + `<text x="1192" y="636" class="xp-tiny" text-anchor="end">Real GPU run: ${NAME[c].toLowerCase()}, ${esc(X.source)}.</text></svg>`;
  }

  const DRAW = { kitchen: (t) => kitchen(t), machine: (t, c) => machine(t, c || "robust") };
  window.XP = { render: (kind, t, c) => DRAW[kind](t, c), END: RUN_END };

  // ── player ──
  function mount(el) {
    const kind = el.dataset.xp, reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const st = { t: reduce ? 46 : 30, play: !reduce, speed: 4, c: "robust", visible: true, last: 0 };
    el.innerHTML = `<div class="xp-stage"></div>
      <div class="xp-bar">
        <button class="xp-btn xp-play" type="button" aria-label="Pause"></button>
        <input class="xp-scrub" type="range" min="0" max="${RUN_END}" step="0.1" aria-label="Time in the run">
        <div class="seg xp-speed">${[1, 4, 8].map(v => `<button type="button" data-v="${v}">${v}×</button>`).join("")}</div>
        ${kind === "machine" ? `<div class="seg xp-ctrl"><button type="button" data-v="naive_slo">naive</button><button type="button" data-v="robust">robust</button></div>` : ""}
        <button class="xp-btn xp-jump" type="button">Jump to the rush</button>
      </div>`;
    const stage = el.querySelector(".xp-stage"), play = el.querySelector(".xp-play"), scrub = el.querySelector(".xp-scrub");
    const sync = () => {
      play.textContent = st.play ? "❚❚ Pause" : "▶ Play"; play.setAttribute("aria-label", st.play ? "Pause" : "Play");
      el.querySelectorAll(".xp-speed button").forEach(b => b.classList.toggle("on", +b.dataset.v === st.speed));
      el.querySelectorAll(".xp-ctrl button").forEach(b => b.classList.toggle("on", b.dataset.v === st.c));
    };
    const draw = () => { stage.innerHTML = DRAW[kind](st.t, st.c); scrub.value = st.t; };
    play.onclick = () => { st.play = !st.play; if (st.t >= RUN_END) st.t = 0; sync(); };
    scrub.oninput = () => { st.t = +scrub.value; draw(); };
    el.querySelector(".xp-jump").onclick = () => { st.t = 37; draw(); };
    el.querySelectorAll(".xp-speed button").forEach(b => b.onclick = () => { st.speed = +b.dataset.v; sync(); });
    el.querySelectorAll(".xp-ctrl button").forEach(b => b.onclick = () => { st.c = b.dataset.v; sync(); draw(); });
    new IntersectionObserver(es => es.forEach(e => st.visible = e.isIntersecting)).observe(el);
    const tick = now => {
      const dt = Math.min(0.1, (now - (st.last || now)) / 1000); st.last = now;
      if (st.play && st.visible && el.offsetParent) {
        st.t += dt * st.speed;
        if (st.t > RUN_END) st.t = 0;
        if (now - (st.drawn || 0) > 33) { draw(); st.drawn = now; }
      }
      requestAnimationFrame(tick);
    };
    sync(); draw(); requestAnimationFrame(tick);
  }
  window.XP.mountAll = () => document.querySelectorAll("[data-xp]:not(.xp-ready)").forEach(el => { el.classList.add("xp-ready"); mount(el); });
})();
