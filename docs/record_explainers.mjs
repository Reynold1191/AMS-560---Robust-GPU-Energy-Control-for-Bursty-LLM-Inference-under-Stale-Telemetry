#!/usr/bin/env node
/*
  docs/record_explainers.mjs
  Captures frames of the dashboard explainer animations (dashboard/explainers.js) with a
  headless Edge or Chrome over the DevTools protocol. docs/make_explainer_gifs.py calls it
  and turns the frames into README GIFs.

    node docs/record_explainers.mjs --kind kitchen --from 30 --to 80 --step 0.3 --out <dir>
    node docs/record_explainers.mjs --kind machine --ctrl naive_slo --from 30 --to 80 --step 0.3 --out <dir>
    node docs/record_explainers.mjs --kind kitchen --at 46 --out <dir>          # one still

  Needs Node 22+ (global WebSocket) and Edge or Chrome. Set BROWSER to pick the binary.

  A headless browser stalls after about 200 screenshots (each one takes longer, up to minutes),
  so frames are taken in chunks of --chunk (default 150), each with a fresh browser.
*/
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn, spawnSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf("--" + k); return i < 0 ? d : args[i + 1]; };
const kind = opt("kind", "kitchen"), ctrl = opt("ctrl", "robust"), out = path.resolve(opt("out", "frames"));
const at = opt("at"), from = +opt("from", 30), to = +opt("to", 80), step = +opt("step", 0.3), scale = +opt("scale", 1);
const CHUNK = +opt("chunk", 150), CALL_MS = 20000;
const HERE = path.dirname(fileURLToPath(import.meta.url));
const page = pathToFileURL(path.join(HERE, "xp_capture.html")).href;

const candidates = [process.env.BROWSER,
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe", "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Google/Chrome/Application/chrome.exe", "/usr/bin/google-chrome", "/usr/bin/chromium",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"].filter(Boolean);
const bin = candidates.find(p => fs.existsSync(p));
if (!bin) { console.error("No Edge or Chrome found; set BROWSER."); process.exit(1); }

const sleep = ms => new Promise(r => setTimeout(r, ms));
async function json(u) { for (let i = 0; i < 50; i++) { try { return await (await fetch(u)).json(); } catch { await sleep(200); } } throw new Error("no devtools"); }

const live = new Set();  // browsers still running; ended on any exit
// On Windows msedge.exe hands off to another process, so the launched PID is not the browser:
// end every process started with this run's profile folder.
function kill(br) {
  if (process.platform === "win32") {
    const dir = path.basename(br.prof).replace(/'/g, "");
    spawnSync("powershell", ["-NoProfile", "-Command", `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*${dir}*' -and $_.ProcessId -ne $PID } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }`], { stdio: "ignore" });
  } else br.kill("SIGKILL");
  live.delete(br);
}
process.on("exit", () => live.forEach(kill));
for (const s of ["SIGINT", "SIGTERM", "SIGHUP"]) process.on(s, () => process.exit(130));

// Takes frames times[i0..i1) with one fresh browser.
async function chunk(times, i0, i1) {
  const port = 9300 + Math.floor(Math.random() * 600);
  const prof = fs.mkdtempSync(path.join(os.tmpdir(), "xp-prof-"));
  const br = spawn(bin, ["--headless=new", "--disable-gpu", "--hide-scrollbars", `--remote-debugging-port=${port}`,
    `--user-data-dir=${prof}`, "--allow-file-access-from-files", "about:blank"], { stdio: "ignore" });
  br.prof = prof;
  live.add(br);
  try {
    const tabs = await json(`http://127.0.0.1:${port}/json/list`);
    const ws = new WebSocket(tabs.find(t => t.type === "page").webSocketDebuggerUrl);
    await new Promise(r => ws.addEventListener("open", r, { once: true }));
    let id = 0; const wait = new Map();
    ws.addEventListener("message", e => { const m = JSON.parse(e.data); if (m.id && wait.has(m.id)) { wait.get(m.id)(m); wait.delete(m.id); } });
    const send = (method, params = {}) => new Promise((res, rej) => {
      const i = ++id, timer = setTimeout(() => { wait.delete(i); rej(new Error(`${method} timed out`)); }, CALL_MS);
      wait.set(i, m => { clearTimeout(timer); m.error ? rej(new Error(m.error.message)) : res(m.result); });
      ws.send(JSON.stringify({ id: i, method, params }));
    });

    await send("Emulation.setDeviceMetricsOverride", { width: 1200, height: 700, deviceScaleFactor: scale, mobile: false });
    await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: "light" }, { name: "prefers-reduced-motion", value: "no-preference" }] });
    await send("Page.enable");
    await send("Page.navigate", { url: page });
    await sleep(1200);
    for (let n = i0; n < i1; n++) {
      const r = await send("Runtime.evaluate", { expression: `setT(${JSON.stringify(kind)}, ${times[n]}, ${JSON.stringify(ctrl)})`, returnByValue: true });
      const h = Math.ceil(r.result.value);
      const shot = await send("Page.captureScreenshot", { format: "png", clip: { x: 0, y: 0, width: 1200, height: h, scale: 1 } });
      fs.writeFileSync(path.join(out, `f${String(n).padStart(4, "0")}.png`), Buffer.from(shot.data, "base64"));
    }
    ws.close();
  } finally {
    kill(br);
    await sleep(500);
    try { fs.rmSync(prof, { recursive: true, force: true }); } catch {}
  }
}

fs.mkdirSync(out, { recursive: true });
const times = at != null ? [+at] : Array.from({ length: Math.floor((to - from) / step) + 1 }, (_, i) => +(from + i * step).toFixed(3));
for (let i0 = 0; i0 < times.length; i0 += CHUNK) {
  const i1 = Math.min(times.length, i0 + CHUNK);
  for (let tries = 1; ; tries++) {
    try { await chunk(times, i0, i1); break; }
    catch (e) { if (tries >= 3) throw e; console.log(`  frames ${i0}-${i1 - 1}: ${e.message}, retrying`); }
  }
  console.log(`  ${i1}/${times.length} frames`);
}
console.log(`${times.length} frame(s) -> ${out}`);
