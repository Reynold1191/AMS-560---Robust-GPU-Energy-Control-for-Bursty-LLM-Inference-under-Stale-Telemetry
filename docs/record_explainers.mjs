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
*/
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf("--" + k); return i < 0 ? d : args[i + 1]; };
const kind = opt("kind", "kitchen"), ctrl = opt("ctrl", "robust"), out = path.resolve(opt("out", "frames"));
const at = opt("at"), from = +opt("from", 30), to = +opt("to", 80), step = +opt("step", 0.3), scale = +opt("scale", 1);
const HERE = path.dirname(fileURLToPath(import.meta.url));
const page = pathToFileURL(path.join(HERE, "xp_capture.html")).href;

const candidates = [process.env.BROWSER,
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe", "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Google/Chrome/Application/chrome.exe", "/usr/bin/google-chrome", "/usr/bin/chromium",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"].filter(Boolean);
const bin = candidates.find(p => fs.existsSync(p));
if (!bin) { console.error("No Edge or Chrome found; set BROWSER."); process.exit(1); }

const port = 9300 + Math.floor(Math.random() * 90);
const prof = fs.mkdtempSync(path.join(os.tmpdir(), "xp-prof-"));
const br = spawn(bin, ["--headless=new", "--disable-gpu", "--hide-scrollbars", `--remote-debugging-port=${port}`,
  `--user-data-dir=${prof}`, "--allow-file-access-from-files", "about:blank"], { stdio: "ignore" });

const sleep = ms => new Promise(r => setTimeout(r, ms));
async function json(u) { for (let i = 0; i < 50; i++) { try { return await (await fetch(u)).json(); } catch { await sleep(200); } } throw new Error("no devtools"); }

try {
  const tabs = await json(`http://127.0.0.1:${port}/json/list`);
  const ws = new WebSocket(tabs.find(t => t.type === "page").webSocketDebuggerUrl);
  await new Promise(r => ws.addEventListener("open", r, { once: true }));
  let id = 0; const wait = new Map();
  ws.addEventListener("message", e => { const m = JSON.parse(e.data); if (m.id && wait.has(m.id)) { wait.get(m.id)(m); wait.delete(m.id); } });
  const send = (method, params = {}) => new Promise((res, rej) => { const i = ++id; wait.set(i, m => m.error ? rej(new Error(m.error.message)) : res(m.result)); ws.send(JSON.stringify({ id: i, method, params })); });

  await send("Emulation.setDeviceMetricsOverride", { width: 1200, height: 700, deviceScaleFactor: scale, mobile: false });
  await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: "light" }, { name: "prefers-reduced-motion", value: "no-preference" }] });
  await send("Page.enable");
  await send("Page.navigate", { url: page });
  await sleep(1200);
  fs.mkdirSync(out, { recursive: true });
  const times = at != null ? [+at] : Array.from({ length: Math.floor((to - from) / step) + 1 }, (_, i) => +(from + i * step).toFixed(3));
  let n = 0;
  for (const t of times) {
    const r = await send("Runtime.evaluate", { expression: `setT(${JSON.stringify(kind)}, ${t}, ${JSON.stringify(ctrl)})`, returnByValue: true });
    const h = Math.ceil(r.result.value);
    const shot = await send("Page.captureScreenshot", { format: "png", clip: { x: 0, y: 0, width: 1200, height: h, scale: 1 } });
    fs.writeFileSync(path.join(out, `f${String(n++).padStart(4, "0")}.png`), Buffer.from(shot.data, "base64"));
  }
  console.log(`${n} frame(s) -> ${out}`);
  ws.close();
} finally {
  br.kill();
  await sleep(300);
  try { fs.rmSync(prof, { recursive: true, force: true }); } catch {}
}
