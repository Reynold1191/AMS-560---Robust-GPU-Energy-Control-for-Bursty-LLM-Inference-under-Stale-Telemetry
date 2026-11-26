"""
docs/make_explainer_gifs.py
Builds the two explainer GIFs used in README.md from the dashboard animations
(dashboard/explainers.js), so the README and the dashboard show the same thing.

  python dashboard/build_explainers.py        # data (real GPU runs) first
  python docs/make_explainer_gifs.py

Writes docs/img/kitchen_rush.gif and docs/img/machine_pipeline.gif.
Needs Node 22+ and Edge or Chrome (see docs/record_explainers.mjs) and Pillow.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tempfile

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "img")
UI_COLOURS = [  # dashboard/style.css (light) and the explainer palette
    "#f6f7f9", "#ffffff", "#14181f", "#4a5361", "#7b8494", "#e3e6eb", "#0e7c66", "#e3f3ef", "#b45309", "#fdf1e2",
    "#b42318", "#fdecea", "#e07b24", "#0e9f80", "#18232b", "#8fa3b3", "#c9d6df", "#f3f7fa", "#ff8b7f", "#fff7e0",
    "#c9a64a", "#ffe2b8", "#d98c1f", "#3b4450", "#f6d58e", "#ffb45c", "#121a21", "#9fe6d2", "#9aa9b5", "#2a3540",
    "#f0a44b", "#1d2731", "#2e3b47", "#3cc6a4", "#f2dfb3", "#ffbf73", "#d98b85", "#f7e1df"]
FPS = 9          # 0.1 s of run per frame at 9 fps -> about 0.9x real time
SCALE = 2        # capture at 2x, then downscale: sharper text than a 1x capture
JOBS = [  # name, kind, controller, from s, to s, step s
    ("kitchen_rush", "kitchen", "robust", 36, 82, 0.1),
    ("machine_pipeline", "machine", "naive_slo", 36, 74, 0.1),
]


def build(name, kind, ctrl, t0, t1, step, width=1200, fps=FPS):
    tmp = tempfile.mkdtemp(prefix="xp-frames-")
    try:
        subprocess.run(["node", os.path.join(HERE, "record_explainers.mjs"), "--kind", kind, "--ctrl", ctrl,
                        "--from", str(t0), "--to", str(t1), "--step", str(step), "--scale", str(SCALE), "--out", tmp], check=True)
        files = sorted(glob.glob(os.path.join(tmp, "f*.png")))
        load = lambda f: Image.open(f).convert("RGB").resize((width, round(Image.open(f).height * width / Image.open(f).width)), Image.LANCZOS)
        # one shared palette, built from 8 spread-out frames: stable colours, and the encoder stores only changes
        picks = [load(f) for f in files[:: max(1, len(files) // 8)][:8]]
        sample = Image.new("RGB", (width, picks[0].height * len(picks) + 120))
        for k, im in enumerate(picks):
            sample.paste(im, (0, k * im.height))
        # seed the exact UI colours so greys stay grey and red stays red after quantising
        for k, c in enumerate(UI_COLOURS):
            sample.paste(c, (k * 30 % width, picks[0].height * len(picks) + (k * 30 // width) * 40, k * 30 % width + 30,
                             picks[0].height * len(picks) + (k * 30 // width) * 40 + 40))
        pal = sample.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        frames = [load(f).quantize(palette=pal, dither=Image.Dither.NONE) for f in files]
        hold = [frames[-1]] * fps * 2  # rest on the last frame for 2 s
        path = os.path.join(OUT, name + ".gif")
        frames[0].save(path, save_all=True, append_images=frames[1:] + hold, duration=round(1000 / fps), loop=0, optimize=True)
        print(f"  {path}  {len(frames)} frames  {os.path.getsize(path) // 1024} kB")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for job in JOBS:
        build(*job)
