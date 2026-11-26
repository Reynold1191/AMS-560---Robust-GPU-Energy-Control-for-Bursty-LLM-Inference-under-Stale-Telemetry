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
JOBS = [  # name, kind, controller, from s, to s, step s
    ("kitchen_rush", "kitchen", "robust", 36, 82, 0.4),
    ("machine_pipeline", "machine", "naive_slo", 36, 74, 0.35),
]


def build(name, kind, ctrl, t0, t1, step, width=880, fps=9):
    tmp = tempfile.mkdtemp(prefix="xp-frames-")
    try:
        subprocess.run(["node", os.path.join(HERE, "record_explainers.mjs"), "--kind", kind, "--ctrl", ctrl,
                        "--from", str(t0), "--to", str(t1), "--step", str(step), "--out", tmp], check=True)
        rgb = []
        for f in sorted(glob.glob(os.path.join(tmp, "f*.png"))):
            im = Image.open(f).convert("RGB")
            rgb.append(im.resize((width, round(im.height * width / im.width)), Image.LANCZOS))
        # one shared palette (built from a few frames) keeps colours stable and lets the encoder store only changes
        sample = Image.new("RGB", (width, rgb[0].height * 8))
        for k, im in enumerate(rgb[:: max(1, len(rgb) // 8)][:8]):
            sample.paste(im, (0, k * im.height))
        pal = sample.quantize(colors=200, method=Image.Quantize.MAXCOVERAGE)
        frames = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in rgb]
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
