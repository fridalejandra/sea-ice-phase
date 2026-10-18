#!/usr/bin/env python3
"""
make_composite.py -- combine existing figure PNGs into one multi-panel figure with
panel letters (a), (b), ... Keeps each source figure exactly as rendered.

Usage:
  python make_composite.py --out FigS1_sensitivity_composite.png --ncols 2 \
      panelA.png panelB.png panelC.png [panelD.png ...]
  Append @L or @R to a file name to use only its left or right half
  (e.g. a two-panel figure split into two composite panels).
"""
import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--ncols", type=int, default=2)
    p.add_argument("--width", type=float, default=7.0, help="figure width in inches")
    p.add_argument("--remote", default="gdrive:sea-ice-phase/results/Ch2_Figures")
    a = p.parse_args()

    def load(spec):
        f, _, half = spec.partition("@")
        im = mpimg.imread(f)
        w = im.shape[1]
        if half.upper() == "L":
            im = im[:, : w // 2]
        elif half.upper() == "R":
            im = im[:, w // 2:]
        return im

    imgs = [load(f) for f in a.images]
    n, nc = len(imgs), min(a.ncols, len(imgs))
    nr = -(-n // nc)
    col_w = a.width / nc
    row_h = [max(im.shape[0] / im.shape[1] * col_w for im in imgs[r * nc:(r + 1) * nc]) for r in range(nr)]
    fig = plt.figure(figsize=(a.width, sum(row_h) + 0.1 * nr))
    gs = fig.add_gridspec(nr, nc, height_ratios=row_h, hspace=0.08, wspace=0.04)
    for i, im in enumerate(imgs):
        ax = fig.add_subplot(gs[i // nc, i % nc])
        ax.imshow(im)
        ax.set_axis_off()
        ax.set_anchor("N")
        ax.text(0.0, 1.0, f"({chr(97 + i)})", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=11, fontweight="bold")
    out = Path(a.out)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    ppt = out.parent / "ppt" / out.name
    ppt.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ppt, dpi=300, bbox_inches="tight", transparent=True)
    print(f"Saved: {out}\nSaved (transparent): {ppt}")
    if a.remote:
        subprocess.run(["rclone", "copy", str(out), a.remote])
        subprocess.run(["rclone", "copy", str(ppt), a.remote + "/ppt"])


if __name__ == "__main__":
    main()
