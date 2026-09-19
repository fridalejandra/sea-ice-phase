"""
ch3_plot.py — shared plotting helpers.
======================================
Layout and annotation utilities used across figures. Visual style (fonts,
colours, rcParams) stays in your existing ch3_style.py; this module only
handles repeated *structure*: sector grids, break-year markers, saving.

FIGURE SIZING (added 2026-09): every figure should target ch3_config.TEXT_WIDTH_IN
(6.5in, standard 1in-margin Letter — update ch3_config.py if your program's
margins differ) instead of picking its own figsize. Call figsize(shape) for
the standard tuple, or pass a FIGSIZE key straight to sector_grid(). save()
warns if a figure ends up wider than TEXT_WIDTH_IN anyway, so an old script
that still hardcodes figsize=(14, 10) gets caught at save time rather than
only when it blows up a page in the PDF.
"""

import os
import subprocess 
import numpy as np
import matplotlib.pyplot as plt

from ch3_config import (
    SECTORS, SECTOR_LABELS, SECTOR_COLORS,
    OUTPUT_DIR, GDRIVE, BREAK_YEAR, YEAR_START, YEAR_END,
    TEXT_WIDTH_IN, FIGSIZE,
)


def figsize(shape="single"):
    """Standard (width, height) in inches for a named panel layout.

    shape: one of ch3_config.FIGSIZE's keys ("single", "row2", "row3",
    "stack2", "grid2x2", "grid2x3"). Unknown shapes fall back to "single"
    with a warning rather than silently returning something oversized.
    """
    if shape not in FIGSIZE:
        print(f"  WARNING: figsize({shape!r}) is not a known shape "
              f"({sorted(FIGSIZE)}) — falling back to 'single'.")
        shape = "single"
    return FIGSIZE[shape]


def sector_grid(nrow=2, ncol=3, figsize=FIGSIZE["grid2x3"], sharex=True, sharey=False,
                sectors=None):
    """Standard one-panel-per-sector grid. Returns (fig, dict[sector -> ax])."""
    sectors = sectors or SECTORS
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize,
                             sharex=sharex, sharey=sharey)
    flat = np.atleast_1d(axes).ravel()
    mapping = {}
    for ax, sec in zip(flat, sectors):
        mapping[sec] = ax
        ax.set_title(SECTOR_LABELS[sec], fontsize=11, fontweight="bold",
                     color=SECTOR_COLORS[sec])
        ax.spines[["top", "right"]].set_visible(False)
    for ax in flat[len(sectors):]:
        ax.set_visible(False)
    return fig, mapping


def mark_break(ax, year=BREAK_YEAR, label=False):
    """Vertical marker at the regime shift."""
    ax.axvline(year, color="#D4537E", lw=1.4, ls="--", alpha=0.9, zorder=2)
    if label:
        ax.text(year, ax.get_ylim()[1], f" {year}", color="#D4537E",
                fontsize=8, va="top", ha="left")


def zero_line(ax, **kw):
    ax.axhline(0, color="#2C2C2A", lw=0.8, ls="--", alpha=0.5, zorder=1, **kw)


def sig_band(ax, r_crit, **kw):
    """Shade the non-significant band on a correlation panel."""
    ax.axhspan(-r_crit, r_crit, color="#888888", alpha=0.10, zorder=0, **kw)
    ax.axhline(0, color="#2C2C2A", lw=0.8, zorder=1)


def panel_letters(axes, start=0, x=0.03, y=0.97, **kw):
    """Add (a), (b), (c)... to a sequence of axes."""
    import string
    letters = string.ascii_lowercase
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        if not ax.get_visible():
            continue
        ax.text(x, y, f"({letters[start + i]})", transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top", ha="left",
                color="#2C2C2A", **kw)


def year_axis(ax, start=YEAR_START, end=YEAR_END):
    ax.set_xlim(start - 1, end + 1)
    ax.set_xlabel("Year", fontsize=10)


def save(fig, name, dpi=300, sync=True, tight=True):
    """Save to OUTPUT_DIR and optionally rclone to Drive.

    Warns (does not block) if the figure is wider than ch3_config.TEXT_WIDTH_IN
    — catches a script that still hardcodes an oversized figsize= instead of
    using ch3_plot.figsize()/sector_grid()'s default.
    """
    w_in, h_in = fig.get_size_inches()
    if w_in > TEXT_WIDTH_IN + 0.1:
        print(f"  WARNING: {name} is {w_in:.1f}in wide, target is "
              f"{TEXT_WIDTH_IN}in — it will run past the text margin or get "
              f"auto-shrunk (and its fonts with it) when placed in the document.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    if tight:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    else:
        fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Saved → {path}")

    if sync:
        try:
            r = subprocess.run(["rclone", "copy", path, GDRIVE],
                               capture_output=True, text=True)
            print(f"Synced → {GDRIVE}" if r.returncode == 0
                  else f"rclone failed: {r.stderr.strip()}")
        except FileNotFoundError:
            print("rclone not found — skipping sync.")
    return path