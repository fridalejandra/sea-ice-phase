"""
ch3_plot.py -- shared plotting helpers (layout and annotation).
Visual style, panel letters and saving live in ch3_style.py; save() and
panel_letters() here are kept for older scripts and route through it.
Updated 2026-10-30.
"""
import numpy as np
import matplotlib.pyplot as plt

import ch3_style as S
from ch3_config import (
    SECTORS, SECTOR_LABELS,
    BREAK_YEAR, YEAR_START, YEAR_END, FIGSIZE,
)


def figsize(shape="single"):
    """Standard (width, height) in inches for a named panel layout."""
    if shape not in FIGSIZE:
        print(f"  WARNING: figsize({shape!r}) is not a known shape ({sorted(FIGSIZE)}); using 'single'.")
        shape = "single"
    return FIGSIZE[shape]


def sector_grid(nrow=2, ncol=3, figsize=FIGSIZE["grid2x3"], sharex=True, sharey=False, sectors=None):
    """One panel per sector. Returns (fig, dict[sector -> ax])."""
    sectors = sectors or SECTORS
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, sharex=sharex, sharey=sharey)
    flat = np.atleast_1d(axes).ravel()
    mapping = {}
    for ax, sec in zip(flat, sectors):
        mapping[sec] = ax
        ax.set_title(SECTOR_LABELS[sec], color=S.SECTOR_COLORS.get(sec, S.INK))
    for ax in flat[len(sectors):]:
        ax.set_visible(False)
    return fig, mapping


def mark_break(ax, year=BREAK_YEAR, label=False):
    """Vertical marker at 2016."""
    ax.axvline(year, color=S.GREY, lw=1.0, ls="--", zorder=2)
    if label:
        ax.text(year, ax.get_ylim()[1], f" {year}", color=S.INK, fontsize=8, va="top", ha="left")


def zero_line(ax, **kw):
    ax.axhline(0, color=S.INK, lw=0.7, ls="--", alpha=0.5, zorder=1, **kw)


def sig_band(ax, r_crit, **kw):
    """Shade the non-significant band on a correlation panel."""
    ax.axhspan(-r_crit, r_crit, color=S.GREY2, alpha=0.4, zorder=0, **kw)
    ax.axhline(0, color=S.INK, lw=0.7, zorder=1)


def panel_letters(axes, start=0, **kw):
    """Old API -- routed through ch3_style.label_panels()."""
    S.label_panels(axes, start=start)


def year_axis(ax, start=YEAR_START, end=YEAR_END):
    ax.set_xlim(start - 1, end + 1)
    ax.set_xlabel("Year")


def save(fig, name, dpi=300, sync=False, tight=True, presentation=True):
    """Old API -- routed through ch3_style.save_figure() (house name, both copies)."""
    return S.save_figure(fig, name, dpi=dpi)[0]
