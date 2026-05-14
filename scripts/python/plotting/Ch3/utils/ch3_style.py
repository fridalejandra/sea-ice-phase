"""
Shared colors, style settings, and small helper functions used across all
Chapter 3 figures. Import what you need at the top of each figure script —
there's a usage example at the bottom of this file.

The goal is that every Ch3 figures looks consistent
without having to copy-paste rcParams and colors dicts everywhere.
"""

import os
import subprocess
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


# --- Style -----------------------------------------------------------------

def apply_style():
    # Call this once at the top of any figure script before plotting.
    # Nimbus Sans keeps things clean; top/right spines off by default.
    plt.rcParams.update({
        "font.family"      : "Nimbus Sans",
        "font.size"        : 11,
        "axes.spines.top"  : False,
        "axes.spines.right": False,
        "axes.linewidth"   : 0.8,
        "axes.labelsize"   : 12,
        "axes.titlesize"   : 13,
        "axes.titleweight" : "bold",
        "xtick.labelsize"  : 10,
        "ytick.labelsize"  : 10,
        "legend.fontsize"  : 10,
        "legend.frameon"   : False,
        "figure.dpi"       : 150,
        "savefig.dpi"      : 300,
        "savefig.bbox"     : "tight",
        "savefig.facecolor": "white",
    })


# --- Sector colours and labels ---------------------------------------------
# These are used in almost every figure so keeping them here avoids drift.
# SECTORS_NO_CIRC is handy for the 5-panel small-multiples layouts, no circumpolar;
# use SECTORS when you want circumpolar included.

SECTOR_COLORS = {
    "SIE_Weddell"                : "#2196F3",   # blue
    "SIE_Amundsen_Bellingshausen": "#F44336",   # red
    "SIE_Ross"                   : "#4CAF50",   # green
    "SIE_East_Antarctica"        : "#FF9800",   # orange
    "SIE_King_Haakon"            : "#9C27B0",   # purple
    "SIE_circumpolar"            : "#2C2C2A",   # near-black
}

SECTOR_LABELS = {
    "SIE_Weddell"                : "Weddell",
    "SIE_Amundsen_Bellingshausen": "ABS",
    "SIE_Ross"                   : "Ross",
    "SIE_East_Antarctica"        : "East Antarctica",
    "SIE_King_Haakon"            : "King Haakon",
    "SIE_circumpolar"            : "Circumpolar",
}

SECTORS         = list(SECTOR_COLORS.keys())
SECTORS_NO_CIRC = [s for s in SECTORS if s != "SIE_circumpolar"]

# Pre-built ordered lists — saves writing list comprehensions in every script
SECTOR_COLOR_LIST          = [SECTOR_COLORS[s] for s in SECTORS]
SECTOR_LABEL_LIST          = [SECTOR_LABELS[s] for s in SECTORS]
SECTOR_COLOR_LIST_NO_CIRC  = [SECTOR_COLORS[s] for s in SECTORS_NO_CIRC]
SECTOR_LABEL_LIST_NO_CIRC  = [SECTOR_LABELS[s] for s in SECTORS_NO_CIRC]


# --- Decade colour scheme --------------------------------------------------
# Used for scatter points in timeseries figures to show temporal context
# without cluttering the axes. The 2016+ era gets its own color because
# it's the key break point in most of our analysis.

def decade_color(year):
    if year < 1990:   return "#888780"   # grey  — 1980s
    elif year < 2000: return "#378ADD"   # blue  — 1990s
    elif year < 2010: return "#1D9E75"   # green — 2000s
    elif year < 2016: return "#BA7517"   # amber — 2010–2015
    else:             return "#D4537E"   # pink  — 2016+

DECADE_LEGEND = [
    ("#888780", "1980s"),
    ("#378ADD", "1990s"),
    ("#1D9E75", "2000s"),
    ("#BA7517", "2010–2015"),
    ("#D4537E", "2016+"),
]


# --- Axes helpers ----------------------------------------------------------
# Small functions that get called repeatedly across figures.
# All take an axes object as the first argument.

def zero_line(ax, **kwargs):
    # Dashed grey line at y=0. kwargs forwarded to axhline if you need tweaks.
    defaults = dict(color="grey", lw=0.7, ls="--", zorder=0)
    defaults.update(kwargs)
    ax.axhline(0, **defaults)


def shade2016(ax, yr_max=None, color="grey", alpha=0.07):
    # Light grey shading over the post-2016 period.
    # yr_max defaults to the current x-axis maximum if not given.
    if yr_max is None:
        yr_max = ax.get_xlim()[1]
    ax.axvspan(2016.5, yr_max + 0.5, color=color, alpha=alpha, zorder=0)


def vline2016(ax, **kwargs):
    # Pink dashed vertical line marking the 2016 breakpoint.
    defaults = dict(color="#D4537E", lw=1.2, ls="--", alpha=0.7, zorder=2)
    defaults.update(kwargs)
    ax.axvline(2016, **defaults)


def stroke(lw=2, foreground="white"):
    # White halo around text — keeps annotations readable over busy backgrounds.
    # Usage: ax.text(..., path_effects=stroke())
    return [pe.withStroke(linewidth=lw, foreground=foreground)]


def sigma_lines(ax, levels=(1,), **kwargs):
    # Horizontal ±N sigma reference lines, e.g. sigma_lines(ax, levels=(1, 2)).
    defaults = dict(color="#2C2C2A", lw=0.5, ls="--", alpha=0.3, zorder=2)
    defaults.update(kwargs)
    for s in levels:
        ax.axhline( s, **defaults)
        ax.axhline(-s, **defaults)


# --- Saving ----------------------------------------------------------------
# Change DEFAULT_OUTPUT_DIR if your results folder is somewhere else.
# gdrive_sync=True will rclone the file to GDrive after saving — useful for
# sharing with ppl or using cluster but slow, so leave it off during iteration.

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.expanduser("~"),
    "Research", "repos", "sea-ice-phase", "scripts", "python", "plotting", "Ch3", "figures"
)

GDRIVE_DEST = "gdrive:MyDrive/sea-ice-phase/results/Ch3_Figures/"


def save_fig(fig, name, output_dir=None, gdrive_sync=True):
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    fig.savefig(path)
    print(f"  -> saved: {name}")
    plt.close(fig)

    if gdrive_sync:
        result = subprocess.run(
            ["rclone", "copy", path, GDRIVE_DEST],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  -> synced to GDrive: {name}")
        else:
            print(f"  -> GDrive sync failed: {result.stderr.strip()}")


# --- Usage example ---------------------------------------------------------
# from ch3_style import (
#     apply_style,
#     SECTORS, SECTORS_NO_CIRC,
#     SECTOR_COLORS, SECTOR_LABELS,
#     SECTOR_COLOR_LIST_NO_CIRC, SECTOR_LABEL_LIST_NO_CIRC,
#     DECADE_LEGEND, decade_color,
#     zero_line, shade2016, vline2016, stroke, sigma_lines,
#     save_fig, DEFAULT_OUTPUT_DIR,
# )
# apply_style()