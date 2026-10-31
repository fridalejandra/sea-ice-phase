"""
ch3_style.py -- the single source of visual style for every Chapter 3 figure.
Rebuilt 2026-10-30 on top of the 2026-09 version (fonts, real-bold workaround,
dual export are kept; palette, ink, naming and folders are new).

HOUSE RULES
  * seaborn, "colorblind" palette (sns.color_palette("colorblind")).
  * Axis labels, tick labels and titles: BOLD, dark grey (INK = #333333).
  * Spines (left/bottom only) and ticks in the same dark grey.
  * Multi-panel figures are lettered (a), (b), ... with label_panels(),
    except the components-by-cycle figure.
  * Width: 6.5 in (text width).
  * Every figure is written twice, under one name:
        results/ch3/figures/manuscript/<name>.png   white background (Overleaf)
        results/ch3/figures/slides/<name>.png       transparent background
  * Names: ch3_figNN_short_name (main text), ch3_figA1_short_name (appendix).

USAGE in a figure script
    import ch3_style as S
    fig, axes = plt.subplots(1, 2, figsize=(S.TEXT_WIDTH_IN, 3))
    ... plot with S.BLUE, S.ORANGE, S.SECTOR_PALETTE[...] ...
    S.label_panels(axes)
    S.save_figure(fig, "ch3_fig05_mean_amp_coupling")

Old scripts that still call ch3_plot.save(fig, "fig03_components_by_cycle.png")
or ch3_style.savefig_pair(fig, ".../fig_level_leak") keep working: both now
route through save_figure(), and NAME_MAP below renames the old file names
to the ch3_figNN_* convention.

BOLD TEXT: fontweight="bold" does not render with Helvetica.ttc on macOS
(matplotlib picks the first sub-face of a .ttc). save_figure() therefore
runs embolden(fig), which sets every axis label, tick label and title to the
real Bold face extracted by bold_font_properties(). Scripts do not need to
set bold themselves.
"""
import os
import string
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# ── names: old output file -> house name ───────────────────────────────────
NAME_MAP = {
    "fig1_cycle_mechanisms":          "ch3_fig01_cycle_schematic",
    "fig02_sector_map":               "ch3_fig02_sector_map",
    "fig_decomposition_2022":         "ch3_fig03_decomposition_2022",
    "fig03_components_by_cycle":      "ch3_fig04_components_by_cycle",
    "fig_mean_amp_coupling":          "ch3_fig05_mean_amp_coupling",
    "fig_ch3_sst_partition":          "ch3_fig06_sst_components",
    "fig_level_leak":                 "ch3_figA1_mean_leak",
}

# ── ink, font, layout ──────────────────────────────────────────────────────
INK = "#333333"
GRID = "0.88"
TEXT_WIDTH_IN = 6.5
FONT_PREFERENCE = ["Helvetica", "Arial", "DejaVu Sans"]

sns.set_theme(style="ticks", palette="colorblind")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONT_PREFERENCE,
    "font.size": 9,
    "font.weight": "normal",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": INK,
    "axes.linewidth": 1.0,
    "axes.labelcolor": INK,
    "axes.labelsize": 9,
    "axes.labelweight": "bold",
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.titlepad": 4,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "lines.linewidth": 1.8,
    "lines.solid_capstyle": "round",
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
})

# ── colours: seaborn colorblind ────────────────────────────────────────────
CB = sns.color_palette("colorblind").as_hex()   # 10 colours
BLUE, ORANGE, GREEN, VERMILLION, PURPLE, BROWN, PINK, CBGREY, YELLOW, SKY = CB
OR = ORANGE                       # alias used by older scripts
CBLIND_COLORS = CB                # alias used by older scripts
GREY = "0.55"                     # reference / invariant lines
GREY2 = "0.82"                    # backdrop
SECTOR_PALETTE = {
    "weddell": BLUE, "amundsen_bellingshausen": VERMILLION, "ross": GREEN,
    "east_antarctica": ORANGE, "king_haakon": PURPLE, "circumpolar": INK,
}
SECTOR_COLORS = {                 # keyed like the data files
    "SIE_Weddell": BLUE, "SIE_Amundsen_Bellingshausen": VERMILLION, "SIE_Ross": GREEN,
    "SIE_East_Antarctica": ORANGE, "SIE_King_Haakon": PURPLE, "SIE_circumpolar": INK,
}
SECTOR_NAMES = {
    "SIE_Weddell": "Weddell", "SIE_Amundsen_Bellingshausen": "Amundsen–Bellingshausen",
    "SIE_Ross": "Ross", "SIE_East_Antarctica": "East Antarctica",
    "SIE_King_Haakon": "King Haakon VII", "SIE_circumpolar": "Circumpolar",
}
COMPONENT_COLORS = {"Trend": GREEN, "Amplitude": ORANGE, "Phase": PINK, "Raw": CBGREY}
POS, NEG = VERMILLION, BLUE       # positive / negative bars
print("  [ch3_style] seaborn colorblind palette, ink", INK)


# ── font check and real bold ───────────────────────────────────────────────
def _resolved_font():
    path = fm.findfont(fm.FontProperties(), fallback_to_default=True)
    return fm.FontProperties(fname=path).get_name(), path


_name, _path = _resolved_font()
print(f"  [ch3_style] font: {_name}")

_BOLD_DIR = os.path.join(tempfile.gettempdir(), "ch3_style_bold_faces")
_bold_cache = {}


def _extract_bold_face(source_path):
    """Pull the Bold sub-face out of a .ttc into a standalone .ttf."""
    if source_path in _bold_cache:
        return _bold_cache[source_path]
    out = None
    if source_path.lower().endswith(".ttc"):
        try:
            from fontTools.ttLib import TTCollection
            for i, font in enumerate(TTCollection(source_path).fonts):
                names = [font["name"].getDebugName(n) for n in (17, 2, 1)]
                names = [n.lower() for n in names if n]
                if any("bold" in n for n in names) and not any(("oblique" in n or "italic" in n) for n in names):
                    os.makedirs(_BOLD_DIR, exist_ok=True)
                    out = os.path.join(_BOLD_DIR, f"{os.path.splitext(os.path.basename(source_path))[0]}_bold{i}.ttf")
                    if not os.path.exists(out):
                        font.save(out)
                    break
        except Exception as e:
            print(f"  WARNING [ch3_style]: could not read {source_path} ({e}); bold falls back to weight='bold'.")
    _bold_cache[source_path] = out
    return out


def bold_font_properties(size=None):
    """FontProperties that renders bold even for Helvetica.ttc."""
    base = fm.findfont(fm.FontProperties(), fallback_to_default=True)
    bold = _extract_bold_face(base)
    props = fm.FontProperties(fname=bold) if bold else fm.FontProperties(weight="bold")
    props.set_size(size if size is not None else plt.rcParams["font.size"])
    return props


def _bolden(text_obj):
    if text_obj is None or not text_obj.get_text():
        return
    color = text_obj.get_color()
    text_obj.set_fontproperties(bold_font_properties(text_obj.get_fontsize()))
    text_obj.set_color(color)


def embolden(fig):
    """Axis labels, tick labels and titles -> real bold, colour preserved."""
    for ax in fig.get_axes():
        _bolden(ax.xaxis.label)
        _bolden(ax.yaxis.label)
        for t in (ax.title, ax._left_title, ax._right_title):
            _bolden(t)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            _bolden(t)
    if fig._suptitle is not None:
        _bolden(fig._suptitle)


# ── panel letters ──────────────────────────────────────────────────────────
def label_panels(axes, start=0, x=-0.02, y=1.02, size=10):
    """Letter each visible panel (a), (b), ... in bold dark grey.
    A panel with a title gets the letter in front of it, left-aligned: "(a)  Weddell".
    A panel without a title gets the letter just above its top-left corner."""
    for i, ax in enumerate([a for a in np.atleast_1d(axes).ravel() if a.get_visible()]):
        letter = f"({string.ascii_lowercase[start + i]})"
        tobj = ax.title if ax.get_title("center") else ax._left_title
        title = tobj.get_text()
        if title:
            fs, col = tobj.get_fontsize(), tobj.get_color()
            ax.set_title("")
            ax.set_title(f"{letter}  {title}", loc="left", fontsize=fs, color=col)
        else:
            ax.text(x, y, letter, transform=ax.transAxes, fontproperties=bold_font_properties(size),
                    color=INK, va="bottom", ha="right")


# ── saving ─────────────────────────────────────────────────────────────────
def _fig_dirs():
    from ch3_config import ROOT          # imported here to avoid a circular import
    base = os.path.join(ROOT, "results", "ch3", "figures")
    ms, sl = os.path.join(base, "manuscript"), os.path.join(base, "slides")
    os.makedirs(ms, exist_ok=True); os.makedirs(sl, exist_ok=True)
    return ms, sl


def house_name(name):
    stem = os.path.splitext(os.path.basename(name))[0]
    if stem.endswith("_presentation"):
        stem = stem[: -len("_presentation")]
    return NAME_MAP.get(stem, stem)


def save_figure(fig, name, dpi=300, close=True):
    """Write manuscript (white) and slides (transparent) copies under the house name."""
    name = house_name(name)
    if not name.startswith("ch3_fig"):
        print(f"  WARNING [ch3_style]: '{name}' is not a ch3_figNN_* name -- add it to NAME_MAP.")
    w = fig.get_size_inches()[0]
    if w > TEXT_WIDTH_IN + 0.1:
        print(f"  WARNING [ch3_style]: {name} is {w:.1f} in wide; text width is {TEXT_WIDTH_IN} in.")
    embolden(fig)
    ms, sl = _fig_dirs()
    p_ms, p_sl = os.path.join(ms, name + ".png"), os.path.join(sl, name + ".png")
    fig.savefig(p_ms, dpi=dpi, bbox_inches="tight", facecolor="white", transparent=False)
    fig.savefig(p_sl, dpi=dpi, bbox_inches="tight", transparent=True)
    print(f"  [ch3_style] wrote {p_ms}\n  [ch3_style] wrote {p_sl}")
    if close:
        plt.close(fig)
    return p_ms, p_sl


def savefig_pair(fig, path_no_ext, dpi=300, **kwargs):
    """Old API -- routed through save_figure()."""
    return save_figure(fig, path_no_ext, dpi=dpi, close=False)
