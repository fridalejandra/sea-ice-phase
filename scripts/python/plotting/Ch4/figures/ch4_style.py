"""
ch4_style.py -- the single source of visual style (fonts, colours, rcParams)
for every Ch4 figure. A copy of ch3_style.py (2026-09-19) so Ch4 figures match
the other chapters; see ch3_style.py for the full notes on the Helvetica .ttc
bold-face workaround.

Usage: `import ch4_style` near the top of a figure script (applies the style at
import time). Do not set font/spine rcParams inside figure scripts. Use
`ch4_style.bold_font_properties()` instead of fontweight="bold".

Sector colours match the SCAR poster / fig_sector_map_poster.py.
"""
import os
import tempfile
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

FONT_PREFERENCE = ["Helvetica", "Tacoma", "Tahoma", "Arial", "DejaVu Sans"]

INK = "0.35"          # axis lines, ticks, tick labels, axis labels
GRID = "0.85"         # only if a figure turns grids on; off by default
SECTOR_COLORS = {"WS": "#F44336", "KH": "#FFC107", "EA": "#FF9800",
                 "RA": "#4CAF50", "ABS": "#2196F3"}
SECTOR_NAMES = {"WS": "Weddell", "KH": "King Haakon VII", "EA": "East Antarctica",
                "RA": "Ross–Amundsen", "ABS": "Amundsen–Bellingshausen"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONT_PREFERENCE,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "axes.labelsize": 9,
    "axes.titlesize": 10.5,
    "axes.titlepad": 4,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "legend.frameon": False,
    "legend.fontsize": 9,
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.facecolor": "white",
})


def _resolved_font_name():
    path = fm.findfont(fm.FontProperties(), fallback_to_default=True)
    return fm.FontProperties(fname=path).get_name(), path


_name, _path = _resolved_font_name()
if _name.lower() not in ("helvetica", "tacoma"):
    print(f"  WARNING [ch4_style]: neither Helvetica nor Tacoma is available -- "
          f"matplotlib resolved to '{_name}' instead ({_path}).")
else:
    print(f"  [ch4_style] using font: {_name}")

_BOLD_CACHE_DIR = os.path.join(tempfile.gettempdir(), "ch4_style_bold_faces")
_bold_face_path_cache = {}


def _extract_bold_face(source_path):
    if source_path in _bold_face_path_cache:
        return _bold_face_path_cache[source_path]
    result = None
    if source_path.lower().endswith(".ttc"):
        try:
            from fontTools.ttLib import TTCollection
            coll = TTCollection(source_path)
            for i, font in enumerate(coll.fonts):
                try:
                    name_table = font["name"]
                    names = [name_table.getDebugName(nid) for nid in (17, 2, 1)]
                    names = [n for n in names if n]
                except Exception:
                    continue
                is_bold = any("bold" in n.lower() for n in names)
                is_obl = any(("oblique" in n.lower() or "italic" in n.lower()) for n in names)
                if is_bold and not is_obl:
                    os.makedirs(_BOLD_CACHE_DIR, exist_ok=True)
                    out_path = os.path.join(
                        _BOLD_CACHE_DIR,
                        f"{os.path.splitext(os.path.basename(source_path))[0]}_bold{i}.ttf")
                    if not os.path.exists(out_path):
                        font.save(out_path)
                    result = out_path
                    break
        except Exception as e:
            print(f"  WARNING [ch4_style]: couldn't inspect {source_path} ({e}); "
                  f"falling back to fontweight='bold'.")
    _bold_face_path_cache[source_path] = result
    return result


def bold_font_properties(size=None):
    base_path = fm.findfont(fm.FontProperties(), fallback_to_default=True)
    bold_path = _extract_bold_face(base_path)
    props = fm.FontProperties(fname=bold_path) if bold_path else fm.FontProperties(weight="bold")
    props.set_size(size if size is not None else plt.rcParams["font.size"])
    return props
