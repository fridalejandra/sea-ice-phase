"""
ch3_style.py -- the single source of visual style (fonts, colours, rcParams)
for every Ch3 figure. ch3_plot.py's own docstring already says style lives
here; this file didn't actually exist yet (confirmed 2026-09-18) -- that's
the real reason fonts weren't consistent: some figure scripts set their own
plt.rcParams.update({"font.size": 10, ...}) inline with no font.family at
all, others set nothing and just got matplotlib's default (DejaVu Sans).

Usage: add `import ch3_style` near the top of any figure script, right
after the ch3_config import (so it benefits from the same sys.path insert)
-- importing this module applies the style immediately, at import time, and
prints which font actually got resolved. If a figure script has its own
plt.rcParams.update({...}) call for font/spines, remove it -- this module
is now the one place that sets those, and a script-local override placed
after `import ch3_style` would silently win and put that one figure back
out of sync with the rest.

BOLD TEXT (added 2026-09-19): fontweight="bold" alone does NOT work with
Helvetica on macOS. Confirmed: /System/Library/Fonts/Helvetica.ttc is a
TrueType COLLECTION -- Regular, Bold, Oblique, and Bold Oblique bundled as
sub-faces of one file. matplotlib's font_manager indexes a .ttc by family
NAME only; fm.findfont(FontProperties(family="Helvetica", weight="bold"))
and the weight="normal" version both resolve to the exact same path, and
matplotlib silently renders whichever sub-face comes first in the
collection (regular) -- no error, no warning, it just isn't bold. This is
a matplotlib/.ttc limitation, not something wrong with a figure script.

Use `ch3_style.bold_font_properties()` wherever a figure needs real bold
text (sector names, axis labels, etc.) instead of fontweight="bold" -- it
extracts the true Bold sub-face out of the .ttc once (via fontTools) into
a standalone .ttf and returns a FontProperties pointing directly at that
file, which sidesteps the broken weight lookup entirely. If the resolved
font isn't a .ttc, or no distinct Bold sub-face can be found inside it,
it prints a warning and falls back to returning a plain bold
FontProperties (same behaviour as before -- may or may not render bold
depending on the font).
"""
import os
import tempfile
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Tries Helvetica first, then Tacoma, then Tahoma (in case "Tacoma" was a
# typo for the similarly-named, far more common system font -- ask Frida
# to confirm which one she actually wants if the resolved font matters),
# then two safe fallbacks so figures never silently revert to an ugly
# default even if none of the above are installed.
FONT_PREFERENCE = ["Helvetica", "Tacoma", "Tahoma", "Arial", "DejaVu Sans"]

# House style (2026-09-19): minimal, direct-labelled where possible, data in
# colour and everything else in grey ink. Lines are the data; spines, ticks
# and labels recede. Sector titles are set bold per figure with
# bold_font_properties() because fontweight="bold" doesn't render with
# Helvetica.ttc (see below). Prefer direct labels on the first panel to a
# legend; when a legend is unavoidable it is frameless.
INK = "0.35"          # axis lines, ticks, tick labels, axis labels
GRID = "0.85"         # only if a figure turns grids on; off by default
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
    """What font matplotlib will ACTUALLY use, not just what was requested --
    findfont() silently falls back to a default if nothing in
    FONT_PREFERENCE is installed, so this checks the real answer rather
    than trusting the rcParams just set above."""
    # No explicit family= here: bare FontProperties() reads the current
    # rcParams (font.family/font.sans-serif) that were just set above --
    # passing family="sans-serif" directly gets misparsed by matplotlib as
    # a fontconfig pattern string in newer versions and raises, rather than
    # being treated as the generic family name.
    path = fm.findfont(fm.FontProperties(), fallback_to_default=True)
    return fm.FontProperties(fname=path).get_name(), path


_name, _path = _resolved_font_name()
if _name.lower() not in ("helvetica", "tacoma"):
    print(f"  WARNING [ch3_style]: neither Helvetica nor Tacoma is available -- "
          f"matplotlib resolved to '{_name}' instead ({_path}). Figures will "
          f"render in this font until Helvetica or Tacoma is installed and "
          f"matplotlib's font cache is rebuilt (delete "
          f"~/.cache/matplotlib and re-run, or "
          f"matplotlib.font_manager._load_fontmanager(try_read_cache=False)).")
else:
    print(f"  [ch3_style] using font: {_name}")

# ── real bold, working around the .ttc weight-selection limitation ─────────
_BOLD_CACHE_DIR = os.path.join(tempfile.gettempdir(), "ch3_style_bold_faces")
_bold_face_path_cache = {}  # source path -> extracted bold .ttf path (or None)


def _extract_bold_face(source_path):
    """If source_path is a .ttc collection, find the sub-face whose name
    says Bold (and not Oblique/Italic) and save it out as a standalone
    .ttf. Returns the extracted path, or None if source_path isn't a
    collection or no clean Bold sub-face is found inside it."""
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
                is_oblique_or_italic = any(("oblique" in n.lower() or "italic" in n.lower())
                                           for n in names)
                if is_bold and not is_oblique_or_italic:
                    os.makedirs(_BOLD_CACHE_DIR, exist_ok=True)
                    out_path = os.path.join(
                        _BOLD_CACHE_DIR,
                        f"{os.path.splitext(os.path.basename(source_path))[0]}_bold{i}.ttf")
                    if not os.path.exists(out_path):
                        font.save(out_path)
                    result = out_path
                    break
        except Exception as e:
            print(f"  WARNING [ch3_style]: couldn't inspect {source_path} as a "
                  f"font collection ({e}) -- bold text will fall back to "
                  f"fontweight='bold', which may not render bold for this font.")

    _bold_face_path_cache[source_path] = result
    return result


def bold_font_properties(size=None):
    """FontProperties that reliably renders BOLD, even when the resolved
    font is a .ttc collection matplotlib can't weight-select within (see
    module docstring). Pass this as `fontproperties=...` to set_xlabel,
    set_yticklabels, set_title, Text.set_fontproperties, etc., instead of
    fontweight="bold".

    size: point size to set on the returned FontProperties. If omitted,
    uses the current rcParams["font.size"] so bold text matches whatever
    size the rest of the figure is using by default.
    """
    base_path = fm.findfont(fm.FontProperties(), fallback_to_default=True)
    bold_path = _extract_bold_face(base_path)
    if bold_path is not None:
        props = fm.FontProperties(fname=bold_path)
    else:
        # Not a .ttc (or no clean Bold sub-face inside one) -- ordinary
        # weight-based selection already works for separate-file fonts, so
        # this isn't the broken case and doesn't need a warning.
        if base_path.lower().endswith(".ttc"):
            print(f"  WARNING [ch3_style]: {base_path} is a font collection but "
                  f"no distinct Bold sub-face was found inside it -- falling "
                  f"back to fontweight='bold', which may render identically "
                  f"to regular weight for this font.")
        props = fm.FontProperties(weight="bold")
    props.set_size(size if size is not None else plt.rcParams["font.size"])
    return props