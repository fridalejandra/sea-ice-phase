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


# ── colour palette (added 2026-09-30) ───────────────────────────────────────
# House rule going forward: every Ch3 figure gets its categorical colours
# from HERE, not from hex codes typed inline in the figure script. This was
# prompted by Frida asking specifically for the `cblind` package (colorblind-
# -safe palettes for matplotlib) after the ad hoc hex colours used in the
# intro/methods schematic figures (fig1_cycle_mechanisms.py,
# fig_phase_adjustment.py, level_leak.py -- "#2a78d6" / "#eb6834" / "0.55")
# were never checked for colorblind-safety.
#
# `cblind` itself could NOT be installed in the Claude sandbox this was
# written in -- `pip install cblind` fails because this sandbox's network
# egress policy does not allow reaching pypi.org (confirmed: pip fails the
# same way for every uninstalled package, e.g. cmcrameri, pytest-xdist --
# this is a sandbox/org policy restriction, not evidence that "cblind" is
# the wrong package name or doesn't exist). So this module tries `import
# cblind` first and uses its categorical colours when it's available (e.g.
# on Frida's own machine, once she `pip install cblind` there); when it
# isn't available, it falls back to the Okabe & Ito (2002) colorblind-safe
# palette -- a specific, published, widely-cited 8-colour set (the same one
# cblind's own categorical palette is built on), not a guess at cblind's
# internal hex values. Either branch gives colorblind-safe colours; the
# WARNING below just tells you honestly which source supplied them, so it's
# never a silent substitution.
#
# Names map onto the four slots the existing schematic scripts already used
# ad hoc (BLUE/OR/GREY/GREY2), so retrofitting a script is:
#   import ch3_style
#   BLUE, ORANGE, GREY, GREY2 = ch3_style.BLUE, ch3_style.ORANGE, ch3_style.GREY, ch3_style.GREY2
# rather than a rewrite of every plot call.
_OKABE_ITO = {
    "black":        "#000000",
    "orange":       "#E69F00",
    "sky_blue":     "#56B4E9",
    "bluish_green": "#009E73",
    "yellow":       "#F0E442",
    "blue":         "#0072B2",
    "vermillion":   "#D55E00",
    "reddish_purple": "#CC79A7",
}

try:
    import cblind as _cb
    _PALETTE_SOURCE = "cblind"
    # cblind.Colorplots exposes a categorical colour list; guard the exact
    # call since cblind's API differs across versions -- fall back to the
    # Okabe-Ito set (which cblind itself is built on) if the call shape here
    # doesn't match whatever version is installed, rather than crashing every
    # figure script on an API mismatch.
    #
    # Fixed 2026-10-04: Colorplots().cblind(n) returns a (colors, linestyle)
    # PAIR, not a flat colour list (confirmed from cblind's own PyPI usage
    # examples: "color, linestyle = cb.Colorplots().function_name(...)").
    # The old code did list(cblind(8)) directly, which wrapped that 2-tuple
    # into a 2-element list -- CBLIND_COLORS[2] and up then raised
    # IndexError, uncaught because it happened outside this try block. Now
    # unpacks explicitly and validates the length before trusting it.
    try:
        _result = _cb.Colorplots().cblind(8)
        _colors = _result[0] if isinstance(_result, tuple) and len(_result) == 2 else _result
        CBLIND_COLORS = list(_colors)
        if len(CBLIND_COLORS) < 8:
            raise ValueError(f"expected >= 8 colours back, got {len(CBLIND_COLORS)}")
    except Exception as _e:
        CBLIND_COLORS = list(_OKABE_ITO.values())
        _PALETTE_SOURCE = f"okabe_ito (cblind installed but its call shape didn't match -- {_e}; used its own base palette)"
except ImportError:
    CBLIND_COLORS = list(_OKABE_ITO.values())
    _PALETTE_SOURCE = "okabe_ito (cblind not installed in this environment)"

print(f"  [ch3_style] colour palette source: {_PALETTE_SOURCE}")

# Fixed 2026-10-04: these five NAMED colours now always come directly from
# the Okabe-Ito hex table, regardless of _PALETTE_SOURCE. The old code
# indexed them out of whatever list cblind's own cblind(8) call returned,
# on the unverified assumption that cblind's own categorical order matches
# Okabe-Ito's position-for-position (index 0 = blue, index 1 = orange,
# etc.) -- confirmed wrong on Frida's machine once cblind was actually
# installed: its palette doesn't follow that order, so BLUE and ORANGE both
# came out as different shades of blue (Fig. 6, 2026-10-04). Okabe-Ito is a
# specific, published, named palette -- using it directly for colours that
# are *named* (BLUE must actually be blue) is correct regardless of which
# library is installed. CBLIND_COLORS above stays as the generic ordered
# list, for a plot that just needs N distinct colorblind-safe categories
# without caring which hue lands in which slot.
BLUE   = _OKABE_ITO["blue"]
ORANGE = _OKABE_ITO["orange"]
OR     = ORANGE  # alias -- matches the variable name the existing scripts already used
GREEN  = _OKABE_ITO["bluish_green"]
PURPLE = _OKABE_ITO["reddish_purple"]
VERMILLION = _OKABE_ITO["vermillion"]
# Six-colour sector palette: five categorical hues (kept visually distinct
# from each other and colorblind-safe either way) plus a neutral dark ink
# for the circumpolar total, which is a sum of the other five, not a sixth
# category -- used by ch3_config.SECTOR_COLORS and anything else that wants
# the same six-way split.
SECTOR_PALETTE = {
    "weddell": BLUE, "amundsen_bellingshausen": VERMILLION, "ross": GREEN,
    "east_antarctica": OR, "king_haakon": PURPLE, "circumpolar": "0.15",
}
GREY   = "0.55"   # ink grey for reference/"ordinary" lines -- not part of the
GREY2  = "0.82"   # categorical palette, kept as neutral grey on purpose so a
                   # "baseline" or "invariant" line reads as backdrop, not data


# ── dual-background export (added 2026-09-30) ──────────────────────────────
# Frida's standing request: every figure gets exported twice -- once for the
# manuscript (opaque white background, the LaTeX-ready version) and once for
# slides (transparent background, so it drops onto any slide-deck colour).
# Use this instead of a bare fig.savefig(...) call so both versions always
# exist together and can't drift out of sync with each other.
def savefig_pair(fig, path_no_ext, dpi=200, **kwargs):
    """Save fig as '<path_no_ext>.png' (white background -- the manuscript/
    LaTeX version, same filename a plain savefig would have used, so
    existing \\includegraphics paths in ch3_main.tex don't need to change)
    and '<path_no_ext>_presentation.png' (transparent background, for
    slides). Returns (manuscript_path, presentation_path).
    """
    manuscript_path = f"{path_no_ext}.png"
    presentation_path = f"{path_no_ext}_presentation.png"
    fig.savefig(manuscript_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", transparent=False, **kwargs)
    fig.savefig(presentation_path, dpi=dpi, bbox_inches="tight",
                transparent=True, **kwargs)
    print(f"  [ch3_style] wrote {manuscript_path} (white bg, manuscript)")
    print(f"  [ch3_style] wrote {presentation_path} (transparent bg, presentation)")
    return manuscript_path, presentation_path


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