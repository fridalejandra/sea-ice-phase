"""
fix_duplicate_cblind.py — removes the accidental duplicate CBLIND_COLORS
block in ch3_config.py (caused by patch_fig01_cblind_dualexport.py getting
run twice before it errored out on ch3_plot.py). Run from figures/patches/.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.dirname(HERE)
path = os.path.join(FIGURES_DIR, "ch3_config.py")

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

BLOCK = '''# ── Colorblind-safe palette (Okabe & Ito 2008 / the "Wong" qualitative set) ──
# Added 2026-09-30 for the figure-by-figure cblind/dual-export pass (manuscript
# = white bg, presentation = transparent bg -- see ch3_plot.save()). Distinct
# under deuteranopia, protanopia, and tritanopia. Not a retrofit of
# SECTOR_COLORS/COMPONENT_COLORS yet -- those get their own pass later, in
# turn, as their figures come up in the one-by-one rotation. This is the
# shared source new figure-local color constants should draw from as each
# figure is redone.
CBLIND_COLORS = {
    "black":          "#000000",
    "orange":         "#E69F00",
    "sky_blue":       "#56B4E9",
    "bluish_green":   "#009E73",
    "yellow":         "#F0E442",
    "blue":           "#0072B2",
    "vermillion":     "#D55E00",
    "reddish_purple": "#CC79A7",
}
'''

doubled = BLOCK + "\n" + BLOCK
n = content.count(doubled)
if n != 1:
    n_single = content.count(BLOCK)
    raise SystemExit(
        f"Expected exactly one back-to-back duplicate, found {n}. "
        f"(Single-block count in file: {n_single}.) Aborting without "
        "writing anything -- diff ch3_config.py by hand instead."
    )

content = content.replace(doubled, BLOCK, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

remaining = content.count("CBLIND_COLORS = {")
print(f"Fixed. CBLIND_COLORS now appears {remaining} time(s) in ch3_config.py (should be 1).")
