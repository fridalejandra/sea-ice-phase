"""
patch_fig03_landscape.py -- Fig 3 goes landscape (confirmed OK with the
Graduate Division, 2026-10-01). The 20-panel bar-chart grid was too cramped
at the portrait ch3_config.TEXT_WIDTH_IN (6.5in) -- 47 years x 4 components
x 5 sectors needs more horizontal room than a portrait text column gives.
Defines a local LANDSCAPE_WIDTH_IN (9.0in) rather than touching the shared
ch3_config.py, since this is currently the only figure going landscape.

Run from figures/patches/: python3 patch_fig03_landscape.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.dirname(HERE)


def patch_file(relname, replacements, skip_if_present=None):
    path = os.path.join(FIGURES_DIR, relname)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if skip_if_present is not None and skip_if_present in content:
        print(f"Skipping {relname} — already patched (found {skip_if_present!r}).")
        return

    for label, old, new in replacements:
        n = content.count(old)
        if n != 1:
            raise SystemExit(
                f"{relname}: expected exactly 1 match for {label!r}, found {n}. "
                "File has drifted from what this patch expects -- aborting "
                "without writing anything to this file. Diff it by hand."
            )
        content = content.replace(old, new, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Patched {relname} ({len(replacements)} edit(s))")


patch_file(
    "fig_03_components_by_cycle.py",
    [
        ("drop unused TEXT_WIDTH_IN import",
         "from ch3_config import SECTORS, TABLES_DIR, OUTPUT_DIR, BREAK_YEAR, CBLIND_COLORS, TEXT_WIDTH_IN",
         "from ch3_config import SECTORS, TABLES_DIR, OUTPUT_DIR, BREAK_YEAR, CBLIND_COLORS"),
        ("add local LANDSCAPE_WIDTH_IN",
         '''# Colorblind-safe (Okabe-Ito), replacing local hardcoded hex (2026-10-01).
POS, NEG, INK = CBLIND_COLORS["vermillion"], CBLIND_COLORS["blue"], "0.35"''',
         '''# Colorblind-safe (Okabe-Ito), replacing local hardcoded hex (2026-10-01).
POS, NEG, INK = CBLIND_COLORS["vermillion"], CBLIND_COLORS["blue"], "0.35"

# Landscape width (2026-10-01, confirmed OK with the Graduate Division): this
# figure is placed as a landscape page in the thesis -- the 20-panel grid
# (47 years x 4 components x 5 sectors) was too cramped at the portrait
# ch3_config.TEXT_WIDTH_IN (6.5in). Sized the same way TEXT_WIDTH_IN is
# (page width minus 1in margins each side), on an 11in landscape page.
LANDSCAPE_WIDTH_IN = 9.0'''),
        ("figsize: landscape width instead of portrait TEXT_WIDTH_IN",
         '''# Width fixed to TEXT_WIDTH_IN (2026-10-01) -- was 2.35*nc+1.0, which grew
# past the page's text margin (10.4in at nc=4) since it was never anchored
# to the page width the way Figs 1-2 are. Height still scales with nr.
fig, axes = plt.subplots(nr, nc, figsize=(TEXT_WIDTH_IN, 1.75 * nr + 0.8),
                         sharex=True, squeeze=False)''',
         '''# Landscape figure (2026-10-01, confirmed OK with the Graduate Division) --
# LANDSCAPE_WIDTH_IN above, not the portrait TEXT_WIDTH_IN. save() will
# still print its "wider than TEXT_WIDTH_IN" warning below since that check
# only knows about the portrait width -- expected and fine to ignore here.
# Place this figure's page as landscape when inserting it into the thesis.
fig, axes = plt.subplots(nr, nc, figsize=(LANDSCAPE_WIDTH_IN, 1.75 * nr + 0.8),
                         sharex=True, squeeze=False)'''),
    ],
    skip_if_present="LANDSCAPE_WIDTH_IN = 9.0",
)

print("\nDone.")
