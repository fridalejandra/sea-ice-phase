"""
patch_fig03_textwidth.py -- fixes Fig 3's figure width to actually target
ch3_config.TEXT_WIDTH_IN (6.5in), same as every other figure, instead of an
ad hoc formula (2.35 * nc + 1.0) that was never anchored to the page width.
That's why save() warned the PNG was 10.4in against a 6.5in target: this
script never imported figsize()/TEXT_WIDTH_IN like Figs 1-2 do.

Height keeps scaling with nr (that part was fine) -- only width changes.

Run from figures/patches/: python3 patch_fig03_textwidth.py
Then: py_compile, git diff, run fig_03_components_by_cycle.py.
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
        ("import TEXT_WIDTH_IN",
         "from ch3_config import SECTORS, TABLES_DIR, OUTPUT_DIR, BREAK_YEAR, CBLIND_COLORS",
         "from ch3_config import SECTORS, TABLES_DIR, OUTPUT_DIR, BREAK_YEAR, CBLIND_COLORS, TEXT_WIDTH_IN"),
        ("figsize: fixed width (TEXT_WIDTH_IN) instead of scaling with nc",
         '''fig, axes = plt.subplots(nr, nc, figsize=(2.35 * nc + 1.0, 1.75 * nr + 0.8),
                         sharex=True, squeeze=False)''',
         '''# Width fixed to TEXT_WIDTH_IN (2026-10-01) -- was 2.35*nc+1.0, which grew
# past the page's text margin (10.4in at nc=4) since it was never anchored
# to the page width the way Figs 1-2 are. Height still scales with nr.
fig, axes = plt.subplots(nr, nc, figsize=(TEXT_WIDTH_IN, 1.75 * nr + 0.8),
                         sharex=True, squeeze=False)'''),
    ],
    skip_if_present="figsize=(TEXT_WIDTH_IN, 1.75 * nr + 0.8)",
)

print("\nDone. Now: py_compile, git diff, then run fig_03_components_by_cycle.py.")
