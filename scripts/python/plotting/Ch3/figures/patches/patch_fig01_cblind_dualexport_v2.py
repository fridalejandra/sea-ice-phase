"""
patch_fig01_cblind_dualexport_v2.py — corrected, idempotent version.
=====================================================================
Fixes two things found while running v1 against the real repo:
  - ch3_config.py already got patched successfully by v1 -- this version
    skips step 1 if CBLIND_COLORS is already present, so it's safe to run
    again without duplicating that block.
  - ch3_plot.py's save() is the LAST thing in the file, and the real file
    has no trailing newline after "return path" -- v1's match text assumed
    one, which is why it silently found 0 matches. Fixed here.
  - Same idempotency guards added for ch3_plot.py and fig_01_conflation.py
    so this script can be re-run safely if it's interrupted partway again.

Run from figures/patches/: python3 patch_fig01_cblind_dualexport_v2.py
Then: py_compile all three, git diff to review, run fig_01_conflation.py.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.dirname(HERE)  # patches/ -> figures/


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


# ── 1. ch3_config.py: add CBLIND_COLORS (skips — already applied) ───────────
patch_file(
    "ch3_config.py",
    [
        ("COMPONENT_COLORS block",
         '''COMPONENT_COLORS = {
    "Trend":     "#C77B26",
    "Amplitude": "#378ADD",
    "Phase":     "#D4537E",
    "Residual":  "#BDBDBD",
}
''',
         '''COMPONENT_COLORS = {
    "Trend":     "#C77B26",
    "Amplitude": "#378ADD",
    "Phase":     "#D4537E",
    "Residual":  "#BDBDBD",
}

# ── Colorblind-safe palette (Okabe & Ito 2008 / the "Wong" qualitative set) ──
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
'''),
    ],
    skip_if_present="CBLIND_COLORS = {",
)

# ── 2. ch3_plot.py: dual manuscript/presentation export ─────────────────────
patch_file(
    "ch3_plot.py",
    [
        ("save() function",
         '''def save(fig, name, dpi=300, sync=True, tight=True):
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
    return path''',
         '''def _presentation_name(name):
    """fig01_concept_manuscript.png -> fig01_concept_presentation.png.

    Strips a trailing "_manuscript" before the extension if present, then
    appends "_presentation"; otherwise just appends "_presentation" before
    the extension. This is the naming convention save() uses for the
    transparent, no-background export (slides/talks).
    """
    root, ext = os.path.splitext(name)
    if root.endswith("_manuscript"):
        root = root[: -len("_manuscript")]
    return f"{root}_presentation{ext}"


def save(fig, name, dpi=300, sync=True, tight=True, presentation=False):
    """Save to OUTPUT_DIR and optionally rclone to Drive.

    Warns (does not block) if the figure is wider than ch3_config.TEXT_WIDTH_IN
    — catches a script that still hardcodes an oversized figsize= instead of
    using ch3_plot.figsize()/sector_grid()'s default.

    presentation: also save a second, transparent-background copy alongside
    the normal white-background one, named by _presentation_name(name), for
    use in slides/talks. Default False, so every OTHER figure script's
    existing save(fig, name, ...) call keeps producing exactly the one
    white-background PNG it always has — opt in per figure as each one gets
    its cblind/dual-export pass.
    """
    w_in, h_in = fig.get_size_inches()
    if w_in > TEXT_WIDTH_IN + 0.1:
        print(f"  WARNING: {name} is {w_in:.1f}in wide, target is "
              f"{TEXT_WIDTH_IN}in — it will run past the text margin or get "
              f"auto-shrunk (and its fonts with it) when placed in the document.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    paths = []

    # Manuscript export (white background) — unchanged from before.
    path = os.path.join(OUTPUT_DIR, name)
    if tight:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    else:
        fig.savefig(path, dpi=dpi, facecolor="white")
    print(f"Saved → {path}")
    paths.append(path)

    # Presentation export (transparent background) — opt-in, saved before the
    # figure is closed so it's the same figure object, same layout, no re-plot.
    if presentation:
        pres_name = _presentation_name(name)
        pres_path = os.path.join(OUTPUT_DIR, pres_name)
        # transparent=True makes matplotlib set the figure's and axes' patches
        # to "none" itself, overriding any facecolor= passed alongside it —
        # so facecolor is deliberately left off this call.
        if tight:
            fig.savefig(pres_path, dpi=dpi, bbox_inches="tight", transparent=True)
        else:
            fig.savefig(pres_path, dpi=dpi, transparent=True)
        print(f"Saved → {pres_path}  (presentation, transparent background)")
        paths.append(pres_path)

    plt.close(fig)

    if sync:
        for p in paths:
            try:
                r = subprocess.run(["rclone", "copy", p, GDRIVE],
                                   capture_output=True, text=True)
                print(f"Synced → {GDRIVE}" if r.returncode == 0
                      else f"rclone failed: {r.stderr.strip()}")
            except FileNotFoundError:
                print("rclone not found — skipping sync.")
    return path
'''),
    ],
    skip_if_present="def _presentation_name(name):",
)

# ── 3. fig_01_conflation.py: use CBLIND_COLORS + dual export + dependencies ──
patch_file(
    "fig_01_conflation.py",
    [
        ("docstring dependencies block",
         '''those aren't part of what ch3_style standardizes.
"""''',
         '''those aren't part of what ch3_style standardizes.

DEPENDENCIES (added 2026-09-30, part of the per-figure cblind/dual-export pass)
--------------------------------------------------------------------------
Data:   data/ch3/daily_fitted.csv (ch3_config.DAILY_CSV) -- column
        `iac_notrend`, filtered to sector="SIE_circumpolar", period="FULL"
        via ch3_data.load_daily(). Written by R/ch3/01_fit_apac.R.
Tables: none. This is a schematic (synthetic PHASE_SHIFT/AMP_SCALE applied to
        the real invariant cycle, not a results figure) -- it reads no
        results/ch3/tables/*.csv and writes none.
Output: results/ch3/figures/fig01_concept_manuscript.png   (white bg, manuscript)
        results/ch3/figures/fig01_concept_presentation.png (transparent bg, slides)
Colors: ch3_config.CBLIND_COLORS (Okabe-Ito colorblind-safe set).
"""'''),
        ("ch3_config import",
         '''import ch3_data as D
from ch3_plot import figsize, save
import ch3_style  # sets font (Helvetica/Tacoma) + spines for every figure
''',
         '''import ch3_data as D
from ch3_plot import figsize, save
import ch3_style  # sets font (Helvetica/Tacoma) + spines for every figure
from ch3_config import CBLIND_COLORS
'''),
        ("color constants",
         '''C_INV   = "#2C2C2A"
C_PHASE = "#D4537E"
C_AMP   = "#1D9E75"
C_ANNOT = "#5F5E5A"
INK     = "0.35"  # for panel titles, consistent with other figures''',
         '''# Colorblind-safe (Okabe-Ito): black for the neutral/invariant curve,
# vermillion for the phase-shift curve, blue for the amplitude-change curve --
# distinct under deuteranopia/protanopia/tritanopia, unlike the old near-black/
# pink/green trio (green vs. magenta-pink is a classic red-green-colorblind
# collision).
C_INV   = CBLIND_COLORS["black"]
C_PHASE = CBLIND_COLORS["vermillion"]
C_AMP   = CBLIND_COLORS["blue"]
C_ANNOT = "#5F5E5A"   # annotation ink only, not data-encoding -- left as is
INK     = "0.35"  # for panel titles, consistent with other figures'''),
        ("save() call",
         'save(fig, "fig01_concept_manuscript.png", sync=True)',
         'save(fig, "fig01_concept_manuscript.png", sync=True, presentation=True)'),
    ],
    skip_if_present="from ch3_config import CBLIND_COLORS",
)

print("\nDone. Now: py_compile all three, git diff, then run fig_01_conflation.py.")
