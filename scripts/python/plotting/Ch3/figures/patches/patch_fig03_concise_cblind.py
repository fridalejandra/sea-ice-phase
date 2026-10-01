"""
patch_fig03_concise_cblind.py -- Fig. 3 content/style pass (2026-10-01).
================================================================================
Three changes, discussed and confirmed against the actual numbers in
t32_component_size_era.csv (not just the figure's screenshot):

1. Drop circumpolar (sum of the other 5 sectors, adds nothing new) and raw
   anomaly (uniformly the smallest, flattest term in every sector/era) from
   the PLOTTED grid -- 30 panels -> 20. Phase stays in: it beats raw anomaly
   in every sector/era and is growing post-2016 in several (Ross, Weddell,
   circumpolar), so it's part of the story, not noise.
2. The full stats computation (all 6 sectors x all 5 components) is split
   into its own pass, independent of what's plotted, so
   t32_component_size_era.csv is completely unaffected by this change.
3. cblind colors (CBLIND_COLORS vermillion/blue, replacing hardcoded hex),
   dual export (presentation=True via ch3_plot.save(), replacing a direct
   fig.savefig() call), and bigger era-mean numbers (7pt -> 9.5pt).

Does NOT touch the LABEL dict (sector display names) -- deliberately: it's
matched by neither old nor new string in the two edits around it, so its
embedded \n line breaks never have to be retyped by this patch.

Run from figures/patches/: python3 patch_fig03_concise_cblind.py
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
        ("docstring: design rationale + DEPENDENCIES block",
         '''"""
fig_03_components_by_cycle.py -- Fig. 3 (replaces the old Figs 3 and 4): each
cycle's departure from the invariant cycle and its four components, drawn
SEPARATELY rather than stacked.

Rows = sectors (Weddell, A-B, Ross, EA, KH, circumpolar).
Columns = the departure from the invariant cycle (what an extent anomaly
sees), then trend, amplitude, phase and raw anomaly. One bar per cycle
(21 Feb to 20 Feb), positive red / negative blue. All panels in a row share
one y-axis, so the components can be compared with each other and with the
total directly. In each panel, two short horizontal lines give the mean SIZE
of that component (mean |bar|) over 1979-2015 and 2016-2024, with the values
printed, which replaces the era-share figure (old Fig. 4) and the NET/GROSS
distinction: the figure shows the sizes themselves.

Phase is small in these bars for a reason the caption should state: a shift
in timing adds ice in the advance and removes it in the retreat, so it
largely cancels in the mean over a cycle. Fig. 10 shows it day by day.

Reads  results/ch3/tables/t32_attribution_by_cycle.csv (from fig_03-04_attribution_annual.py)
Writes results/ch3/figures/fig03_components_by_cycle.png
       results/ch3/tables/t32_component_size_era.csv
"""''',
         '''"""
fig_03_components_by_cycle.py -- Fig. 3 (replaces the old Figs 3 and 4): each
cycle's departure from the invariant cycle and three of its components,
drawn SEPARATELY rather than stacked.

Rows = sectors (Weddell, A-B, Ross, EA, KH). Circumpolar total (sum of these
five) is omitted from the plotted grid -- it adds no information beyond the
five rows -- but its numbers are still in the companion CSV below.
Columns = the departure from the invariant cycle (what an extent anomaly
sees), then trend, amplitude, and phase. Raw anomaly (the 4th component, a
residual) is also omitted from the plotted grid: checked against
t32_component_size_era.csv, it is the smallest and flattest term in every
sector and era (0.01-0.06 x10^6 km^2 throughout) -- unlike phase, kept
deliberately: phase beats raw anomaly in every sector/era and is GROWING
post-2016 in several (Ross 0.069->0.099, Weddell 0.055->0.087, circumpolar
0.121->0.133), so it's part of the post-2016 story this figure is making,
not noise to cut. Dropping circumpolar + raw anomaly together takes the
grid from 30 panels (6x5) to 20 (5x4).
One bar per cycle (21 Feb to 20 Feb), positive red / negative blue. All
panels in a row share one y-axis, so the components can be compared with
each other and with the total directly. In each panel, two short horizontal
lines give the mean SIZE of that component (mean |bar|) over 1979-2015 and
2016-2024, with the values printed, which replaces the era-share figure
(old Fig. 4) and the NET/GROSS distinction: the figure shows the sizes
themselves.

Phase is small in these bars for a reason the caption should state: a shift
in timing adds ice in the advance and removes it in the retreat, so it
largely cancels in the mean over a cycle. Fig. 10 shows it day by day.

DEPENDENCIES (added 2026-10-01, part of the per-figure cblind/dual-export pass)
--------------------------------------------------------------------------
Data:   none beyond the table below -- no direct NetCDF/pipeline read.
Tables: reads results/ch3/tables/t32_attribution_by_cycle.csv (written by
        fig_03-04_attribution_annual.py -- this script exits with a clear
        message if that file is missing).
        Writes results/ch3/tables/t32_component_size_era.csv -- era-mean
        |contribution| for ALL 6 sectors x ALL 5 components, independent of
        the 5x4 grid actually plotted (see above).
Colors: positive/negative bars now draw from ch3_config.CBLIND_COLORS
        (vermillion / blue) instead of local hardcoded hex.

Reads  results/ch3/tables/t32_attribution_by_cycle.csv (from fig_03-04_attribution_annual.py)
Writes results/ch3/figures/fig03_components_by_cycle.png              (white bg, manuscript)
       results/ch3/figures/fig03_components_by_cycle_presentation.png (transparent bg, slides)
       results/ch3/tables/t32_component_size_era.csv
"""'''),
        ("imports: add CBLIND_COLORS + save",
         '''from ch3_config import SECTORS, TABLES_DIR, OUTPUT_DIR, BREAK_YEAR
import ch3_style''',
         '''from ch3_config import SECTORS, TABLES_DIR, OUTPUT_DIR, BREAK_YEAR, CBLIND_COLORS
from ch3_plot import save
import ch3_style'''),
        ("add PLOT_SECTORS / PLOT_COLS right after COLS (LABEL dict untouched)",
         '''COLS = [("anomaly_from_iac", "Extent anomaly"),
        ("trend_component", "Trend"),
        ("amplitude_component", "Amplitude"),
        ("phase_component", "Phase"),
        ("residual_apac", "Raw anomaly")]''',
         '''COLS = [("anomaly_from_iac", "Extent anomaly"),
        ("trend_component", "Trend"),
        ("amplitude_component", "Amplitude"),
        ("phase_component", "Phase"),
        ("residual_apac", "Raw anomaly")]
# Plotted subset (2026-10-01): circumpolar + raw anomaly dropped from the
# grid -- see the DEPENDENCIES note in the module docstring for why. Both
# are still computed and written to t32_component_size_era.csv in full.
PLOT_SECTORS = [s for s in SECTORS if s != "SIE_circumpolar"]
PLOT_COLS = [(c, lab) for c, lab in COLS if c != "residual_apac"]'''),
        ("cblind colors, right after LABEL dict (LABEL dict untouched)",
         'POS, NEG, INK = "#d6604d", "#4393c3", "0.35"',
         '''# Colorblind-safe (Okabe-Ito), replacing local hardcoded hex (2026-10-01).
POS, NEG, INK = CBLIND_COLORS["vermillion"], CBLIND_COLORS["blue"], "0.35"'''),
        ("stats pass split out + plot loop over PLOT_SECTORS/PLOT_COLS + dual export",
         '''nr, nc = len(SECTORS), len(COLS)
fig, axes = plt.subplots(nr, nc, figsize=(2.35 * nc + 1.0, 1.75 * nr + 0.8),
                         sharex=True, squeeze=False)
bold = ch3_style.bold_font_properties(size=10.5)
rows = []
for i, s in enumerate(SECTORS):
    g = t[t.sector == s].sort_values(YR)
    yr = g[YR].values
    lim = 1.08 * np.nanmax(np.abs(g[[c for c, _ in COLS]].values))
    for j, (c, lab) in enumerate(COLS):
        ax = axes[i, j]
        v = g[c].values
        ax.bar(yr, v, width=0.8, color=np.where(v >= 0, POS, NEG), lw=0)
        ax.axhline(0, color="0.2", lw=0.6)
        ax.axvline(BREAK_YEAR - 0.5, color="0.5", lw=0.8, ls=(0, (2, 2)))
        ax.set_ylim(-lim, lim)
        # era mean size, drawn at the top of the panel as a bracket-like tick
        for (y0, y1), yy in (((yr.min(), BREAK_YEAR - 1), 0.86), ((BREAK_YEAR, yr.max()), 0.86)):
            m = np.nanmean(np.abs(v[(yr >= y0) & (yr <= y1)]))
            ax.plot([y0, y1], [yy * lim, yy * lim], color="0.15", lw=1.4, solid_capstyle="butt")
            ax.text((y0 + y1) / 2, yy * lim * 1.02, f"{m:.2f}", ha="center", va="bottom",
                    fontsize=7, color="0.15")
            rows.append(dict(sector=s, component=c, era=f"{int(y0)}-{int(y1)}", mean_abs=m))
        ax.tick_params(labelsize=7.5, colors=INK, length=2.5)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(INK)
        ax.spines[["top", "right"]].set_visible(False)
        if j > 0:
            ax.tick_params(labelleft=False)
        if i == 0:
            ax.set_title(lab, fontproperties=bold, color="0.1", pad=14)
        if j == 0:
            ax.set_ylabel(LABEL.get(s, s), fontproperties=bold, color="0.1", labelpad=6)
        if i == nr - 1:
            ax.set_xticks([1980, 2000, 2020])
fig.text(0.005, 0.5, "cycle-mean contribution (10$^6$ km$^2$)", rotation=90, va="center",
         fontsize=9, color=INK)
fig.tight_layout(rect=[0.02, 0, 1, 1], h_pad=0.9, w_pad=0.4)
out = os.path.join(OUTPUT_DIR, "fig03_components_by_cycle.png")
fig.savefig(out, dpi=250, bbox_inches="tight")
plt.close(fig)
e = pd.DataFrame(rows)
e.to_csv(os.path.join(TABLES_DIR, "t32_component_size_era.csv"), index=False)
print(f"wrote {out}")
print(e.pivot_table(index=["sector", "component"], columns="era", values="mean_abs").round(3).to_string())''',
         '''# -- stats pass: ALL 6 sectors x ALL 5 components, independent of what's
# plotted below (2026-10-01) -- t32_component_size_era.csv is unaffected by
# which panels end up in the figure.
rows = []
for s in SECTORS:
    g = t[t.sector == s].sort_values(YR)
    yr = g[YR].values
    for c, lab in COLS:
        v = g[c].values
        for y0, y1 in ((yr.min(), BREAK_YEAR - 1), (BREAK_YEAR, yr.max())):
            m = np.nanmean(np.abs(v[(yr >= y0) & (yr <= y1)]))
            rows.append(dict(sector=s, component=c, era=f"{int(y0)}-{int(y1)}", mean_abs=m))
e = pd.DataFrame(rows)
e.to_csv(os.path.join(TABLES_DIR, "t32_component_size_era.csv"), index=False)
print(e.pivot_table(index=["sector", "component"], columns="era", values="mean_abs").round(3).to_string())

# -- plot pass: PLOT_SECTORS x PLOT_COLS only (circumpolar + raw anomaly
# dropped from the grid, 2026-10-01 -- full stats above are unaffected).
nr, nc = len(PLOT_SECTORS), len(PLOT_COLS)
fig, axes = plt.subplots(nr, nc, figsize=(2.35 * nc + 1.0, 1.75 * nr + 0.8),
                         sharex=True, squeeze=False)
bold = ch3_style.bold_font_properties(size=10.5)
for i, s in enumerate(PLOT_SECTORS):
    g = t[t.sector == s].sort_values(YR)
    yr = g[YR].values
    lim = 1.08 * np.nanmax(np.abs(g[[c for c, _ in PLOT_COLS]].values))
    for j, (c, lab) in enumerate(PLOT_COLS):
        ax = axes[i, j]
        v = g[c].values
        ax.bar(yr, v, width=0.8, color=np.where(v >= 0, POS, NEG), lw=0)
        ax.axhline(0, color="0.2", lw=0.6)
        ax.axvline(BREAK_YEAR - 0.5, color="0.5", lw=0.8, ls=(0, (2, 2)))
        ax.set_ylim(-lim, lim)
        # era mean size, drawn at the top of the panel as a bracket-like tick
        # (text bumped 7 -> 9.5pt, 2026-10-01, now that the grid is smaller)
        for y0, y1 in ((yr.min(), BREAK_YEAR - 1), (BREAK_YEAR, yr.max())):
            m = np.nanmean(np.abs(v[(yr >= y0) & (yr <= y1)]))
            ax.plot([y0, y1], [0.86 * lim, 0.86 * lim], color="0.15", lw=1.4, solid_capstyle="butt")
            ax.text((y0 + y1) / 2, 0.86 * lim * 1.02, f"{m:.2f}", ha="center", va="bottom",
                    fontsize=9.5, color="0.15")
        ax.tick_params(labelsize=7.5, colors=INK, length=2.5)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(INK)
        ax.spines[["top", "right"]].set_visible(False)
        if j > 0:
            ax.tick_params(labelleft=False)
        if i == 0:
            ax.set_title(lab, fontproperties=bold, color="0.1", pad=14)
        if j == 0:
            ax.set_ylabel(LABEL.get(s, s), fontproperties=bold, color="0.1", labelpad=6)
        if i == nr - 1:
            ax.set_xticks([1980, 2000, 2020])
fig.text(0.005, 0.5, "cycle-mean contribution (10$^6$ km$^2$)", rotation=90, va="center",
         fontsize=9, color=INK)
fig.tight_layout(rect=[0.02, 0, 1, 1], h_pad=0.9, w_pad=0.4)
out = save(fig, "fig03_components_by_cycle.png", dpi=250, sync=False, presentation=True)
print("wrote", out)'''),
    ],
    skip_if_present="PLOT_SECTORS = [s for s in SECTORS",
)

print("\nDone. Now: py_compile, git diff, then run fig_03_components_by_cycle.py.")
