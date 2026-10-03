#!/usr/bin/env python3
"""
fig_04b_component_schematic.py -- DRAFT, not yet numbered/placed.

Conceptual companion to Fig. 4 (and the APAC decomposition in Methods):
a purely SCHEMATIC figure (no real data) showing what it means, physically,
for a mode to act on only one component of the annual cycle. Three panels,
one per component -- trend, amplitude, phase -- each showing the same toy
invariant annual cycle (grey dashed) against a version with ONLY that one
component perturbed (solid, colored), holding the other two fixed.

The point: Fig. 4's heatmap reports correlations between indices and
components as numbers; a reader who isn't already fluent in "amplitude vs.
phase vs. trend" has to take "this index loads on amplitude, not phase" on
faith. This figure gives that sentence a picture -- what a pure amplitude
shift looks like on the cycle curve, as opposed to a pure phase shift or a
pure trend shift -- without claiming anything about the real data at all.

Deliberately NOT tied to ch3_config/ch3_plot's data-reading machinery --
there is nothing to read, so this can run anywhere ch3_style imports.

Open questions for Frida (left as TODOs below):
  - Placement: Methods (right after defining APAC) vs. beside Fig. 4 in
    3.2 -- this draft assumes Methods, hence the "schematic" framing and
    arbitrary-units axis rather than real units.
  - Which three components to show -- this draft picks trend/amplitude/
    phase (dropping raw anomaly, the same choice Fig. 3 already made) for
    consistency. Easy to add a 4th panel if you want raw anomaly too.
  - Numbering -- "04b" is a placeholder; rename once you decide where it
    lands (e.g. fig_02_component_schematic.py if it moves to Methods,
    ahead of the first real data figure).

Writes  results/ch3/figures/fig04b_component_schematic.png              (white bg, DRAFT)
        results/ch3/figures/fig04b_component_schematic_presentation.png (transparent bg, slides)
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ch3_style

try:
    from ch3_config import OUTPUT_DIR
except Exception:
    OUTPUT_DIR = os.path.join(HERE, "_draft_output")

# -- toy invariant annual cycle (not real data) --------------------------
# Antarctic SIE: summer minimum near day 0 (~21 Feb), winter maximum near
# day ~182 (~August), back to minimum at day 365 -- a single cosine is a
# fine stand-in for "the shape everyone recognizes," nothing more.
day = np.linspace(0, 365, 400)
BASE, AMP = 0.0, 1.0


def cycle(day, baseline=BASE, amplitude=AMP, phase_shift_days=0.0):
    return baseline - amplitude * np.cos(2 * np.pi * (day - phase_shift_days) / 365)


invariant = cycle(day)

PANELS = [
    ("(a)  Trend", "a pure trend: the whole cycle sits\nhigher, same size and timing",
     cycle(day, baseline=BASE + 0.55), ch3_style.VERMILLION),
    ("(b)  Amplitude", "a pure amplitude change: bigger swing,\nsame mean level and timing",
     cycle(day, amplitude=AMP * 1.55), ch3_style.BLUE),
    ("(c)  Phase", "a pure phase shift: same size and\nmean level, earlier timing",
     cycle(day, phase_shift_days=-30), ch3_style.GREEN),
]

fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.3), sharey=True)
bold = ch3_style.bold_font_properties(size=10.5)
INK = "0.35"

for ax, (title, note, perturbed, color) in zip(axes, PANELS):
    ax.plot(day, invariant, color="0.55", lw=1.6, ls=(0, (4, 2)), label="invariant cycle")
    ax.plot(day, perturbed, color=color, lw=2.2, label="perturbed")
    ax.axhline(BASE, color="0.8", lw=0.7, zorder=0)
    ax.set_title(title, loc="left", fontproperties=bold, color="0.1", pad=10)
    ax.text(0.5, -0.30, note, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.5, color=INK)
    ax.set_xticks([0, 182, 365])
    ax.set_xticklabels(["Feb", "Aug", "Feb"])
    ax.set_yticks([])
    ax.tick_params(colors=INK, labelsize=9, length=3)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("extent anomaly\n(schematic, arbitrary units)", color=INK, fontsize=9.5)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9.5,
           bbox_to_anchor=(0.5, 1.12))
fig.suptitle("")  # no title text beyond the legend; caption carries the framing
fig.tight_layout(rect=[0, 0.08, 1, 0.92])

os.makedirs(OUTPUT_DIR, exist_ok=True)
stem = os.path.join(OUTPUT_DIR, "fig04b_component_schematic")
fig.savefig(f"{stem}_presentation.png", dpi=220, bbox_inches="tight", transparent=True)
fig.savefig(f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white", transparent=False)
print(f"wrote {stem}.png (+ _presentation.png)")
plt.close(fig)