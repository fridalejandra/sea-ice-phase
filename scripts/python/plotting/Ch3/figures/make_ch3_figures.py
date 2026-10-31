#!/usr/bin/env python
"""make_ch3_figures.py -- build every Chapter 3 figure, in order, on the laptop.

    cd scripts/python/plotting/Ch3/figures
    python make_ch3_figures.py              # all
    python make_ch3_figures.py fig05        # only steps whose name contains "fig05"

Outputs (from ch3_style.save_figure):
    results/ch3/figures/manuscript/ch3_figNN_*.png   white background -> Overleaf figures/
    results/ch3/figures/slides/ch3_figNN_*.png       transparent background
Each step runs in its own process; a failure is reported and the next step still runs."""
import os, sys, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
CH3 = os.path.dirname(HERE)

STEPS = [
    # prerequisite tables (rebuilt from the current data every time)
    ("t32 attribution table", os.path.join(HERE, "fig_03-04_attribution_annual.py")),
    ("SST partition tables",  os.path.join(CH3, "ch3_sst_partition.py")),
    # figures
    ("ch3_fig01 cycle schematic",     os.path.join(HERE, "ch3_fig01_cycle_schematic.py")),
    ("ch3_fig02 sector map",          os.path.join(HERE, "ch3_fig02_sector_map.py")),
    ("ch3_fig03 decomposition 2022",  os.path.join(HERE, "ch3_fig03_decomposition_2022.py")),
    ("ch3_fig04 components by cycle", os.path.join(HERE, "ch3_fig04_components_by_cycle.py")),
    ("ch3_fig05 mean-amp coupling",   os.path.join(HERE, "ch3_fig05_mean_amp_coupling.py")),
    ("ch3_fig06 SST components",      os.path.join(HERE, "ch3_fig06_sst_components.py")),
    ("ch3_figA1 mean leak",           os.path.join(HERE, "ch3_figA1_mean_leak.py")),
]


def main():
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    results = []
    for label, path in STEPS:
        if only and not any(o in label or o in os.path.basename(path) for o in only):
            continue
        print(f"\n=== {label}  ({os.path.relpath(path, CH3)})")
        if not os.path.exists(path):
            print("    MISSING -- skipped"); results.append((label, "missing")); continue
        t0 = time.time()
        r = subprocess.run([sys.executable, path], cwd=os.path.dirname(path))
        status = "ok" if r.returncode == 0 else f"FAILED (exit {r.returncode})"
        results.append((label, f"{status}, {time.time() - t0:.0f}s"))
    print("\n=== summary")
    for label, status in results:
        print(f"    {label:32s} {status}")


if __name__ == "__main__":
    main()
