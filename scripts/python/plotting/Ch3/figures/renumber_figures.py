#!/usr/bin/env python3
"""
Renumber the Chapter 3 figure scripts to match the manuscript.

The sections were reordered, so the manuscript figure numbers no longer match
the script numbers, and figures are being placed under the wrong captions.
This makes the scripts agree with the manuscript, permanently.

    old script                            ->  new script            manuscript
    fig_05_rolling_phase_amp              ->  fig_07_...            Fig. 7  (3.4)
    fig_06_raw_anomaly_persistence        ->  fig_09_...            Fig. 9  (3.5)
    fig_07_component_comparison_heatmap   ->  fig_05_...            Fig. 5  (3.2.2)
    fig_08_atmosphere_sevenpairs          ->  _dropped_...          (not in the manuscript)
    fig_09_ross_asl_nonstationarity       ->  fig_06_...            Fig. 6  (3.3)
    fig_10_abs_growth_season              ->  fig_08_...            Fig. 8  (3.4.2)
    compute_fig07_component_comparison    ->  compute_fig05_...

Figures 1-4 and S1-S4 are unchanged.

It is a swap (5<->7, 6<->9), so every rename goes through a temporary name
first; nothing can be overwritten. For each renamed script it also rewrites
the PNG name the script saves to (whatever padding it used: fig06_, fig6_,
fig_06_), fixes the entry in run_all.py, and renames any already-existing
PNG in the figures output directory.

DRY RUN BY DEFAULT. Nothing is changed until you pass --apply.

    cd .../scripts/python/plotting/Ch3/figures
    python renumber_figures.py            # shows every action, changes nothing
    python renumber_figures.py --apply    # does it
    python renumber_figures.py --apply --figdir /path/to/results/ch3/figures

Uses `git mv` when the folder is inside a git repo, so history follows the
file. Safe to rerun: anything already renamed is skipped.
"""
import os
import re
import sys
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
APPLY = "--apply" in sys.argv

# (old stem, new stem, old fig number, new fig number or None)
PLAN = [
    ("fig_05_rolling_phase_amp",            "fig_07_rolling_phase_amp",            5,  7),
    ("fig_06_raw_anomaly_persistence",      "fig_09_raw_anomaly_persistence",      6,  9),
    ("fig_07_component_comparison_heatmap", "fig_05_component_comparison_heatmap", 7,  5),
    ("fig_08_atmosphere_sevenpairs",        "_dropped_atmosphere_sevenpairs",      8,  None),
    ("fig_09_ross_asl_nonstationarity",     "fig_06_ross_asl_nonstationarity",     9,  6),
    ("fig_10_abs_growth_season",            "fig_08_abs_growth_season",            10, 8),
    ("compute_fig07_component_comparison",  "compute_fig05_component_comparison",  7,  5),
]

RUN_ALL_LABELS = {  # new stem -> new "what it makes" label for run_all.py
    "fig_05_component_comparison_heatmap": "Fig. 5 index/wind vs components",
    "fig_06_ross_asl_nonstationarity":     "Fig. 6 Ross-ASL",
    "fig_07_rolling_phase_amp":            "Fig. 7 timing-amplitude correlation",
    "fig_08_abs_growth_season":            "Fig. 8 ABS growth season",
    "fig_09_raw_anomaly_persistence":      "Fig. 9 raw-anomaly autocorrelation; Table 4",
    "compute_fig05_component_comparison":  "t37, t37b, t37c; same-day wind coupling (Sect. 3.2.2)",
}


def say(msg):
    print(("  " if APPLY else "  [dry] ") + msg)


def in_git():
    try:
        r = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                           cwd=HERE, capture_output=True, text=True, timeout=10)
        return r.returncode == 0 and r.stdout.strip() == "true"
    except Exception:
        return False


GIT = in_git()


def mv(src, dst):
    say("mv  %s  ->  %s" % (os.path.basename(src), os.path.basename(dst)))
    if not APPLY:
        return
    if os.path.exists(dst):
        sys.exit("refusing to overwrite %s" % dst)
    if GIT:
        r = subprocess.run(["git", "mv", src, dst], cwd=HERE, capture_output=True, text=True)
        if r.returncode != 0:          # untracked file -> plain move
            shutil.move(src, dst)
    else:
        shutil.move(src, dst)


def fig_tokens(n):
    """every spelling a figure number turns up in: fig05_, fig5_, fig_05_, fig_5_"""
    return sorted({"fig%02d_" % n, "fig%d_" % n, "fig_%02d_" % n, "fig_%d_" % n}, key=len, reverse=True)


def rewrite_png_names(path, old_n, new_n):
    """inside one script, change the figure number in its output PNG name(s)."""
    with open(path, encoding="utf-8") as f:
        s = f.read()
    orig = s
    if new_n is None:
        for t in fig_tokens(old_n):
            s = s.replace(t, "dropped_")
    else:
        for t in fig_tokens(old_n):
            s = s.replace(t, "fig%02d_" % new_n)
    if s != orig:
        n = sum(orig.count(t) for t in fig_tokens(old_n))
        say("    %d internal fig-number token(s) rewritten in %s" % (n, os.path.basename(path)))
        if APPLY:
            with open(path, "w", encoding="utf-8") as f:
                f.write(s)


def rename_pngs(figdir, old_n, new_n):
    if not figdir or not os.path.isdir(figdir):
        return
    for fn in sorted(os.listdir(figdir)):
        for t in fig_tokens(old_n):
            if fn.startswith(t) and fn.lower().endswith((".png", ".pdf", ".svg")):
                new = ("dropped_" if new_n is None else "fig%02d_" % new_n) + fn[len(t):]
                mv(os.path.join(figdir, fn), os.path.join(figdir, "__tmp__" + new))
                break


def main():
    print("figure-script renumber  (%s)\n" % ("APPLY" if APPLY else "dry run -- add --apply to execute"))
    print("folder: %s%s\n" % (HERE, "   [git]" if GIT else ""))

    figdir = None
    if "--figdir" in sys.argv:
        figdir = sys.argv[sys.argv.index("--figdir") + 1]
    else:
        try:
            sys.path.insert(0, HERE)
            import ch3_config
            figdir = getattr(ch3_config, "FIGURES_DIR", None) or getattr(ch3_config, "OUTPUT_DIR", None)
        except Exception:
            pass
    print("figure output dir: %s\n" % (figdir or "(not found -- existing PNGs left alone; pass --figdir)"))

    todo = []
    for old, new, on, nn in PLAN:
        src = os.path.join(HERE, old + ".py")
        dst = os.path.join(HERE, new + ".py")
        if os.path.exists(dst) and not os.path.exists(src):
            print("  already done: %s" % new)
            continue
        if not os.path.exists(src):
            print("  not present, skipped: %s" % old)
            continue
        todo.append((old, new, on, nn, src, dst))
    if not todo:
        print("\nnothing to do.")
        return

    # ── phase 1: everything to a temporary name, so swaps cannot collide ──
    print("\nphase 1 -- to temporary names")
    for old, new, on, nn, src, dst in todo:
        mv(src, os.path.join(HERE, "__tmp__" + new + ".py"))
    for old, new, on, nn, src, dst in todo:
        if on is not None:
            rename_pngs(figdir, on, nn)

    # ── phase 2: temporary names to final names ───────────────────────────
    print("\nphase 2 -- to final names")
    for old, new, on, nn, src, dst in todo:
        tmp = os.path.join(HERE, "__tmp__" + new + ".py")
        mv(tmp, dst)
        if APPLY or os.path.exists(src):
            rewrite_png_names(dst if APPLY else src, on, nn)
    if figdir and os.path.isdir(figdir):
        for fn in sorted(os.listdir(figdir)):
            if fn.startswith("__tmp__"):
                mv(os.path.join(figdir, fn), os.path.join(figdir, fn[len("__tmp__"):]))

    # ── phase 3: run_all.py ───────────────────────────────────────────────
    ra = os.path.join(HERE, "run_all.py")
    if os.path.exists(ra):
        print("\nphase 3 -- run_all.py")
        with open(ra, encoding="utf-8") as f:
            s = f.read()
        orig = s
        for old, new, on, nn, src, dst in todo:
            pat = re.compile(r'(os\.path\.join\(HERE,\s*")' + re.escape(old) + r'(\.py"\),\s*"py",\s*")[^"]*(")')
            label = RUN_ALL_LABELS.get(new, "(dropped from the manuscript)")
            s, k = pat.subn(lambda m: m.group(1) + new + m.group(2) + label + m.group(3), s)
            if k:
                say("entry %s -> %s   [%s]" % (old, new, label))
        s = s.replace("python run_all.py fig_07     one step", "python run_all.py fig_05     one step")
        if s != orig and APPLY:
            with open(ra, "w", encoding="utf-8") as f:
                f.write(s)
    print("\ndone." if APPLY else "\ndry run complete. Rerun with --apply to execute.")


if __name__ == "__main__":
    main()
