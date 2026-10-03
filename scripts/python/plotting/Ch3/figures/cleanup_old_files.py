#!/usr/bin/env python3
"""
cleanup_old_files.py -- delete superseded fit versions and stale copies, so
no script can silently read an old file again (the duplicated-row bug behind
Table 3's old "p < 1e-5" came from exactly this kind of leftover).

Frida's call (2026-10-04): delete, don't archive. This replaces the earlier
"nothing is deleted" rule in ch3_repo_triage.md for these groups only:

  1. data/ch3/ stale fit versions: *_B.csv, *_C.csv, *_D.csv, *_D_withDOY.csv,
     *_E.csv (the canonical files are now daily_fitted.csv / annual_params.csv
     / rmse_summary.csv, per ch3_config.py)
  2. archive/ch3/outputs_pre_E/ and archive/ch3/outputs_pre_1979fix/ --
     old pipeline OUTPUTS (data and figures), whole folders. Old SCRIPTS in
     archive/ch3/pipelines/ etc. are NOT touched: they're the provenance record.
  3. the stray figures/ folder inside scripts/.../Ch3/figures/, where three
     schematic scripts wrote before 2026-10-04 (wrong output path, now fixed)
  4. *.bak files left by the patch_*.py scripts

SAFETY
  - Dry run by default: lists everything, deletes nothing.
        python cleanup_old_files.py            # show
        python cleanup_old_files.py --delete   # do it
  - A file is SKIPPED if any live script (.py/.R/.sh outside archive/) still
    reads it -- a non-comment line naming the file together with read_csv,
    os.path.join, file.path, open(, fread, etc. Plain mentions in comments or
    error messages are listed but don't block.
  - Files tracked by git are removed with `git rm` (recoverable from git
    history). Untracked files (data/ch3 is mostly .gitignored) are deleted
    permanently -- the dry run marks each one so you can see which is which.
"""
import fnmatch
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import ROOT, DATA_DIR

DELETE = "--delete" in sys.argv
STALE_DATA_PATTERNS = ["*_B.csv", "*_C.csv", "*_D.csv", "*_D_withDOY.csv", "*_E.csv"]
ARCHIVE_OUTPUT_DIRS = ["archive/ch3/outputs_pre_E", "archive/ch3/outputs_pre_1979fix"]
STRAY_FIG_DIR = os.path.join(HERE, "figures")
STRAY_FIG_STEMS = ["fig1_cycle_mechanisms", "fig_phase_adjustment", "fig_level_leak"]
READ_HINT = re.compile(r"read_csv|read\.csv|read_excel|fread|readRDS|os\.path\.join|file\.path|open\(|load\(|np\.load",
                       re.I)
SKIP_DIRS = {".git", "archive", "__pycache__", ".ipynb_checkpoints", "node_modules"}


def rel(p):
    return os.path.relpath(p, ROOT)


# ── 1. collect candidates ───────────────────────────────────────────────────
cands = []  # (group, path, is_dir)
if os.path.isdir(DATA_DIR):
    for f in sorted(os.listdir(DATA_DIR)):
        if any(fnmatch.fnmatch(f, pat) for pat in STALE_DATA_PATTERNS):
            cands.append(("1 stale fit version", os.path.join(DATA_DIR, f), False))
for d in ARCHIVE_OUTPUT_DIRS:
    p = os.path.join(ROOT, d)
    if os.path.isdir(p):
        cands.append(("2 archived old outputs", p, True))
if os.path.isdir(STRAY_FIG_DIR):
    for f in sorted(os.listdir(STRAY_FIG_DIR)):
        if any(f.startswith(s) and f.endswith(".png") for s in STRAY_FIG_STEMS):
            cands.append(("3 stray figure copy", os.path.join(STRAY_FIG_DIR, f), False))
for dp, dns, fns in os.walk(ROOT):
    dns[:] = [d for d in dns if d not in SKIP_DIRS]
    for f in fns:
        if f.endswith(".bak"):
            cands.append(("4 patch backup", os.path.join(dp, f), False))

if not cands:
    print("Nothing to clean up -- no stale files found.")
    sys.exit(0)

# ── 2. reference guard: does any live script still read these? ─────────────
live = []
for dp, dns, fns in os.walk(ROOT):
    dns[:] = [d for d in dns if d not in SKIP_DIRS]
    for f in fns:
        if f.endswith((".py", ".R", ".r", ".sh", ".Rmd")) and os.path.join(dp, f) != os.path.abspath(__file__):
            live.append(os.path.join(dp, f))


def references(name):
    reads, mentions = [], []
    for s in live:
        try:
            lines = open(s, encoding="utf-8", errors="ignore").read().splitlines()
        except OSError:
            continue
        for i, ln in enumerate(lines, 1):
            if name in ln:
                where = f"{rel(s)}:{i}"
                if ln.lstrip().startswith("#") or not READ_HINT.search(ln):
                    mentions.append(where)
                else:
                    reads.append(where)
    return reads, mentions


def tracked(path):
    try:
        r = subprocess.run(["git", "-C", ROOT, "ls-files", "--error-unmatch", rel(path)],
                           capture_output=True, text=True)
        return r.returncode == 0
    except FileNotFoundError:
        return False


def size(p, is_dir):
    if not is_dir:
        return os.path.getsize(p)
    return sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fns in os.walk(p) for f in fns)


# ── 3. report / act ─────────────────────────────────────────────────────────
print(f"repo root: {ROOT}")
print("DELETING" if DELETE else "DRY RUN -- nothing will be deleted (add --delete to do it)")
total, skipped, done = 0, [], 0
group = None
for g, p, is_dir in cands:
    if g != group:
        group = g
        print(f"\n== group {g}")
    reads, mentions = ([], []) if is_dir else references(os.path.basename(p))
    if reads:
        skipped.append(p)
        print(f"  SKIP  {rel(p)}\n        still read by: {', '.join(reads[:4])}")
        continue
    t = tracked(p)
    mb = size(p, is_dir) / 1e6
    total += mb
    tag = "git rm (recoverable from history)" if t else "PERMANENT (not in git)"
    print(f"  {'del ' if DELETE else 'would'} {rel(p)}{'/' if is_dir else ''}  [{mb:.1f} MB, {tag}]")
    if mentions:
        print(f"        mentioned (comment/message only, not blocking): {', '.join(mentions[:3])}"
              f"{' ...' if len(mentions) > 3 else ''}")
    if DELETE:
        if t:
            subprocess.run(["git", "-C", ROOT, "rm", "-r", "-q", rel(p)], check=True)
        elif is_dir:
            shutil.rmtree(p)
        else:
            os.remove(p)
        done += 1

if DELETE and os.path.isdir(STRAY_FIG_DIR) and not os.listdir(STRAY_FIG_DIR):
    os.rmdir(STRAY_FIG_DIR)
    print(f"\n  removed empty folder {rel(STRAY_FIG_DIR)}/")

print(f"\n{'deleted' if DELETE else 'would delete'} {done if DELETE else len(cands) - len(skipped)} item(s), "
      f"{total:.1f} MB; skipped {len(skipped)} still in use.")
if skipped:
    print("Skipped files are still read by a live script -- point that script at the canonical file "
          "(ch3_config.DAILY_CSV / ANNUAL_CSV / RMSE_CSV) and rerun.")
if DELETE:
    print("If anything was git-removed, commit it:  git commit -m 'Remove superseded fit versions and stale copies'")
