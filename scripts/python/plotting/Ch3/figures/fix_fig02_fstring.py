"""
fix_fig02_fstring.py — repairs the one broken fragment left by
patch_fig02_ice_edge.py. That patch's own `raise SystemExit(f"Not found:
{MEAN_ICE_CONC_NC}\n" ...)` line got mangled in transcription: the intended
backslash-n escape became an actual line break, splitting the f-string
across two lines with no closing quote -- a SyntaxError at py_compile time.

This touches ONLY that one fragment. It does not re-touch (and will not
re-match) any of the other 4 edits from that patch -- the expanded imports,
the two draw_ice_edge(ax) calls, or the legend handle -- which the diff
already confirmed are correct.

Uses chr(92) to build the replacement's backslash rather than a literal
escape sequence in this script's own source, specifically to avoid another
round of the same transcription bug.

Run from figures/patches/: python3 fix_fig02_fstring.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.dirname(HERE)
TARGET = os.path.join(FIGURES_DIR, "fig_02_sector_map.py")

with open(TARGET, "r", encoding="utf-8") as f:
    content = f.read()

BACKSLASH = chr(92)

broken = (
    '    raise SystemExit(\n'
    '        f"Not found: {MEAN_ICE_CONC_NC}\n'
    '"\n'
    '        "Run compute/compute_mean_ice_edge_climatology.py first (one-off, "\n'
    '        "reads the 13GB merged concentration file and caches a small result)."\n'
    '    )'
)

fixed = (
    '    raise SystemExit(\n'
    '        f"Not found: {MEAN_ICE_CONC_NC}' + BACKSLASH + 'n"\n'
    '        "Run compute/compute_mean_ice_edge_climatology.py first (one-off, "\n'
    '        "reads the 13GB merged concentration file and caches a small result)."\n'
    '    )'
)

if broken == fixed:
    raise SystemExit("Internal error: broken and fixed are identical -- aborting.")

n = content.count(broken)
if n != 1:
    already = content.count(fixed)
    if already == 1:
        print("Skipping — already fixed (found the correct version in place).")
        raise SystemExit(0)
    raise SystemExit(
        f"Expected exactly 1 match for the broken fragment, found {n}. "
        "File doesn't match what this fix expects -- aborting without "
        "writing anything. Open fig_02_sector_map.py around the "
        "'Not found: {MEAN_ICE_CONC_NC}' line and compare by hand."
    )

content = content.replace(broken, fixed, 1)

with open(TARGET, "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed the f-string in fig_02_sector_map.py.")
print("Now run: python3 -m py_compile ../fig_02_sector_map.py")
