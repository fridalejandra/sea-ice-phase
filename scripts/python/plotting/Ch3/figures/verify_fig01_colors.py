"""
verify_fig01_colors.py — pixel-level check that fig01's two exports actually
use the cblind palette and have the right backgrounds. Run from figures/
(needs Pillow: pip install pillow, or it's already a matplotlib dependency).
"""
import os
from PIL import Image

OUTDIR = os.path.join("..", "..", "..", "..", "..", "results", "ch3", "figures")
MANUSCRIPT = os.path.join(OUTDIR, "fig01_concept_manuscript.png")
PRESENTATION = os.path.join(OUTDIR, "fig01_concept_presentation.png")

TARGETS = {
    "C_INV (black)":       (0x00, 0x00, 0x00),
    "C_PHASE (vermillion)": (0xD5, 0x5E, 0x00),
    "C_AMP (blue)":         (0x00, 0x72, 0xB2),
}
TOL = 6  # per-channel tolerance for anti-aliasing


def close(a, b, tol=TOL):
    return all(abs(a[i] - b[i]) <= tol for i in range(3))


def scan_for_colors(path, has_alpha_bg=False):
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    px = img.load()
    found = {name: 0 for name in TARGETS}
    examples = {name: None for name in TARGETS}

    for y in range(0, h, 2):        # stride 2 for speed
        for x in range(0, w, 2):
            r, g, b, a = px[x, y]
            if a < 200:              # skip near-transparent pixels
                continue
            for name, target in TARGETS.items():
                if close((r, g, b), target):
                    found[name] += 1
                    if examples[name] is None:
                        examples[name] = (x, y, (r, g, b, a))

    print(f"\n--- {os.path.basename(path)} ({w}x{h}) ---")
    for name in TARGETS:
        n = found[name]
        status = "FOUND" if n > 5 else "NOT FOUND"
        ex = f"  e.g. at {examples[name][:2]} = {examples[name][2]}" if examples[name] else ""
        print(f"  {name}: {status} ({n} matching px){ex}")

    # corner check for background
    corner = px[2, 2]
    print(f"  corner pixel (2,2) = {corner}", end="  ")
    if has_alpha_bg:
        print("(alpha should be ~0 for a transparent presentation export)")
    else:
        print("(should be opaque white ~(255,255,255,255) for the manuscript export)")


if not os.path.exists(MANUSCRIPT) or not os.path.exists(PRESENTATION):
    print("Can't find one or both PNGs at:")
    print(" ", MANUSCRIPT)
    print(" ", PRESENTATION)
    raise SystemExit(1)

scan_for_colors(MANUSCRIPT, has_alpha_bg=False)
scan_for_colors(PRESENTATION, has_alpha_bg=True)
