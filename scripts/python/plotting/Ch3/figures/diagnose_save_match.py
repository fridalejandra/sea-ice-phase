"""
diagnose_save_match.py — find exactly why the save() block doesn't match.
Run from figures/ directory: python3 diagnose_save_match.py
"""
import difflib

EXPECTED = '''def save(fig, name, dpi=300, sync=True, tight=True):
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
    return path
'''

with open("ch3_plot.py", encoding="utf-8") as f:
    content = f.read()

idx = content.find("def save(fig, name")
if idx == -1:
    print("Couldn't even find 'def save(fig, name' as a substring at all!")
    raise SystemExit(1)

actual_slice = content[idx: idx + len(EXPECTED) + 40]

print(f"EXPECTED length: {len(EXPECTED)}")
print(f"Found 'def save' at file offset {idx}")
print()

mismatch_at = None
for i, (a, b) in enumerate(zip(EXPECTED, actual_slice)):
    if a != b:
        mismatch_at = i
        break

if mismatch_at is None:
    if len(actual_slice) < len(EXPECTED):
        print("Actual file is SHORTER than expected at this point -- "
              "the function is cut off early (missing lines?).")
    else:
        print("No char mismatch found in the overlapping range -- "
              "weird. Possibly the EXPECTED block matches after all; "
              "re-run the real patch script.")
else:
    lo = max(0, mismatch_at - 40)
    print(f"FIRST MISMATCH at position {mismatch_at} (within the save() block):")
    print()
    print("  expected context:", repr(EXPECTED[lo:mismatch_at + 20]))
    print("  actual context:  ", repr(actual_slice[lo:mismatch_at + 20]))
    print()
    print(f"  expected char: {EXPECTED[mismatch_at]!r} ({hex(ord(EXPECTED[mismatch_at]))})")
    print(f"  actual char:   {actual_slice[mismatch_at]!r} ({hex(ord(actual_slice[mismatch_at]))})")

print()
print("=== unified diff (expected vs actual slice) ===")
diff = difflib.unified_diff(
    EXPECTED.splitlines(keepends=True),
    actual_slice[:len(EXPECTED) + 20].splitlines(keepends=True),
    fromfile="expected", tofile="actual",
)
print("".join(diff))
