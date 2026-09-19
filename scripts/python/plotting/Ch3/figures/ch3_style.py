"""
ch3_style.py -- the single source of visual style (fonts, colours, rcParams)
for every Ch3 figure. ch3_plot.py's own docstring already says style lives
here; this file didn't actually exist yet (confirmed 2026-09-18) -- that's
the real reason fonts weren't consistent: some figure scripts set their own
plt.rcParams.update({"font.size": 10, ...}) inline with no font.family at
all, others set nothing and just got matplotlib's default (DejaVu Sans).

Usage: add `import ch3_style` near the top of any figure script, right
after the ch3_config import (so it benefits from the same sys.path insert)
-- importing this module applies the style immediately, at import time, and
prints which font actually got resolved. If a figure script has its own
plt.rcParams.update({...}) call for font/spines, remove it -- this module
is now the one place that sets those, and a script-local override placed
after `import ch3_style` would silently win and put that one figure back
out of sync with the rest.
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Tries Helvetica first, then Tacoma, then Tahoma (in case "Tacoma" was a
# typo for the similarly-named, far more common system font -- ask Frida
# to confirm which one she actually wants if the resolved font matters),
# then two safe fallbacks so figures never silently revert to an ugly
# default even if none of the above are installed.
FONT_PREFERENCE = ["Helvetica", "Tacoma", "Tahoma", "Arial", "DejaVu Sans"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONT_PREFERENCE,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _resolved_font_name():
    """What font matplotlib will ACTUALLY use, not just what was requested --
    findfont() silently falls back to a default if nothing in
    FONT_PREFERENCE is installed, so this checks the real answer rather
    than trusting the rcParams just set above."""
    # No explicit family= here: bare FontProperties() reads the current
    # rcParams (font.family/font.sans-serif) that were just set above --
    # passing family="sans-serif" directly gets misparsed by matplotlib as
    # a fontconfig pattern string in newer versions and raises, rather than
    # being treated as the generic family name.
    path = fm.findfont(fm.FontProperties(), fallback_to_default=True)
    return fm.FontProperties(fname=path).get_name(), path


_name, _path = _resolved_font_name()
if _name.lower() not in ("helvetica", "tacoma"):
    print(f"  WARNING [ch3_style]: neither Helvetica nor Tacoma is available -- "
          f"matplotlib resolved to '{_name}' instead ({_path}). Figures will "
          f"render in this font until Helvetica or Tacoma is installed and "
          f"matplotlib's font cache is rebuilt (delete "
          f"~/.cache/matplotlib and re-run, or "
          f"matplotlib.font_manager._load_fontmanager(try_read_cache=False)).")
else:
    print(f"  [ch3_style] using font: {_name}")
