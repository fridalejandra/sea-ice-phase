"""
ch4_figstyle.py -- two render modes for every Ch4 figure:

    manuscript : white background, 7.2 in text width, 9-10 pt type, PNG 300 dpi + PDF
    slide      : transparent background, 13.3 x 6.5 in (16:9 slide body), 16-20 pt type, PNG 200 dpi

Builds on ch4_style.py (Helvetica, recessive ink, no top/right spines), which
must sit in the same directory. Usage in a figure script:

    import ch4_figstyle as fs
    fs.set_mode("slide")                 # or "manuscript"
    fig, ax = plt.subplots(figsize=fs.size(1.0, 0.75))
    ...
    fs.save(fig, "fig03_wind_vs_drift_two_products")

Sector colours: Okabe-Ito (cblind). It passes the CVD validator; the Ch3 SCAR
palette (ch4_style.SECTOR_COLORS) does not (yellow/orange DeltaE 9.7, green/orange
3.6 under protan). Flip USE_CH3_PALETTE = True to match Ch3 exactly anyway.
"""
import os
import matplotlib.pyplot as plt
import ch4_style as st  # noqa: F401  (applies base style on import)

ROOT = "/user/geog/falejandraperez/sea-ice-phase"
FIG_ROOT = f"{ROOT}/results/ch4/figures"

USE_CH3_PALETTE = False

# sector id (as in the sector-daily CSVs) -> short code, long name
SECTOR_ID = {1: "ABS", 2: "WS", 3: "KH", 4: "EA", 5: "RA"}   # ids as in canonical_sectors.nc (checked 2026-10-21)
SECTOR_NAME = {"WS": "Weddell", "ABS": "Amundsen–Bellingshausen", "RA": "Ross–Amundsen",
               "EA": "East Antarctica", "KH": "King Haakon VII"}
SECTOR_ORDER = ["WS", "KH", "EA", "RA", "ABS"]          # clockwise from Greenwich
_OKABE = {"WS": "#E69F00", "ABS": "#56B4E9", "RA": "#009E73", "EA": "#D55E00", "KH": "#CC79A7"}
SECTOR_COLORS = st.SECTOR_COLORS if USE_CH3_PALETTE else _OKABE

SEASON_MARKERS = {"DJF": "o", "MAM": "s", "JJA": "^", "SON": "D"}
GREY = "#8C8C8C"       # for wind lines and reference marks
BAND = "#E8E8E8"       # shaded highlight bands
INK = st.INK

_MODE = {"name": None}

_PRESETS = {
    "manuscript": dict(width=7.2, font=9, title=10, tick=8, legend=8, lw=1.6, ms=7,
                       dpi=300, transparent=False, formats=("png", "pdf")),
    "slide":      dict(width=13.3, font=16, title=18, tick=14, legend=14, lw=3.0, ms=12,
                       dpi=200, transparent=True, formats=("png",)),
}


def set_mode(name):
    p = _PRESETS[name]
    _MODE.update(name=name, **p)
    plt.rcParams.update({
        "font.size": p["font"], "axes.labelsize": p["font"], "axes.titlesize": p["title"],
        "xtick.labelsize": p["tick"], "ytick.labelsize": p["tick"],
        "legend.fontsize": p["legend"], "lines.linewidth": p["lw"],
        "lines.markersize": p["ms"], "savefig.dpi": p["dpi"],
        "savefig.transparent": p["transparent"],
        "axes.linewidth": 0.8 if name == "manuscript" else 1.2,
        "xtick.major.width": 0.8 if name == "manuscript" else 1.2,
        "ytick.major.width": 0.8 if name == "manuscript" else 1.2,
    })
    return p


def mode():
    return _MODE["name"]


def size(wfrac=1.0, aspect=0.62):
    """Figure size in inches: wfrac of the mode's full width, height = width*aspect."""
    w = _MODE["width"] * wfrac
    return (w, w * aspect)


def lw(k=1.0):
    return _MODE["lw"] * k


def ms(k=1.0):
    return _MODE["ms"] * k


def save(fig, stem, subdir=None):
    out_dir = os.path.join(FIG_ROOT, subdir or ("slides" if _MODE["name"] == "slide" else "manuscript"))
    os.makedirs(out_dir, exist_ok=True)
    for ext in _MODE["formats"]:
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
        print(f"  saved {path}")
    plt.close(fig)
