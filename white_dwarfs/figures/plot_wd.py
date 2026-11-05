"""
plot_wd.py — WD figures for the MESA Colors paper.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.ticker import LogLocator, LogFormatter
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

HISTORY_PATH = Path("../LOGS/history.data")
OUT_DIR      = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

SMOOTH_SIGMA = 3

mpl.rcParams.update({
    "font.family":         "serif",
    "font.size":           10,
    "axes.labelsize":      11,
    "axes.titlesize":      11,
    "legend.fontsize":     9,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.top":           True,
    "ytick.right":         True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "figure.dpi":          150,
})

def read_history(path):
    with open(path) as f:
        lines = f.readlines()
    header_line = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("model_number") or "star_age" in stripped:
            header_line = i
            data_start  = i + 1
            break
    if header_line is None:
        raise ValueError("Could not find header in history.data")
    cols = lines[header_line].split()
    rows = []
    for line in lines[data_start:]:
        s = line.strip()
        if s and not s.startswith("#"):
            rows.append([float(x) for x in s.split()])
    data = np.array(rows)
    return {c: data[:, i] for i, c in enumerate(cols)}

print(f"Reading {HISTORY_PATH} ...")
h = read_history(HISTORY_PATH)
print(f"Available columns ({len(h)}): {', '.join(list(h.keys())[:20])} ...")

age      = h["star_age"]
log_Teff = h["log_Teff"]
Teff     = 10**log_Teff

bands = ["u", "g", "r", "i", "z", "y"]
band_colors = {
    "u": "#7b2d8b",
    "g": "#2196F3",
    "r": "#4CAF50",
    "i": "#FF9800",
    "z": "#F44336",
    "y": "#8B0000",
}

missing = [b for b in bands if b not in h]
if missing:
    print(f"\nWARNING: photometry columns not found: {missing}")
    import sys; sys.exit(0)

mag = {b: h[b] for b in bands}
cooling_age = age - age[0]
g_minus_r   = mag["g"] - mag["r"]

# ── Figure 1: CMD ─────────────────────────────────────────────────────────────
print("Plotting CMD ...")
fig, ax = plt.subplots(figsize=(3.5, 4.5))

log_cool = np.log10(np.clip(cooling_age, 1e3, None))

# Use a power-norm to pull more colour contrast out of the distribution.
# Most of the track IS old — so we compress the old end and stretch the young.
norm = mpl.colors.PowerNorm(gamma=0.9, vmin=3.0, vmax=log_cool.max())
cmap = cm.plasma

points = np.array([g_minus_r, mag["g"]]).T.reshape(-1, 1, 2)
segs   = np.concatenate([points[:-1], points[1:]], axis=1)
lc     = LineCollection(segs, cmap=cmap, norm=norm, linewidth=1.5, zorder=2)
lc.set_array(log_cool[:-1])
ax.add_collection(lc)

ax.set_xlim(g_minus_r.min() - 0.05, g_minus_r.max() + 0.05)
ax.set_ylim(mag["g"].max() + 0.5,   mag["g"].min() - 0.5)

cbar = fig.colorbar(lc, ax=ax, pad=0.0)
cbar.set_label(r"$\log_{10}$(cooling age / yr)")
# Put explicit ticks so the colorbar is readable with the power norm
cbar.set_ticks([3, 4, 5, 6, 7, 8, 9])

ax.set_xlabel(r"$(g - r)_{\rm AB}$")
ax.set_ylabel(r"$M_g$ (AB mag)")

fig.tight_layout()
out = OUT_DIR / "wd_cmd.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"  saved {out}")
plt.close(fig)

# ── Figure 2: multi-band light curves ─────────────────────────────────────────
print("Plotting light curves ...")
fig, ax = plt.subplots(figsize=(4.5, 3.5))

cool_gyr = cooling_age / 1e9
for b in bands:
    ax.plot(cool_gyr, mag[b], color=band_colors[b], lw=1.2, label=f"${b}$")

ax.invert_yaxis()
ax.set_xlabel("Cooling age (Gyr)")
ax.set_ylabel(r"$M$ (AB mag)")
ax.legend(ncol=3, framealpha=0.8)

fig.tight_layout()
out = OUT_DIR / "wd_lightcurves.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"  saved {out}")
plt.close(fig)

# ── Figure 3: color–temperature relations ─────────────────────────────────────
print("Plotting color-temperature relations ...")

color_pairs = [
    ("u", "g", "#7b2d8b"),
    ("g", "r", "#2196F3"),
    ("r", "i", "#4CAF50"),
    ("i", "z", "#FF9800"),
    ("z", "y", "#F44336"),
]

fig, ax = plt.subplots(figsize=(4.5, 3.5))

for b1, b2, c in color_pairs:
    raw      = mag[b1] - mag[b2]
    smoothed = gaussian_filter1d(raw, sigma=SMOOTH_SIGMA)
    ax.plot(Teff, smoothed, color=c, lw=1.2, label=f"$({b1}-{b2})$")

ax.set_xscale("log")
ax.invert_xaxis()

# Explicit ticks so at least 3 show on the log axis
ax.set_xticks([5000, 10000, 20000, 40000])
ax.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(
    lambda x, _: f"{int(x):,}"))
ax.tick_params(axis='x', which='minor', bottom=False, top=False)

ax.set_xlabel(r"$T_{\rm eff}$ (K)")
ax.set_ylabel(r"color (AB mag)")
ax.legend(ncol=2, framealpha=0.8)

fig.tight_layout()
out = OUT_DIR / "wd_color_evolution.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"  saved {out}")
plt.close(fig)

print("Done.")
