#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
import pymsg


INPUT = Path(
    "archive/alpha_cen_a_dynamic_best/"
    "alpha_cen_a_msg_track.csv"
)
OUTPUT = Path(
    "archive/alpha_cen_a_dynamic_best/"
    "alpha_cen_a_colors_msg_comparison.csv"
)

FEH = 0.24
MIN_AGE = 2.0e9
PC = 3.085677581491367e18
DISTANCE = 10.0 * PC


track = np.genfromtxt(
    INPUT,
    names=True,
    delimiter=",",
)

grid_file = os.path.join(
    os.environ["MSG_DIR"],
    "data/grids/sg-CAP18-coarse.h5",
)
passband_dir = os.path.join(
    os.environ["MSG_DIR"],
    "data/passbands",
)

phot_b = pymsg.PhotGrid(
    grid_file,
    os.path.join(
        passband_dir,
        "pb-Generic-Johnson.B-Vega.h5",
    ),
)
phot_v = pymsg.PhotGrid(
    grid_file,
    os.path.join(
        passband_dir,
        "pb-Generic-Johnson.V-Vega.h5",
    ),
)

use = np.where(track["star_age_yr"] >= MIN_AGE)[0]
rows = []

for n, i in enumerate(use, start=1):
    x = {
        "Teff": float(track["Teff_K"][i]),
        "[Fe/H]": FEH,
        "log(g)": float(track["log_g"][i]),
    }

    scale = (
        float(track["radius_cm"][i]) / DISTANCE
    )**2

    msg_b = -2.5 * np.log10(
        scale * phot_b.flux(x)
    )
    msg_v = -2.5 * np.log10(
        scale * phot_v.flux(x)
    )
    msg_bv = msg_b - msg_v

    colors_b = float(track["Colors_B"][i])
    colors_v = float(track["Colors_V"][i])
    colors_bv = float(
        track["Colors_B_minus_V"][i]
    )

    rows.append([
        track["model_number"][i],
        track["star_age_yr"][i],
        track["Teff_K"][i],
        track["log_g"][i],
        track["radius_cm"][i],
        FEH,
        colors_b,
        colors_v,
        colors_bv,
        msg_b,
        msg_v,
        msg_bv,
        msg_b - colors_b,
        msg_v - colors_v,
        msg_bv - colors_bv,
    ])

    if n == 1 or n % 250 == 0 or n == len(use):
        print(f"{n:5d}/{len(use)} models")

rows = np.asarray(rows)

np.savetxt(
    OUTPUT,
    rows,
    delimiter=",",
    header=(
        "model_number,star_age_yr,Teff_K,log_g,"
        "radius_cm,FeH,"
        "Colors_B,Colors_V,Colors_B_minus_V,"
        "MSG_B,MSG_V,MSG_B_minus_V,"
        "delta_B,delta_V,delta_B_minus_V"
    ),
    comments="",
    fmt=[
        "%d",
        "%.12e",
        "%.8f",
        "%.8f",
        "%.12e",
        "%.8f",
        "%.10f",
        "%.10f",
        "%.10f",
        "%.10f",
        "%.10f",
        "%.10f",
        "%.10f",
        "%.10f",
        "%.10f",
    ],
)

print()
print("Wrote:", OUTPUT)

for name, values in (
    ("delta B", rows[:, 12]),
    ("delta V", rows[:, 13]),
    ("delta B-V", rows[:, 14]),
):
    print(
        f"{name:9s}: "
        f"median={np.median(values):+.6f}, "
        f"rms={np.sqrt(np.mean(values**2)):.6f}, "
        f"min={np.min(values):+.6f}, "
        f"max={np.max(values):+.6f}"
    )
