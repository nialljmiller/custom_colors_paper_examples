"""Compare extended Colors tracks against an MSG grid built from the same SEDs."""

import json
import os
from pathlib import Path

import numpy as np
import pymsg

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "kurucz2003_colors"
MSG = Path(os.environ["MSG_DIR"])
GRID = OUT / "sg-Kurucz2003all.h5"
DISTANCE_CM = 10 * 3.085677581491367e18

phot = [pymsg.PhotGrid(str(GRID), str(MSG / f"data/passbands/pb-Generic-Johnson.{b}-Vega.h5"))
        for b in ("B", "V")]

for component in ("a", "b"):
    name = f"alpha_cen_{component}"
    source = ROOT / name / f"{name}_extended_colors_track.csv"
    track = np.genfromtxt(source, delimiter=",", names=True)
    rows = []
    skipped = []
    for i, r in enumerate(track):
        params = {"Teff": r["Teff_K"], "[Fe/H]": r["metallicity_dex"], "log(g)": r["log_g"]}
        if not all(p.axis_x_min[k] <= params[k] <= p.axis_x_max[k] for p in phot for k in params):
            skipped.append(int(r["model_number"]))
            continue
        scale = (r["radius_cm"] / DISTANCE_CM) ** 2
        b, v = [-2.5 * np.log10(scale * p.flux(params)) for p in phot]
        rows.append([r[k] for k in ("model_number", "star_age_yr", "Teff_K", "log_g", "radius_cm", "metallicity_dex",
                                    "Colors_B", "Colors_V", "Colors_B_minus_V")] +
                    [b, v, b-v, b-r["Colors_B"], v-r["Colors_V"], (b-v)-r["Colors_B_minus_V"]])
        if i == 0 or (i + 1) % 500 == 0 or i + 1 == len(track):
            print(f"{name}: {i+1}/{len(track)} models", flush=True)
    rows = np.asarray(rows)
    assert len(rows) == len(track), f"Outside grid: {skipped[:10]}"
    path = OUT / f"{name}_extended_colors_msg_kurucz2003.csv"
    np.savetxt(path, rows, delimiter=",", comments="", fmt="%.16e",
               header="model_number,star_age_yr,Teff_K,log_g,radius_cm,FeH,Colors_B,Colors_V,Colors_B_minus_V,MSG_B,MSG_V,MSG_B_minus_V,delta_B,delta_V,delta_B_minus_V")
    summary = {"models": len(rows), "skipped": skipped, "statistics": {}}
    for label, values in zip(("delta_B", "delta_V", "delta_B_minus_V"), rows[:, 12:].T):
        summary["statistics"][label] = {"median": float(np.median(values)),
                                         "rms": float(np.sqrt(np.mean(values**2))),
                                         "min": float(values.min()), "max": float(values.max())}
    (OUT / f"{name}_kurucz_comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(name, json.dumps(summary, indent=2), sep="\n")
    print("Wrote", path)
