#!/usr/bin/env python3
"""Recompute the archived Alpha Cen comparison with Colors-matched responses.

Run with the MSG virtual environment and MSG_DIR set. The existing comparison
CSV model selections are retained; only the MSG magnitudes are recalculated.
"""

import csv
import hashlib
import json
import os
import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MSG = Path(os.environ["MSG_DIR"]).expanduser().resolve()
GRID = ROOT / "msg_extended_tracks/kurucz2003_colors/sg-Kurucz2003all.h5"
LOCAL = ROOT / "msg_extended_tracks/kurucz2003_colors/matched_passbands"
COLORS_DATA = Path.home() / "MESA/mesa/data/colors_data"
VEGA_FILE = COLORS_DATA / "stellar_models/vega_flam.csv"
FILTER_DIR = COLORS_DATA / "filters/Generic/Johnson"
MAKE_PASSBAND = MSG / "bin/make_passband"
DISTANCE_CM = 10.0 * 3.085677581491367e18
GAUSS_X, GAUSS_W = np.polynomial.legendre.leggauss(8)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def spline_values(x, y, dy, query):
    idx = np.searchsorted(x, query, side="right") - 1
    idx = np.clip(idx, 0, len(x) - 2)
    h = x[idx + 1] - x[idx]
    t = (query - x[idx]) / h
    return ((2*t**3 - 3*t**2 + 1) * y[idx]
            + (t**3 - 2*t**2 + t) * h * dy[idx]
            + (-2*t**3 + 3*t**2) * y[idx + 1]
            + (t**3 - t**2) * h * dy[idx + 1])


def integrate_spline(x, y, dy, left, right, vega_x=None, vega_f=None):
    """Integrate S or Vega(lambda)*S using MSG's stored cubic Hermite spline."""
    breaks = x[(x > left) & (x < right)]
    if vega_x is not None:
        breaks = np.concatenate((breaks, vega_x[(vega_x > left) & (vega_x < right)]))
    edges = np.concatenate(([left], np.unique(breaks), [right]))
    lo, hi = edges[:-1], edges[1:]
    q = ((hi - lo)[:, None] * GAUSS_X[None, :] + (hi + lo)[:, None]) / 2
    s = spline_values(x, y, dy, q)
    if vega_x is None:
        f = 1.0
    else:
        f = np.interp(q, vega_x, vega_f)
    return float(np.sum((hi-lo)[:, None] / 2 * GAUSS_W[None, :] * s * f))


def make_local_passbands():
    import h5py

    LOCAL.mkdir(parents=True, exist_ok=True)
    vega = np.genfromtxt(VEGA_FILE, delimiter=",", names=True)
    vega_wave, vega_flux = vega["WAVELENGTH"], vega["FLUX"]
    outputs = {}
    manifest = {
        "vega_reference": str(VEGA_FILE),
        "vega_sha256": digest(VEGA_FILE),
        "passbands": {},
        "response_definition": "S(lambda) = lambda * Colors Johnson transmission",
        "zero_point_definition": "integral(Vega F_lambda * S d_lambda) / integral(S d_lambda), using the exact MSG cubic response interpolant and the Colors Vega reference spectrum",
    }
    for band in ("B", "V"):
        source = FILTER_DIR / f"{band}.dat"
        values = np.genfromtxt(source, delimiter=",", names=True)
        wavelength = values["Wavelength"]
        response = wavelength * values["Transmission"]
        response_file = LOCAL / f"Johnson_{band}_lambda_times_T.dat"
        np.savetxt(response_file, np.column_stack((wavelength, response)),
                   fmt="%.12g")

        provisional = LOCAL / f".provisional_{band}.h5"
        target = LOCAL / f"Johnson_{band}_Vega_matched.h5"
        subprocess.run([str(MAKE_PASSBAND), str(response_file), str(provisional),
                        "--zero-point=1.0"], check=True)
        with h5py.File(provisional, "r") as h5:
            x = h5["cubint/x"][...]
            y = h5["cubint/f"][...]
            dy = h5["cubint/df_dx"][...]
        f0 = integrate_spline(x, y, dy, x[0], x[-1], vega_wave, vega_flux) / \
            integrate_spline(x, y, dy, x[0], x[-1])
        provisional.unlink()
        subprocess.run([str(MAKE_PASSBAND), str(response_file), str(target),
                        f"--zero-point={f0:.17g}"], check=True)
        with h5py.File(target, "r") as h5:
            actual_x = h5["cubint/x"][...]
            actual_s = h5["cubint/f"][...]
            actual_f0 = float(h5.attrs["F_0"])
        if not (np.array_equal(actual_x, wavelength)
                and np.allclose(actual_s, response, rtol=0, atol=1e-12)
                and actual_f0 == f0):
            raise RuntimeError(f"Local {band} response or Vega zero point failed verification")

        manifest["passbands"][band] = {
            "colors_filter_input": str(source),
            "colors_filter_sha256": digest(source),
            "response_table": str(response_file),
            "response_table_sha256": digest(response_file),
            "msg_passband_h5": str(target),
            "passband_sha256": digest(target),
            "vega_zero_point_F_0": f0,
            "vega_zero_point_units": "mean F_lambda in the Colors Vega SED wavelength/flux units",
            "wavelength_unit": "Angstrom, unchanged from the Colors passband table",
        }
        outputs[band] = target
        print(f"Verified local {band}: F_0={f0:.12e}; response=lambda*T; "
              f"HDF5={target}")

    (LOCAL / "passband_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return outputs


def run_component(component, passbands):
    import pymsg

    name = f"alpha_cen_{component.lower()}"
    archive = ROOT / name / "archive" / f"{name}_dynamic_best"
    track_path = archive / f"{name}_msg_track.csv"
    comparison_path = archive / f"{name}_colors_msg_comparison.csv"
    track = np.genfromtxt(track_path, names=True, delimiter=",")
    old = np.genfromtxt(comparison_path, names=True, delimiter=",")
    selected = set(np.rint(old["model_number"]).astype(int))
    track_models = np.rint(track["model_number"]).astype(int)
    use = np.flatnonzero(np.isin(track_models, list(selected)))
    if len(use) != len(old) or set(track_models[use]) != selected:
        raise RuntimeError(f"{name}: current comparison model selection does not align with track CSV")

    phot_b = pymsg.PhotGrid(str(GRID), str(passbands["B"]))
    phot_v = pymsg.PhotGrid(str(GRID), str(passbands["V"]))
    columns = ("model_number", "star_age_yr", "Teff_K", "log_g", "radius_cm",
               "FeH", "Colors_B", "Colors_V", "Colors_B_minus_V", "MSG_B", "MSG_V",
               "MSG_B_minus_V", "delta_B", "delta_V", "delta_B_minus_V")
    rows = []
    for n, i in enumerate(use, start=1):
        feh = float(track["metallicity_dex"][i])
        params = {"Teff": float(track["Teff_K"][i]), "[Fe/H]": feh,
                  "log(g)": float(track["log_g"][i])}
        for band, phot in (("B", phot_b), ("V", phot_v)):
            if any(not (phot.axis_x_min[k] <= v <= phot.axis_x_max[k])
                   for k, v in params.items()):
                raise ValueError(f"{name}: model {track_models[i]} outside Kurucz MSG grid")
        scale = (float(track["radius_cm"][i]) / DISTANCE_CM) ** 2
        msg_b, msg_v = (-2.5 * np.log10(scale * p.flux(params))
                        for p in (phot_b, phot_v))
        cb, cv = float(track["Colors_B"][i]), float(track["Colors_V"][i])
        cbv = float(track["Colors_B_minus_V"][i])
        mbv = msg_b - msg_v
        rows.append([track_models[i], track["star_age_yr"][i], track["Teff_K"][i],
                     track["log_g"][i], track["radius_cm"][i], feh, cb, cv, cbv,
                     msg_b, msg_v, mbv, msg_b-cb, msg_v-cv, mbv-cbv])
        if n == 1 or n % 500 == 0 or n == len(use):
            print(f"Alpha Cen {component}: {n}/{len(use)} selected models", flush=True)

    rows = np.asarray(rows)
    with comparison_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"Wrote {comparison_path} ({len(rows)} rows; preserved existing model selection)")
    for label, values in (("delta B", rows[:, 12]), ("delta V", rows[:, 13]),
                          ("delta B-V", rows[:, 14])):
        print(f"Alpha Cen {component} {label}: median={np.median(values):+.6f}, "
              f"rms={np.sqrt(np.mean(values**2)):.6f}, min={np.min(values):+.6f}, "
              f"max={np.max(values):+.6f}")
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-passbands-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_passbands_only:
        make_local_passbands()
        return
    if not GRID.is_file() or not MAKE_PASSBAND.is_file():
        raise FileNotFoundError("Kurucz MSG grid or MSG make_passband executable is missing")
    # The MSG virtualenv supplies pymsg but not h5py; use system Python to
    # create and inspect the HDF5 passbands, then return here for photometry.
    subprocess.run(["python3", str(Path(__file__).resolve()), "--prepare-passbands-only"],
                   check=True)
    passbands = {band: LOCAL / f"Johnson_{band}_Vega_matched.h5" for band in ("B", "V")}
    for component in ("A", "B"):
        run_component(component, passbands)


if __name__ == "__main__":
    main()
