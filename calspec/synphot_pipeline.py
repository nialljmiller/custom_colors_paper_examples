#!/usr/bin/env python3
"""
CALSPEC synthetic photometry pipeline — using the actual Custom Colors
Fortran kernels via SED_Model's cc_api f2py wrappers.

This is critical: we do NOT reimplement the integration in Python.  Every
magnitude is computed by the same Fortran routines that run inside MESA
during stellar evolution, exposed through sed_model.cc_api.cc_api:
  * synthetic_magnitude(sed_wave, sed_flux, filt_wave, filt_trans, zp)
  * vega_zero_point   (vega_wave, vega_flux, filt_wave, filt_trans)
This way the test actually validates Custom Colors, not a Python surrogate.

Detector-type handling (photon vs energy counting) is applied exactly as in
Custom Colors' colors_utils.f90:load_filter: for an energy-counting filter
curve, the transmission is converted to its photon-counting equivalent
T_E / lambda (Bessell & Murphy 2012, eq. A9), rescaled to preserve the peak,
and then passed to the Fortran photon-counting integral.  This lets the
Fortran kernel run in its standard mode while still consuming an
energy-counting filter correctly.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits

from sed_model.cc_api import cc_api as f  # Fortran kernels

VEGA_SED = "/home/njm/MESA/mesa/data/colors_data/stellar_models/vega_flam.csv"

FILTERS = [
    ("G",         "G",         "/home/njm/MESA/mesa/data/colors_data/filters/GAIA/GAIA/G.dat"),
    ("Gbp",       "Gbp",       "/home/njm/MESA/mesa/data/colors_data/filters/GAIA/GAIA/Gbp.dat"),
    ("Grp",       "Grp",       "/home/njm/MESA/mesa/data/colors_data/filters/GAIA/GAIA/Grp.dat"),
    ("J_2mass",   "J_2mass",   "/home/njm/SED_Tools/data/filters/2MASS/2MASS/J.dat"),
    ("H_2mass",   "H_2mass",   "/home/njm/SED_Tools/data/filters/2MASS/2MASS/H.dat"),
    ("Ks_2mass",  "Ks_2mass",  "/home/njm/SED_Tools/data/filters/2MASS/2MASS/Ks.dat"),
    ("V",         "V",         "/home/njm/MESA/mesa/data/colors_data/filters/Generic/Johnson/V.dat"),
    ("B",         "B",         "/home/njm/MESA/mesa/data/colors_data/filters/Generic/Johnson/B.dat"),
]

RELIABLE_FOR_ALL_BANDS = {
    "bd_17d4708", "g191b2b", "gd71", "gd153", "hd209458", "hz43",
    "feige34", "feige110", "p330e", "p177d",
}

# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------

def read_calspec_spectrum(path):
    hdul = fits.open(path)
    try:
        data = hdul[1].data  # type: ignore[attr-defined]
        wl = np.asarray(data["WAVELENGTH"], dtype=float)
        fl = np.asarray(data["FLUX"], dtype=float)
    finally:
        hdul.close()
    m = np.isfinite(wl) & np.isfinite(fl) & (fl > 0)
    return wl[m], fl[m]


def read_filter(path):
    header = None
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",", 2)
            if len(parts) < 2:
                continue
            try:
                w, t = float(parts[0]), float(parts[1])
                rows.append((w, t))
            except ValueError:
                header = line
    wl = np.array([r[0] for r in rows], dtype=float)
    tr = np.array([r[1] for r in rows], dtype=float)
    detector_type = "photon"
    if header:
        for part in header.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                if k.strip().lower() == "detectortype":
                    detector_type = v.strip().lower()
    return wl, tr, detector_type


def apply_detector_type(wl_f, tr_f, detector_type):
    """Convert energy-counting T_E to photon-counting equivalent T_E/lambda,
    rescaled to preserve the peak.  Mirrors colors_utils.f90:load_filter."""
    if detector_type != "energy":
        return tr_f.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        tr_eff = np.where(wl_f > 0, tr_f / wl_f, 0.0)
    peak_in = tr_f.max()
    peak_out = tr_eff.max()
    if peak_out > 0:
        tr_eff = tr_eff * (peak_in / peak_out)
    return tr_eff


def read_vega():
    df = pd.read_csv(VEGA_SED, comment="#")
    cols = {c.lower().strip(): c for c in df.columns}
    wl_col = cols.get("wavelength") or list(df.columns)[0]
    fl_col = cols.get("flux") or cols.get("flam") or list(df.columns)[1]
    wl = df[wl_col].to_numpy(dtype=float)
    fl = df[fl_col].to_numpy(dtype=float)
    m = np.isfinite(wl) & np.isfinite(fl)
    return wl[m], fl[m]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    base = Path(__file__).resolve().parent
    spectra_dir = base / "spectra"
    published = pd.read_csv(base / "published_photometry.csv")
    if "B" not in published.columns and {"V", "B_V"} <= set(published.columns):
        published["B"] = published["V"] + published["B_V"]
        published["B_err"] = np.sqrt(published["V_err"]**2 + published["B_V_err"]**2)

    wl_vega, fl_vega = read_vega()

    # Compute Vega zero-point for each filter via Fortran kernel
    filter_cache = {}
    vega_zp = {}
    for band, _, fpath in FILTERS:
        wl_f, tr_f, dt = read_filter(fpath)
        tr_eff = apply_detector_type(wl_f, tr_f, dt)
        zp, ierr = f.vega_zero_point(wl_vega, fl_vega, wl_f, tr_eff)
        assert ierr == 0, f"vega_zero_point ierr={ierr} for {band}"
        filter_cache[band] = (wl_f, tr_f, tr_eff, dt)
        vega_zp[band] = zp
        print(f"Filter {band:10s} ({dt:6s}): Vega F_X = {zp:.4e}  [Fortran]")

    print()
    results = []
    for _, star in published.iterrows():
        prefix = star["calspec_prefix"]
        matches = sorted(spectra_dir.glob(f"{prefix}_*.fits"))
        if not matches:
            print(f"  SKIP {prefix}: no FITS file")
            continue
        try:
            wl, fl = read_calspec_spectrum(matches[0])
        except Exception as e:
            print(f"  ERR  {prefix}: {e}")
            continue

        row = {
            "calspec_prefix": prefix,
            "simbad_name":    star["simbad_name"],
            "sptype":         star["sptype"],
            "Teff_lit":       star["Teff_lit"],
            "n_wavelengths":  len(wl),
            "wl_min_AA":      wl.min(),
            "wl_max_AA":      wl.max(),
        }

        for band, pcol, _ in FILTERS:
            wl_f, tr_f, tr_eff, dt = filter_cache[band]

            # Coverage: fraction of lambda-weighted filter integral covered by spectrum
            cov_mask = (wl_f >= wl.min()) & (wl_f <= wl.max())
            full_integ = float(f.trapz(wl_f, tr_f * wl_f))
            cov_integ = float(f.trapz(wl_f[cov_mask], (tr_f * wl_f)[cov_mask])) if cov_mask.sum() > 1 else 0.0
            coverage = cov_integ / full_integ if full_integ > 0 else 0.0
            row[f"{band}_coverage"] = coverage

            if coverage < 0.99:
                row[f"{band}_synth"] = np.nan
                row[f"{band}_resid_mmag"] = np.nan
                row[f"{band}_quality"] = "no_coverage"
                continue

            # Call the actual Fortran kernel
            mag, bflux, ierr = f.synthetic_magnitude(wl, fl, wl_f, tr_eff, vega_zp[band])
            if ierr != 0 or not np.isfinite(mag):
                row[f"{band}_synth"] = np.nan
                row[f"{band}_resid_mmag"] = np.nan
                row[f"{band}_quality"] = f"fortran_ierr_{ierr}"
                continue

            row[f"{band}_synth"] = float(mag)

            if pcol is not None and pcol in star.index and pd.notna(star[pcol]):
                m_pub = float(star[pcol])
                err_col = f"{pcol}_err"
                m_err = float(star[err_col]) if err_col in star.index and pd.notna(star[err_col]) else None
                row[f"{band}_resid_mmag"] = (mag - m_pub) * 1000.0
                row[f"{band}_quality"] = "pub_noisy" if (m_err is not None and m_err > 0.050) else "ok"
            else:
                row[f"{band}_resid_mmag"] = np.nan
                row[f"{band}_quality"] = "no_pub"

        results.append(row)

    out = base / "calspec_residuals.csv"
    df = pd.DataFrame(results)
    df.to_csv(out, index=False)
    print(f"Wrote {out} with {len(df)} rows")

    # Summaries
    def report(label, mask):
        print(f"\n=== Residual summary (mmag) — {label} ===")
        print(f"{'Band':10s} {'N':>4s} {'med':>8s} {'|med|':>8s} {'rms':>8s} {'max|':>8s}")
        for band, _, _ in FILTERS:
            col = f"{band}_resid_mmag"
            qcol = f"{band}_quality"
            if col not in df.columns: continue
            m = mask & (df.get(qcol, "ok") == "ok")
            v = df.loc[m, col].dropna()
            if len(v) == 0:
                print(f"{band:10s} {0:>4d}   no match"); continue
            print(f"{band:10s} {len(v):>4d} {v.median():>8.2f} {v.abs().median():>8.2f} "
                  f"{np.sqrt((v**2).mean()):>8.2f} {v.abs().max():>8.2f}")

    report("ALL stars", pd.Series(True, index=df.index))
    report("reliable set", df["calspec_prefix"].isin(list(RELIABLE_FOR_ALL_BANDS)))


if __name__ == "__main__":
    main()
