"""Reproduce fig_extinction_coefficients.pdf (Section 3.6).
Band extinction A_X = m_X(A_V) - m_X(0) from SED_Model 0.4.7 (pip install sed-model==0.4.7):
SED_Model reddens the SED (CCM89, R_V = 3.1) and integrates through the Custom Colors Fortran (cc_api).
Grid: MESA data/colors_data/stellar_models/Kurucz2003all; filters: GAIA/GAIA and Generic/Johnson; Vega system.
Evaluated at grid nodes only ([M/H]=0; log g = 4.5 for 3500-30000 K; log g = 2.0 for 3500-9000 K, where SED_Model reports no empty-node cells)."""
import numpy as np, pandas as pd, sed_model
D = "/home/njm/MESA/mesa/data/colors_data"
grid = sed_model.load_grid(f"{D}/stellar_models/Kurucz2003all")
vega = f"{D}/stellar_models/vega_flam.csv"
fl = [f for f in sed_model.load_filters_from_instrument_dir(f"{D}/filters/GAIA/GAIA", vega_sed_path=vega) if f.name in ("G", "Gbp", "Grp")]
fl += [f for f in sed_model.load_filters_from_instrument_dir(f"{D}/filters/Generic/Johnson", vega_sed_path=vega) if f.name in ("B", "V", "I", "J")]
R, d = 6.957e10, 10 * sed_model.PC_TO_CM
ext = {av: sed_model.make_extinction_model(enabled=True, law="ccm89", r_v=3.1, a_v=av) for av in (1.0, 3.0)}
rows = []
for logg, tmax in ((4.5, 30000), (2.0, 9000)):
    for T in grid.teff_grid[grid.teff_grid <= tmax]:
        r0 = sed_model.run_forward(T, logg, 0.0, R, d, grid, fl, mag_system="Vega")
        if r0.nearest_grid_distance > 1e-9 or r0.clamped: continue
        row = {"logg": logg, "Teff": T}
        for av, em in ext.items():
            r = sed_model.run_forward(T, logg, 0.0, R, d, grid, fl, mag_system="Vega", extinction=em)
            for b in r0.magnitudes: row[f"{b}_k{int(av)}"] = (r.magnitudes[b] - r0.magnitudes[b]) / av
        rows.append(row)
pd.DataFrame(rows).to_csv("ext_coeffs_sedmodel.csv", index=False)
