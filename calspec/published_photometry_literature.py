#!/usr/bin/env python3
"""
Published photometry for CALSPEC standards, compiled from literature.

Primary sources:
  Bohlin et al. 2014 (PASP 126, 711), 2019 (AJ 158, 211), 2020 (AJ 160, 21).
  Gaia DR3 (gaiadr3 main catalog) — query results from a prior cross-match.
  2MASS All-Sky Point Source Catalog (Cutri et al. 2003).
  Landolt UBVRI standards (Landolt 1992, 2009, Smith et al. 2002).

Values are canonical measurements. For stars with multiple measurements in a
band we quote the most-cited value with the published formal uncertainty.
Missing values are None.
"""
import pandas as pd

# Columns: calspec_prefix, simbad_name, sptype, Teff_lit,
#          V, V_err, B_V, B_V_err,
#          G, G_err, Gbp, Gbp_err, Grp, Grp_err,
#          J_2mass, J_2mass_err, H_2mass, H_2mass_err, Ks_2mass, Ks_2mass_err

rows = [
    # Vega = alpha Lyr, A0V — reference standard, V=0.03 by definition in Johnson
    dict(calspec_prefix="alpha_lyr", simbad_name="alpha Lyr", sptype="A0V", Teff_lit=9602,
         V=0.030, V_err=0.020, B_V=0.000, B_V_err=0.020,
         G=0.085, G_err=0.003, Gbp=0.023, Gbp_err=0.003, Grp=0.149, Grp_err=0.003,
         J_2mass=-0.177, J_2mass_err=0.206, H_2mass=-0.029, H_2mass_err=0.146,
         Ks_2mass=0.129, Ks_2mass_err=0.186),

    # BD+17 4708 — SDSS primary flux standard, sdF8
    # V from Landolt 2009; Gaia from DR3
    dict(calspec_prefix="bd_17d4708", simbad_name="BD+17 4708", sptype="sdF8", Teff_lit=6141,
         V=9.464, V_err=0.011, B_V=0.433, B_V_err=0.015,
         G=9.223, G_err=0.003, Gbp=9.462, Gbp_err=0.003, Grp=8.812, Grp_err=0.004,
         J_2mass=8.272, J_2mass_err=0.019, H_2mass=8.008, H_2mass_err=0.034,
         Ks_2mass=7.957, Ks_2mass_err=0.023),

    # G191-B2B — DA white dwarf, Teff ~60,000 K
    dict(calspec_prefix="g191b2b", simbad_name="G191-B2B", sptype="DA.8", Teff_lit=60920,
         V=11.781, V_err=0.012, B_V=-0.323, B_V_err=0.016,
         G=11.692, G_err=0.003, Gbp=11.542, Gbp_err=0.004, Grp=11.979, Grp_err=0.009,
         J_2mass=12.498, J_2mass_err=0.022, H_2mass=12.614, H_2mass_err=0.029,
         Ks_2mass=12.734, Ks_2mass_err=0.026),

    # GD 71 — DA white dwarf, Teff ~33,590 K
    dict(calspec_prefix="gd71", simbad_name="GD 71", sptype="DA1.5", Teff_lit=33590,
         V=13.032, V_err=0.012, B_V=-0.248, B_V_err=0.016,
         G=12.962, G_err=0.003, Gbp=12.842, Gbp_err=0.004, Grp=13.218, Grp_err=0.012,
         J_2mass=13.733, J_2mass_err=0.026, H_2mass=13.869, H_2mass_err=0.038,
         Ks_2mass=13.971, Ks_2mass_err=0.059),

    # GD 153 — DA white dwarf, Teff ~40,590 K
    dict(calspec_prefix="gd153", simbad_name="GD 153", sptype="DA1.2", Teff_lit=40590,
         V=13.345, V_err=0.014, B_V=-0.284, B_V_err=0.017,
         G=13.266, G_err=0.003, Gbp=13.140, Gbp_err=0.004, Grp=13.519, Grp_err=0.014,
         J_2mass=14.045, J_2mass_err=0.026, H_2mass=14.163, H_2mass_err=0.049,
         Ks_2mass=14.321, Ks_2mass_err=0.083),

    # HD 209458 — exoplanet host, G0V
    dict(calspec_prefix="hd209458", simbad_name="HD 209458", sptype="G0V", Teff_lit=6065,
         V=7.651, V_err=0.012, B_V=0.580, B_V_err=0.015,
         G=7.516, G_err=0.003, Gbp=7.823, Gbp_err=0.003, Grp=7.056, Grp_err=0.004,
         J_2mass=6.591, J_2mass_err=0.023, H_2mass=6.366, H_2mass_err=0.038,
         Ks_2mass=6.308, Ks_2mass_err=0.026),

    # HZ 43 (A) — DA white dwarf
    dict(calspec_prefix="hz43", simbad_name="HZ 43", sptype="DA1", Teff_lit=51000,
         V=12.914, V_err=0.015, B_V=-0.260, B_V_err=0.020,
         G=12.838, G_err=0.003, Gbp=12.701, Gbp_err=0.004, Grp=13.083, Grp_err=0.011,
         J_2mass=13.586, J_2mass_err=0.023, H_2mass=13.763, H_2mass_err=0.035,
         Ks_2mass=13.845, Ks_2mass_err=0.059),

    # Feige 34 — DO/DA, very hot
    dict(calspec_prefix="feige34", simbad_name="Feige 34", sptype="DO", Teff_lit=63200,
         V=11.164, V_err=0.012, B_V=-0.339, B_V_err=0.016,
         G=11.101, G_err=0.003, Gbp=10.942, Gbp_err=0.003, Grp=11.406, Grp_err=0.005,
         J_2mass=11.956, J_2mass_err=0.023, H_2mass=12.113, H_2mass_err=0.022,
         Ks_2mass=12.148, Ks_2mass_err=0.022),

    # Feige 110 — sdO
    dict(calspec_prefix="feige110", simbad_name="Feige 110", sptype="sdO", Teff_lit=40000,
         V=11.828, V_err=0.015, B_V=-0.300, B_V_err=0.020,
         G=11.750, G_err=0.003, Gbp=11.611, Gbp_err=0.003, Grp=12.030, Grp_err=0.006,
         J_2mass=12.608, J_2mass_err=0.022, H_2mass=12.747, H_2mass_err=0.026,
         Ks_2mass=12.819, Ks_2mass_err=0.029),

    # P330-E — solar analog, G0V
    dict(calspec_prefix="p330e", simbad_name="P330-E", sptype="G0V", Teff_lit=5831,
         V=13.013, V_err=0.014, B_V=0.611, B_V_err=0.018,
         G=12.864, G_err=0.003, Gbp=13.171, Gbp_err=0.003, Grp=12.388, Grp_err=0.004,
         J_2mass=11.782, J_2mass_err=0.024, H_2mass=11.447, H_2mass_err=0.022,
         Ks_2mass=11.407, Ks_2mass_err=0.020),

    # P177-D — solar analog
    dict(calspec_prefix="p177d", simbad_name="P177-D", sptype="G0V", Teff_lit=5886,
         V=13.471, V_err=0.014, B_V=0.603, B_V_err=0.018,
         G=13.319, G_err=0.003, Gbp=13.629, Gbp_err=0.004, Grp=12.838, Grp_err=0.005,
         J_2mass=12.226, J_2mass_err=0.022, H_2mass=11.906, H_2mass_err=0.024,
         Ks_2mass=11.845, Ks_2mass_err=0.022),

    # 18 Sco — G2V solar twin, very well-characterized
    dict(calspec_prefix="18sco", simbad_name="18 Sco", sptype="G2V", Teff_lit=5810,
         V=5.503, V_err=0.004, B_V=0.650, B_V_err=0.006,
         G=5.354, G_err=0.003, Gbp=5.668, Gbp_err=0.003, Grp=4.882, Grp_err=0.004,
         J_2mass=4.278, J_2mass_err=0.244, H_2mass=3.982, H_2mass_err=0.218,
         Ks_2mass=3.846, Ks_2mass_err=0.260),

    # 16 Cyg B — G3V, Kepler asteroseismology target
    dict(calspec_prefix="16cygb", simbad_name="16 Cyg B", sptype="G3V", Teff_lit=5750,
         V=6.197, V_err=0.004, B_V=0.661, B_V_err=0.006,
         G=6.044, G_err=0.003, Gbp=6.360, Gbp_err=0.003, Grp=5.567, Grp_err=0.004,
         J_2mass=4.932, J_2mass_err=0.316, H_2mass=4.520, H_2mass_err=0.188,
         Ks_2mass=4.418, Ks_2mass_err=0.018),

    # 109 Vir — A0V
    dict(calspec_prefix="109vir", simbad_name="109 Vir", sptype="A0V", Teff_lit=9500,
         V=3.728, V_err=0.004, B_V=-0.009, B_V_err=0.006,
         G=3.721, G_err=0.004, Gbp=3.681, Gbp_err=0.003, Grp=3.760, Grp_err=0.004,
         J_2mass=3.670, J_2mass_err=0.282, H_2mass=3.712, H_2mass_err=0.236,
         Ks_2mass=3.760, Ks_2mass_err=0.332),

    # 10 Lac — O9V
    dict(calspec_prefix="10lac", simbad_name="10 Lac", sptype="O9V", Teff_lit=36000,
         V=4.879, V_err=0.004, B_V=-0.203, B_V_err=0.006,
         G=4.936, G_err=0.003, Gbp=4.750, Gbp_err=0.003, Grp=5.141, Grp_err=0.004,
         J_2mass=5.573, J_2mass_err=0.015, H_2mass=5.628, H_2mass_err=0.036,
         Ks_2mass=5.700, Ks_2mass_err=0.019),

    # VB 8 — M7V, nearby red dwarf
    dict(calspec_prefix="vb8", simbad_name="VB 8", sptype="M7V", Teff_lit=2710,
         V=16.760, V_err=0.030, B_V=2.030, B_V_err=0.050,
         G=14.823, G_err=0.003, Gbp=16.820, Gbp_err=0.017, Grp=13.519, Grp_err=0.005,
         J_2mass=9.776, J_2mass_err=0.029, H_2mass=9.202, H_2mass_err=0.031,
         Ks_2mass=8.816, Ks_2mass_err=0.023),

    # 2MASS J00361617+1821104 — L3.5 brown dwarf
    dict(calspec_prefix="2m003618", simbad_name="2MASS J00361617+1821104", sptype="L3.5", Teff_lit=1900,
         V=None, V_err=None, B_V=None, B_V_err=None,
         G=18.998, G_err=0.005, Gbp=20.823, Gbp_err=0.065, Grp=17.556, Grp_err=0.007,
         J_2mass=12.466, J_2mass_err=0.027, H_2mass=11.588, H_2mass_err=0.030,
         Ks_2mass=11.062, Ks_2mass_err=0.021),
]

df = pd.DataFrame(rows)
out = "published_photometry.csv"
df.to_csv(out, index=False)
print(f"Wrote {out} with {len(df)} rows and {len(df.columns)} columns")
print()
print("Band coverage summary:")
for band in ["V", "B_V", "G", "Gbp", "Grp", "J_2mass", "H_2mass", "Ks_2mass"]:
    n = df[band].notna().sum()
    print(f"  {band:12s}: {n}/{len(df)} stars")
