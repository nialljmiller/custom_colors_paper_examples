#!/usr/bin/env python3
"""
Compile published photometry for CALSPEC standards from SIMBAD, Gaia DR3, 2MASS.

Output: published_photometry.csv with columns:
  calspec_file, simbad_name, ra, dec, Teff_lit,
  G, G_err, Gbp, Gbp_err, Grp, Grp_err,              # Gaia DR3
  J, J_err, H, H_err, Ks, Ks_err,                    # 2MASS
  V, V_err, B, B_err,                                # Johnson (if available)
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import pandas as pd

# Map CALSPEC file prefix -> SIMBAD identifier + nominal Teff from Bohlin+
# Teff values from Bohlin et al. (2014, 2019, 2024) tables
CALSPEC_STANDARDS = [
    # (calspec_prefix,   simbad_name,         sptype,   Teff_lit_K)
    ("alpha_lyr",        "alpha Lyr",         "A0V",    9550),
    ("bd_17d4708",       "BD+17 4708",        "sdF8",   6141),
    ("g191b2b",          "G191-B2B",          "DA.8",   60920),
    ("gd71",             "GD 71",             "DA1.5",  33590),
    ("gd153",             "GD 153",           "DA1.2",  40590),
    ("hd209458",          "HD 209458",        "G0V",    6065),
    ("hz43",              "HZ 43",            "DA1",    51000),
    ("feige34",           "Feige 34",         "DO",     63200),
    ("feige110",          "Feige 110",        "sdO",    40000),
    ("p330e",             "P330-E",           "G0V",    5831),
    ("p177d",             "P177-D",           "G0V",    5886),
    ("18sco",             "18 Sco",           "G2V",    5810),
    ("16cygb",            "16 Cyg B",         "G3V",    5750),
    ("109vir",            "109 Vir",          "A0V",    9500),
    ("10lac",             "10 Lac",           "O9V",    36000),
    ("vb8",               "VB 8",             "M7V",    2710),
    ("2m003618",          "2MASS J00361617+1821104", "L3.5", 1900),
]

# Configure SIMBAD to return many fields
Simbad.reset_votable_fields()
Simbad.add_votable_fields('otype', 'sp_type', 'flux(V)', 'flux_error(V)',
                          'flux(B)', 'flux_error(B)', 'flux(J)', 'flux_error(J)',
                          'flux(H)', 'flux_error(H)', 'flux(K)', 'flux_error(K)')
Simbad.TIMEOUT = 60


def query_simbad(name):
    """Query SIMBAD for coordinates + aggregated flux values."""
    try:
        res = Simbad.query_object(name)
        if res is None or len(res) == 0:
            return None
        row = res[0]
        # Some SIMBAD columns may differ in name between versions; handle both
        def getcol(r, *candidates, default=None):
            for c in candidates:
                if c in r.colnames and not np.ma.is_masked(r[c]):
                    val = r[c]
                    return float(val) if hasattr(val, '__float__') else val
            return default

        ra_str = getcol(row, 'RA', 'ra')
        dec_str = getcol(row, 'DEC', 'dec')
        if ra_str is None or dec_str is None:
            return None
        coord = SkyCoord(f"{ra_str} {dec_str}", unit=(u.hourangle, u.deg))

        return {
            "ra":  coord.ra.deg,
            "dec": coord.dec.deg,
            "V":   getcol(row, 'FLUX_V', 'flux_V'),
            "V_err": getcol(row, 'FLUX_ERROR_V', 'flux_error_V'),
            "B":   getcol(row, 'FLUX_B', 'flux_B'),
            "B_err": getcol(row, 'FLUX_ERROR_B', 'flux_error_B'),
            "J":   getcol(row, 'FLUX_J', 'flux_J'),
            "J_err": getcol(row, 'FLUX_ERROR_J', 'flux_error_J'),
            "H":   getcol(row, 'FLUX_H', 'flux_H'),
            "H_err": getcol(row, 'FLUX_ERROR_H', 'flux_error_H'),
            "Ks":  getcol(row, 'FLUX_K', 'flux_K'),
            "Ks_err": getcol(row, 'FLUX_ERROR_K', 'flux_error_K'),
        }
    except Exception as e:
        print(f"  SIMBAD error: {e}", file=sys.stderr)
        return None


def query_gaia_dr3(coord, radius_arcsec=5.0):
    """Query Gaia DR3 (VizieR I/355/gaiadr3) for G, BP, RP."""
    try:
        v = Vizier(columns=["Source", "RA_ICRS", "DE_ICRS",
                            "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag"])
        v.ROW_LIMIT = 5
        res = v.query_region(coord, radius=radius_arcsec * u.arcsec, catalog="I/355/gaiadr3")
        if not res or len(res) == 0:
            return None
        t = res[0]
        if len(t) == 0:
            return None
        # If multiple matches, take the brightest in G
        if len(t) > 1 and 'Gmag' in t.colnames:
            t.sort('Gmag')
        r = t[0]
        return {
            "G":   float(r['Gmag']) if not np.ma.is_masked(r['Gmag']) else None,
            "G_err": float(r['e_Gmag']) if not np.ma.is_masked(r['e_Gmag']) else None,
            "Gbp": float(r['BPmag']) if not np.ma.is_masked(r['BPmag']) else None,
            "Gbp_err": float(r['e_BPmag']) if not np.ma.is_masked(r['e_BPmag']) else None,
            "Grp": float(r['RPmag']) if not np.ma.is_masked(r['RPmag']) else None,
            "Grp_err": float(r['e_RPmag']) if not np.ma.is_masked(r['e_RPmag']) else None,
        }
    except Exception as e:
        print(f"  Gaia error: {e}", file=sys.stderr)
        return None


def query_2mass(coord, radius_arcsec=5.0):
    """Query 2MASS Point Source Catalog (VizieR II/246)."""
    try:
        v = Vizier(columns=["RAJ2000", "DEJ2000", "Jmag", "e_Jmag",
                            "Hmag", "e_Hmag", "Kmag", "e_Kmag"])
        v.ROW_LIMIT = 5
        res = v.query_region(coord, radius=radius_arcsec * u.arcsec, catalog="II/246/out")
        if not res or len(res) == 0:
            return None
        t = res[0]
        if len(t) == 0:
            return None
        if len(t) > 1 and 'Kmag' in t.colnames:
            t.sort('Kmag')
        r = t[0]
        return {
            "J_2mass":   float(r['Jmag']) if not np.ma.is_masked(r['Jmag']) else None,
            "J_2mass_err": float(r['e_Jmag']) if not np.ma.is_masked(r['e_Jmag']) else None,
            "H_2mass":   float(r['Hmag']) if not np.ma.is_masked(r['Hmag']) else None,
            "H_2mass_err": float(r['e_Hmag']) if not np.ma.is_masked(r['e_Hmag']) else None,
            "Ks_2mass":  float(r['Kmag']) if not np.ma.is_masked(r['Kmag']) else None,
            "Ks_2mass_err": float(r['e_Kmag']) if not np.ma.is_masked(r['e_Kmag']) else None,
        }
    except Exception as e:
        print(f"  2MASS error: {e}", file=sys.stderr)
        return None


def main():
    rows = []
    for prefix, name, sptype, teff in CALSPEC_STANDARDS:
        print(f"\n{prefix} ({name}, {sptype}, Teff={teff}K)")
        row = {
            "calspec_prefix": prefix,
            "simbad_name": name,
            "sptype": sptype,
            "Teff_lit": teff,
        }
        simbad = query_simbad(name)
        if simbad is None:
            print("  no SIMBAD match — skipping remote catalog queries")
            rows.append(row)
            continue
        row.update(simbad)
        print(f"  SIMBAD: ra={row['ra']:.4f} dec={row['dec']:.4f}  "
              f"V={row.get('V')} J={row.get('J')}")

        coord = SkyCoord(row['ra'] * u.deg, row['dec'] * u.deg)
        time.sleep(0.3)

        gaia = query_gaia_dr3(coord)
        if gaia:
            row.update(gaia)
            print(f"  Gaia DR3: G={gaia.get('G')} BP={gaia.get('Gbp')} RP={gaia.get('Grp')}")
        else:
            print("  Gaia: no match")
        time.sleep(0.3)

        tm = query_2mass(coord)
        if tm:
            row.update(tm)
            print(f"  2MASS: J={tm.get('J_2mass')} H={tm.get('H_2mass')} Ks={tm.get('Ks_2mass')}")
        else:
            print("  2MASS: no match")
        time.sleep(0.3)

        rows.append(row)

    df = pd.DataFrame(rows)
    out = "published_photometry.csv"
    df.to_csv(out, index=False)
    print(f"\n\nWrote {out} with {len(df)} rows and {len(df.columns)} columns")
    print("\nSummary of coverage:")
    for col in ['V', 'B', 'J', 'H', 'Ks', 'G', 'Gbp', 'Grp', 'J_2mass', 'H_2mass', 'Ks_2mass']:
        if col in df.columns:
            n = df[col].notna().sum()
            print(f"  {col:10s}: {n}/{len(df)} stars")


if __name__ == "__main__":
    main()
