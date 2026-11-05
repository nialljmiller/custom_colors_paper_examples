#!/usr/bin/env python3
"""
Produce publication-quality figure and LaTeX table for the CALSPEC validation.

Figure: residual (synth - published, mmag) vs Teff for each tested band.
Table:  per-band summary statistics (N, median, |median|, RMS, max).

Inputs:
  calspec_residuals.csv
  published_photometry.csv
Outputs:
  figures/calspec_residuals.pdf / .png
  table_calspec_residuals.tex
  table_calspec_residuals.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 11,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
})

BASE = Path(__file__).resolve().parent

BANDS = [
    ("V",         "Johnson $V$",       "energy"),
    ("B",         "Johnson $B$",       "energy"),
    ("G",         r"Gaia $G$",         "photon"),
    ("Gbp",       r"Gaia $G_{\rm BP}$", "photon"),
    ("Grp",       r"Gaia $G_{\rm RP}$", "photon"),
    ("J_2mass",   "2MASS $J$",         "photon"),
    ("H_2mass",   "2MASS $H$",         "photon"),
    ("Ks_2mass",  "2MASS $K_s$",       "photon"),
]


def main():
    df = pd.read_csv(BASE / "calspec_residuals.csv")
    pub = pd.read_csv(BASE / "published_photometry.csv")
    # Merge to pick up published uncertainties for error bars
    df = df.merge(
        pub[["calspec_prefix"] + [c for c in pub.columns if c.endswith("_err")]],
        on="calspec_prefix", how="left"
    )

    # ------------------------------------------------------------------
    # LaTeX + CSV table
    # ------------------------------------------------------------------
    rows_all = []
    rows_rel = []
    reliable = {"bd_17d4708", "g191b2b", "gd71", "gd153", "hd209458",
                "hz43", "feige34", "feige110", "p330e", "p177d"}
    for band, label, dt in BANDS:
        col = f"{band}_resid_mmag"
        qcol = f"{band}_quality"
        # All stars, OK quality
        if qcol in df.columns:
            v_all = df.loc[df[qcol] == "ok", col].dropna()
        else:
            v_all = df[col].dropna()
        # Reliable set
        v_rel = df.loc[df["calspec_prefix"].isin(reliable) &
                       (df.get(qcol, "ok") == "ok"), col].dropna()

        def stats(v):
            if len(v) == 0:
                return (0, np.nan, np.nan, np.nan, np.nan)
            return (len(v), v.median(), v.abs().median(),
                    float(np.sqrt((v**2).mean())), v.abs().max())

        n, med, am, rms, mx = stats(v_all)
        rows_all.append(dict(band=band, label=label, dt=dt,
                             N=n, median=med, abs_median=am, rms=rms, max_abs=mx))
        n, med, am, rms, mx = stats(v_rel)
        rows_rel.append(dict(band=band, label=label, dt=dt,
                             N=n, median=med, abs_median=am, rms=rms, max_abs=mx))

    table_all = pd.DataFrame(rows_all)
    table_rel = pd.DataFrame(rows_rel)
    table_all.to_csv(BASE / "table_calspec_residuals_all.csv", index=False)
    table_rel.to_csv(BASE / "table_calspec_residuals_reliable.csv", index=False)

    # LaTeX table (reliable set; most appropriate for paper)
    with open(BASE / "table_calspec_residuals.tex", "w") as f:
        f.write(r"\begin{table}" "\n")
        f.write(r"\centering" "\n")
        f.write(r"\caption{CALSPEC synthetic photometry residuals (synthetic $-$ published) "
                r"across the primary filter sets exercised in this work. The reliable set "
                r"excludes stars with saturated catalog photometry or incomplete spectral "
                r"coverage. Residuals are in mmag.}" "\n")
        f.write(r"\label{tab:calspec_residuals}" "\n")
        f.write(r"\begin{tabular}{llrrrrr}" "\n")
        f.write(r"\hline\hline" "\n")
        f.write(r"Band & Convention & $N$ & median & $|$median$|$ & RMS & max$|\Delta|$ \\" "\n")
        f.write(r"\hline" "\n")
        for r in rows_rel:
            if r["N"] == 0:
                continue
            f.write(f"{r['label']} & {r['dt']} & {r['N']:d} & "
                    f"{r['median']:+.1f} & {r['abs_median']:.1f} & "
                    f"{r['rms']:.1f} & {r['max_abs']:.1f} \\\\\n")
        f.write(r"\hline" "\n")
        f.write(r"\end{tabular}" "\n")
        f.write(r"\end{table}" "\n")
    print(f"Wrote {BASE/'table_calspec_residuals.tex'}")

    # ------------------------------------------------------------------
    # Figure: residual vs Teff for each band (grid of 8 panels)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.3), sharex=True)
    axes = axes.flatten()

    for ax, (band, label, dt) in zip(axes, BANDS):
        col = f"{band}_resid_mmag"
        qcol = f"{band}_quality"
        # Select valid data
        mask = df[col].notna()
        if qcol in df.columns:
            mask_ok = mask & (df[qcol] == "ok")
            mask_bad = mask & (df[qcol] != "ok")
        else:
            mask_ok = mask
            mask_bad = pd.Series(False, index=df.index)

        # pub uncertainty for error bars (mmag)
        published_col = band if band in ["V", "B", "G", "Gbp", "Grp"] else band
        pub_err_col = published_col.replace("_mmag", "") + "_err"
        pub_err_col = f"{published_col}_err" if f"{published_col}_err" in df.columns else None

        teff = df["Teff_lit"].to_numpy()
        resid = df[col].to_numpy()
        yerr = df[pub_err_col].to_numpy() * 1000.0 if pub_err_col else np.zeros_like(teff)

        # reliable vs other
        is_rel = df["calspec_prefix"].isin(reliable).to_numpy()
        sel_rel = mask_ok.to_numpy() & is_rel
        sel_oth = mask_ok.to_numpy() & ~is_rel

        ax.axhline(0, color="k", lw=0.5, alpha=0.6)
        ax.errorbar(teff[sel_rel], resid[sel_rel], yerr=np.where(np.isfinite(yerr[sel_rel]), yerr[sel_rel], 0),
                    fmt='o', ms=5, mec='k', mew=0.5, color="#0072B2",
                    elinewidth=0.6, capsize=0, zorder=5, label="reliable" if ax is axes[0] else None)
        ax.errorbar(teff[sel_oth], resid[sel_oth], yerr=np.where(np.isfinite(yerr[sel_oth]), yerr[sel_oth], 0),
                    fmt='s', ms=4, mec='k', mew=0.4, color="#D55E00",
                    elinewidth=0.5, capsize=0, zorder=4, alpha=0.7,
                    label="other" if ax is axes[0] else None)
        ax.errorbar(teff[mask_bad.to_numpy()], resid[mask_bad.to_numpy()],
                    fmt='x', ms=6, color="gray", mew=1.0, zorder=3, alpha=0.5,
                    label="pub saturated" if ax is axes[0] else None)

        v_rel = resid[sel_rel]
        if len(v_rel) > 0:
            med = np.median(np.abs(v_rel))
            ax.text(0.03, 0.03, f"$|$med$|$ = {med:.0f} mmag\n$N$ = {len(v_rel)}",
                    transform=ax.transAxes, fontsize=9, va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.5))

        ax.set_xscale("log"); ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_title(f"{label}  [{dt}]", fontsize=11)
        ax.set_ylim(-250, 250)
        ax.grid(True, which="major", alpha=0.25, lw=0.4)

    for ax in axes[4:]:
        ax.set_xlabel(r"$T_{\rm eff}$ (K)", fontsize=11)
    for i in (0, 4):
        axes[i].set_ylabel(r"$\Delta m = m_{\rm synth} - m_{\rm pub}$ (mmag)", fontsize=11)

    axes[0].legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9)

    fig.tight_layout()
    out_pdf = BASE / "figures" / "calspec_residuals.pdf"
    out_png = BASE / "figures" / "calspec_residuals.png"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")

    # Print summary
    print("\n=== LaTeX table preview ===")
    with open(BASE / "table_calspec_residuals.tex") as f:
        print(f.read())


if __name__ == "__main__":
    main()
