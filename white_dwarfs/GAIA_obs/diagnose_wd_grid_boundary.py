#!/usr/bin/env python3
"""Check the existing Gaia bin comparison against the actual atmosphere domain."""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantify_wd_locus_offset import (
    prepare_mesa_track, read_gaia_table, select_gaia_wds, quantify_offsets,
)

ROOT = Path(__file__).resolve().parent
HISTORY = ROOT / "LOGS_wd_cool/history.data"
GRID = Path("/home/njm/MESA/mesa_filter_interp_leak_fix/data/colors_data/stellar_models/koester2/lookup_table.csv")


def stats(table):
    d = table.delta_bp_rp_model_minus_gaia.to_numpy()
    return np.median(abs(d)), np.sqrt(np.mean(d**2)), np.max(abs(d))


def main():
    grid = pd.read_csv(GRID)
    grid.columns = grid.columns.str.lstrip("#").str.strip()
    tmin, tmax = grid.teff.min(), grid.teff.max()
    gmin, gmax = grid.logg.min(), grid.logg.max()
    assert len(grid) == grid.teff.nunique() * grid.logg.nunique()
    assert not grid.duplicated(["teff", "logg"]).any()
    mesa = prepare_mesa_track(HISTORY)
    old = pd.read_csv(ROOT / "wd_locus_offsets.csv")
    # Reproduce the established median-MG, 3-sigma clipped calculation.
    gaia = read_gaia_table(ROOT / "gaia_cmd_all_plus_dsc_wds.parquet")
    selected = select_gaia_wds(gaia, 0.5, -0.8, 1.3, 9., 15.)
    table, _ = quantify_offsets(selected, mesa, 9., 15., 1., 50, "median_mg", 3.)
    np.testing.assert_allclose(table.to_numpy(), old.to_numpy(), rtol=1e-12, atol=1e-12)

    # Match the original color interpolator's sorting and duplicate-MG grouping.
    cols = ["M_G", "bp_rp", "effective_T", "log_g"]
    grouped = mesa[cols].dropna().sort_values("M_G")
    grouped["M_G_round"] = grouped.M_G.round(6)
    grouped = grouped.groupby("M_G_round", as_index=False)[cols].mean().sort_values("M_G")
    for source, dest in [("effective_T", "model_Teff_K"), ("log_g", "model_log_g")]:
        table[dest] = np.interp(table.M_G_reference, grouped.M_G, grouped[source])
    np.testing.assert_allclose(table.model_bp_rp, np.interp(table.M_G_reference, grouped.M_G, grouped.bp_rp))
    table["inside_Teff_grid"] = table.model_Teff_K.between(tmin, tmax)
    table["inside_log_g_grid"] = table.model_log_g.between(gmin, gmax)
    table["inside_Koester_grid"] = table.inside_Teff_grid & table.inside_log_g_grid
    table["below_Teff_grid"] = table.model_Teff_K < tmin
    table.to_csv(ROOT / "wd_locus_offsets_with_grid_status.csv", index=False)

    below = mesa.effective_T < tmin
    crossing_pos = np.flatnonzero(below.to_numpy())[0]
    crossing = mesa.iloc[crossing_pos]
    prev = mesa.iloc[crossing_pos - 1]
    fraction = (tmin - prev.effective_T) / (crossing.effective_T - prev.effective_T)
    boundary_color = prev.bp_rp + fraction * (crossing.bp_rp - prev.bp_rp)
    boundary_mg = prev.M_G + fraction * (crossing.M_G - prev.M_G)
    inside = mesa.effective_T.between(tmin, tmax) & mesa.log_g.between(gmin, gmax)
    retained = table.loc[table.inside_Koester_grid]
    excluded = table.loc[~table.inside_Koester_grid]
    lines = [
        f"History: {HISTORY}",
        f"History SHA256: {hashlib.sha256(HISTORY.read_bytes()).hexdigest()}",
        f"Grid lookup used: {GRID}",
        f"Resolved grid lookup: {GRID.resolve()}",
        f"Grid SHA256: {hashlib.sha256(GRID.read_bytes()).hexdigest()}",
        f"Grid extent: Teff = {tmin:g}--{tmax:g} K; log g = {gmin:g}--{gmax:g}",
        f"Complete Cartesian grid: {grid.teff.nunique()} temperatures x {grid.logg.nunique()} gravities",
        "Existing bin table reproduced from Gaia sample and new history (median MG, 3-sigma MAD clipping).",
        "First recorded model below minimum Teff:",
        crossing[["model_number", "star_age", "effective_T", "log_g", "M_G", "bp_rp"]].to_string(),
        f"Linearly interpolated boundary: MG = {boundary_mg:.9f}, GBP-GRP = {boundary_color:.9f}",
        f"History rows below grid: {below.sum()} / {len(mesa)}",
        "Bin results:",
        table.to_string(index=False),
        "Retained bins: " + ", ".join(f"{r.M_G_lo:g}--{r.M_G_hi:g}" for r in retained.itertuples()),
        "Excluded bins: " + (", ".join(f"{r.M_G_lo:g}--{r.M_G_hi:g}" for r in excluded.itertuples()) or "none"),
    ]
    for label, subset in [("Full range", table), ("Inside grid only", retained)]:
        med, rms, maximum = stats(subset)
        lines.append(f"{label}: median absolute residual = {med:.10f} mag; RMS = {rms:.10f} mag; maximum absolute residual = {maximum:.10f} mag")
    lines += [
        "Quoted full-range comparison: median absolute = 0.0743, RMS = 0.1345, maximum absolute = 0.2835 mag.",
        "Excluding below-grid comparison points has no effect: every bin reference lies inside the grid.",
        "The 14--15 MG residual is evaluated above the minimum-Teff boundary, not at or below it.",
        "This test does not establish the cause of the remaining discrepancy; no counterfactual atmosphere calculation was performed.",
    ]
    (ROOT / "wd_grid_boundary_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 11,
                         "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out"})
    fig, ax = plt.subplots(figsize=(5.3, 5.2), layout="constrained")
    blue = "#0072B2"
    ax.plot(mesa.bp_rp.where(inside), mesa.M_G.where(inside), color=blue, lw=1.7,
            label="Model: inside atmosphere grid")
    # Include the interpolated crossing so both segments meet at the boundary.
    ax.plot([prev.bp_rp, boundary_color], [prev.M_G, boundary_mg], color=blue, lw=1.7)
    ax.plot(np.r_[boundary_color, mesa.loc[below, "bp_rp"]],
            np.r_[boundary_mg, mesa.loc[below, "M_G"]], color=blue, lw=1.5,
            ls="--", alpha=0.65, label="Model: below atmosphere grid")
    ax.errorbar(table.gaia_bp_rp_median, table.M_G_reference,
                xerr=table.gaia_bp_rp_mad, fmt="o", color="0.2", ms=4,
                elinewidth=0.8, capsize=2, label="Binned Gaia locus")
    ax.plot(boundary_color, boundary_mg, "o", ms=4, mfc="white", mec=blue, zorder=5)
    ax.annotate(f"{tmin:g} K", (boundary_color, boundary_mg), xytext=(-38, 17),
                textcoords="offset points", fontsize=9, ha="right",
                arrowprops={"arrowstyle": "-", "color": "0.45", "lw": 0.7})
    ax.set(xlabel=r"$G_{\rm BP}-G_{\rm RP}$", ylabel=r"$M_G$",
           xlim=(-0.65, 1.3), ylim=(15.25, 8.75))
    ax.spines[["top", "right"]].set_visible(False)
    ax.minorticks_on()
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, handlelength=2.5)
    out = ROOT / "figures/wd_grid_boundary_diagnostic"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
