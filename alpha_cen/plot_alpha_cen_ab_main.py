#!/usr/bin/env python3
"""Main-paper figure for the live Alpha Cen A/B Custom Colors fits.

Run from ~/MESA/alpha_cen_colors, or pass that directory with --root.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TARGETS = {
    "A": {"mv": 4.38, "bv": 0.71, "sigma_mv": 0.03, "sigma_bv": 0.03},
    "B": {"mv": 5.71, "bv": 0.88, "sigma_mv": 0.03, "sigma_bv": 0.03},
}
COLORS = {"A": "#0072B2", "B": "#D55E00"}


def read_mesa_table(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return np.atleast_1d(
        np.genfromtxt(path, names=True, skip_header=5, dtype=None, encoding=None)
    )


def finite_rows(table, *columns):
    keep = np.ones(len(table), dtype=bool)
    for column in columns:
        keep &= np.isfinite(np.asarray(table[column], dtype=float))
    return table[keep]


def best_result(results):
    valid = finite_rows(results, "chi2", "alpha", "age", "M_V", "B_minus_V")
    return valid[np.argmin(valid["chi2"])]


def collapse_alpha(results):
    """Keep the best score for effectively repeated simplex alpha values."""
    valid = finite_rows(results, "alpha", "chi2", "age")
    order = np.argsort(valid["alpha"])
    valid = valid[order]
    groups = []
    for row in valid:
        if not groups or not np.isclose(
            row["alpha"], groups[-1]["alpha"], rtol=0.0, atol=1e-10
        ):
            groups.append(row)
        elif row["chi2"] < groups[-1]["chi2"]:
            groups[-1] = row
    return np.asarray(groups, dtype=valid.dtype)


def load_component(root, component):
    work = root / f"alpha_cen_{component.lower()}"
    archive = work / "archive" / f"alpha_cen_{component.lower()}_dynamic_best"
    history = read_mesa_table(archive / "LOGS" / "history.data")
    results = read_mesa_table(
        archive / f"alpha_cen_{component.lower()}_simplex_results.data"
    )
    return history, results


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(),
        help="Directory containing alpha_cen_a and alpha_cen_b",
    )
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    outdir = (args.outdir or root / "plots").expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    data = {}
    for component in ("A", "B"):
        history, results = load_component(root, component)
        data[component] = {
            "history": history,
            "results": results,
            "profile": collapse_alpha(results),
            "best": best_result(results),
        }

    plt.rcParams.update({
        "font.size": 9.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.5,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(7.15, 6.25), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=(1.18, 1.0))
    ax_cmd = fig.add_subplot(gs[0, :])
    ax_loss = fig.add_subplot(gs[1, 0])
    ax_age = fig.add_subplot(gs[1, 1])

    # (a) Best-track Johnson CMD.  Restrict to the fitted age interval.
    # Grey points are the best-age models returned by all trial tracks; the
    # complete history was retained only for the final best track.
    for component in ("A", "B"):
        results = finite_rows(
            data[component]["results"], "B_minus_V", "M_V", "alpha", "chi2"
        )
        order = np.argsort(results["alpha"])
        ax_cmd.plot(
            results["B_minus_V"][order], results["M_V"][order],
            color="0.78", linewidth=0.65, alpha=0.7, zorder=1,
        )
        ax_cmd.scatter(
            results["B_minus_V"], results["M_V"],
            s=9, color="0.62", alpha=0.62, linewidths=0, zorder=2,
        )

    for component in ("A", "B"):
        h = data[component]["history"]
        h = finite_rows(h, "star_age", "B", "V")
        h = h[h["star_age"] >= 2e9]
        bv = h["B"] - h["V"]
        mv = h["V"]
        target = TARGETS[component]
        best = data[component]["best"]
        color = COLORS[component]

        ax_cmd.plot(
            bv, mv, color=color, linewidth=1.6, alpha=0.82,
            label=rf"$\alpha$ Cen {component}: best-fit track",
        )
        ax_cmd.errorbar(
            target["bv"], target["mv"],
            xerr=target["sigma_bv"], yerr=target["sigma_mv"],
            fmt="*", markersize=8, color=color, markeredgecolor="black",
            markeredgewidth=0.55, capsize=2.0, elinewidth=0.9, zorder=6,
            label=rf"$\alpha$ Cen {component}: observed",
        )
        ax_cmd.scatter(
            best["B_minus_V"], best["M_V"], s=25, facecolors="white",
            edgecolors=color, linewidths=1.25, zorder=7,
        )

    ax_cmd.invert_yaxis()
    ax_cmd.set(
        xlabel=r"Johnson $B-V$ (mag)",
        ylabel=r"$M_V$ (mag)",
    )
    ax_cmd.legend(ncols=2, loc="best", columnspacing=1.0, handlelength=2.3)

    # (b) Profiled loss: each point is the best-age model on one full track.
    for component in ("A", "B"):
        profile = data[component]["profile"]
        best = data[component]["best"]
        color = COLORS[component]
        ax_loss.plot(
            profile["alpha"], profile["chi2"], "o-", color=color,
            markersize=3.8, linewidth=1.35, label=rf"$\alpha$ Cen {component}",
        )
        ax_loss.scatter(
            best["alpha"], best["chi2"], marker="*", s=115,
            color=color, edgecolor="black", linewidth=0.7, zorder=5,
        )
    ax_loss.set_yscale("log")
    ax_loss.set(
        xlabel=r"Mixing length $\alpha_{\rm MLT}$",
        ylabel=r"Profiled $\chi^2_{\rm phot}$",
    )
    ax_loss.legend()

    # (c) Age selected within each trial track.
    for component in ("A", "B"):
        profile = data[component]["profile"]
        best = data[component]["best"]
        color = COLORS[component]
        ax_age.plot(
            profile["alpha"], profile["age"] / 1e9, "o-", color=color,
            markersize=3.8, linewidth=1.35, label=rf"$\alpha$ Cen {component}",
        )
        ax_age.scatter(
            best["alpha"], best["age"] / 1e9, marker="*", s=115,
            color=color, edgecolor="black", linewidth=0.7, zorder=5,
        )
    ax_age.set(
        xlabel=r"Mixing length $\alpha_{\rm MLT}$",
        ylabel="Best-fitting age (Gyr)",
    )
    best_a = data["A"]["best"]
    best_b = data["B"]["best"]
    ax_age.text(
        0.97, 0.96,
        rf"A: $\alpha_{{\rm MLT}}={best_a['alpha']:.3f}$, "
        rf"$t={best_a['age']/1e9:.3f}$ Gyr" + "\n" +
        rf"B: $\alpha_{{\rm MLT}}={best_b['alpha']:.3f}$, "
        rf"$t={best_b['age']/1e9:.3f}$ Gyr",
        transform=ax_age.transAxes, ha="right", va="top", fontsize=8.2,
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "white",
              "edgecolor": "0.75", "alpha": 0.92},
        zorder=8,
    )

    fig.savefig(outdir / "fig_alpha_cen_runtime_fit.pdf", bbox_inches="tight")
    fig.savefig(
        outdir / "fig_alpha_cen_runtime_fit.png", dpi=300, bbox_inches="tight"
    )

    print("Saved:", outdir / "fig_alpha_cen_runtime_fit.pdf")
    for component in ("A", "B"):
        best = data[component]["best"]
        print(
            f"Alpha Cen {component}: alpha={best['alpha']:.12f}, "
            f"age={best['age']/1e9:.9f} Gyr, chi2={best['chi2']:.12e}, "
            f"M_V={best['M_V']:.9f}, B-V={best['B_minus_V']:.9f}"
        )


if __name__ == "__main__":
    main()
