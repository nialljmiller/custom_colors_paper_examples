#!/usr/bin/env python3
"""Compare Custom Colors and MSG photometry along Alpha Cen tracks.

Run from ~/MESA/alpha_cen_colors, or pass that directory with --root.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TOOL_COLORS = {"Colors": "#0072B2", "MSG": "#D55E00"}
RESIDUAL_COLORS = {"B": "#0072B2", "V": "#D55E00", "B-V": "#009E73"}


def read_comparison(root, component):
    name = f"alpha_cen_{component.lower()}"
    path = (
        root / name / "archive" / f"{name}_dynamic_best"
        / f"{name}_colors_msg_comparison.csv"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.atleast_1d(np.genfromtxt(path, names=True, delimiter=","))
    required = (
        "star_age_yr", "Colors_B", "Colors_V", "Colors_B_minus_V",
        "MSG_B", "MSG_V", "MSG_B_minus_V", "delta_B", "delta_V",
        "delta_B_minus_V",
    )
    missing = [column for column in required if column not in data.dtype.names]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    keep = np.ones(len(data), dtype=bool)
    for column in required:
        keep &= np.isfinite(data[column])
    data = data[keep]
    return data[np.argsort(data["star_age_yr"])]


def read_extended_comparison(root, component):
    name = f"alpha_cen_{component.lower()}"
    path = (
        root / "msg_extended_tracks" / name
        / f"{name}_extended_colors_msg.csv"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.atleast_1d(np.genfromtxt(path, names=True, delimiter=","))
    required = (
        "star_age_yr", "Colors_B", "Colors_V", "Colors_B_minus_V",
        "MSG_B", "MSG_V", "MSG_B_minus_V", "delta_B", "delta_V",
        "delta_B_minus_V",
    )
    missing = [column for column in required if column not in data.dtype.names]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    keep = np.ones(len(data), dtype=bool)
    for column in required:
        keep &= np.isfinite(data[column])
    data = data[keep]
    return data[np.argsort(data["star_age_yr"])]


def read_kurucz_extended_comparison(root, component):
    name = f"alpha_cen_{component.lower()}"
    path = (
        root / "msg_extended_tracks" / "kurucz2003_colors"
        / f"{name}_extended_colors_msg_kurucz2003.csv"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.atleast_1d(np.genfromtxt(path, names=True, delimiter=","))
    required = (
        "star_age_yr", "Teff_K", "Colors_B", "Colors_V", "Colors_B_minus_V",
        "MSG_B", "MSG_V", "MSG_B_minus_V", "delta_B", "delta_V",
        "delta_B_minus_V",
    )
    missing = [column for column in required if column not in data.dtype.names]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    keep = np.ones(len(data), dtype=bool)
    for column in required:
        keep &= np.isfinite(data[column])
    return data[keep][np.argsort(data[keep]["star_age_yr"])]


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(),
        help="Directory containing alpha_cen_a and alpha_cen_b",
    )
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument(
        "--extended", action="store_true",
        help="Plot the extended validation tracks instead of fit-era tracks",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    default_outdir = root / "msg_extended_tracks" / "plots" if args.extended else root / "plots"
    outdir = (args.outdir or default_outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    reader = read_extended_comparison if args.extended else read_comparison
    tracks = {component: reader(root, component) for component in ("A", "B")}
    kurucz_tracks = (
        {component: read_kurucz_extended_comparison(root, component)
         for component in ("A", "B")}
        if args.extended else None
    )

    plt.rcParams.update({
        "font.size": 9.5,
        "axes.labelsize": 10.5,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(
        2, 2, figsize=(7.15, 6.15), layout="constrained",
        gridspec_kw={"height_ratios": (1.05, 1.0)},
    )

    for column, component in enumerate(("A", "B")):
        data = tracks[component]
        kurucz = kurucz_tracks[component] if args.extended else None
        ax_cmd = axes[0, column]
        ax_res = axes[1, column]

        ax_cmd.plot(
            data["Colors_B_minus_V"], data["Colors_V"],
            color=TOOL_COLORS["Colors"], linewidth=1.7,
            label="Custom Colors", zorder=3,
        )
        ax_cmd.plot(
            data["MSG_B_minus_V"], data["MSG_V"],
            color=TOOL_COLORS["MSG"], linewidth=1.45, linestyle="--",
            label="MSG CAP18" if args.extended else "MSG", zorder=2,
        )
        if args.extended:
            ax_cmd.plot(
                kurucz["MSG_B_minus_V"], kurucz["MSG_V"],
                color="#009E73", linewidth=1.35, linestyle="-.",
                label="MSG Kurucz2003all", zorder=2,
            )
        ax_cmd.invert_yaxis()
        ax_cmd.set_xlabel(r"Johnson $B-V$ (mag)")
        if column == 0:
            ax_cmd.set_ylabel(r"$M_V$ (mag)")
            ax_cmd.legend(loc="best")
        ax_cmd.text(
            0.97, 0.96, rf"$\alpha$ Cen {component}",
            transform=ax_cmd.transAxes, ha="right", va="top",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                  "edgecolor": "0.8", "alpha": 0.9},
        )

        x = data["Teff_K"] if args.extended else data["star_age_yr"] / 1e9
        for delta_column, color, label in (
            ("delta_B", RESIDUAL_COLORS["B"], r"$\Delta B$"),
            ("delta_V", RESIDUAL_COLORS["V"], r"$\Delta V$"),
            ("delta_B_minus_V", RESIDUAL_COLORS["B-V"], r"$\Delta(B-V)$"),
        ):
            ax_res.plot(x, data[delta_column], color=color, linewidth=1.25,
                        label=label + (" (CAP18)" if args.extended else ""))
            if args.extended:
                ax_res.plot(kurucz["Teff_K"], kurucz[delta_column], color=color,
                            linewidth=1.25, linestyle="--",
                            label=label + " (Kurucz2003all)")
        ax_res.set_ylim(bottom=0.0)
        ax_res.set_xlabel(r"Effective temperature $T_{\rm eff}$ (K)" if args.extended else "Age (Gyr)")
        if args.extended:
            ax_res.invert_xaxis()
        if column == 0:
            ax_res.set_ylabel(r"MSG $-$ Custom Colors (mag)")
            ax_res.legend(loc="best", ncols=2 if args.extended else 1,
                          fontsize=6.7 if args.extended else None)

    suffix = "_extended" if args.extended else ""
    pdf = outdir / f"fig_alpha_cen_msg_comparison{suffix}.pdf"
    png = outdir / f"fig_alpha_cen_msg_comparison{suffix}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print("Saved:", pdf)

    for component in ("A", "B"):
        data = tracks[component]
        print(f"Alpha Cen {component} ({len(data)} models)")
        for label, column in (
            ("delta B", "delta_B"),
            ("delta V", "delta_V"),
            ("delta B-V", "delta_B_minus_V"),
        ):
            values = data[column]
            print(
                f"  {label:9s}: median={np.median(values):+.6f}, "
                f"rms={np.sqrt(np.mean(values**2)):.6f}, "
                f"min={np.min(values):+.6f}, max={np.max(values):+.6f}"
            )


if __name__ == "__main__":
    main()
