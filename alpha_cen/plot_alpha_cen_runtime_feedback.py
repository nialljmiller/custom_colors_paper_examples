#!/usr/bin/env python3
"""Plot archived Alpha Cen runtime photometric feedback histories.

Only the archived dynamic-best LOGS/history.data files are read. In the
timestep panels, chi2 and age from row i are aligned with 10**log_dt from
row i+1, the timestep used to produce the next accepted model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
COLORS = {"A": "#0072B2", "B": "#D55E00"}
HISTORY = {
    "A": ROOT / "alpha_cen_a/archive/alpha_cen_a_dynamic_best/LOGS/history.data",
    "B": ROOT / "alpha_cen_b/archive/alpha_cen_b_dynamic_best/LOGS/history.data",
}
CHI_CAPS = ((0.1, 2e4), (1.0, 1e5), (4.0, 5e5))
CHI_LINES = (4.0, 1.0, 0.1)
DT_LINES = (5e5, 1e5, 2e4)
MARGIN_POINTS = 6


def load_history(component):
    path = HISTORY[component]
    if not path.is_file():
        raise FileNotFoundError(
            f"Required archived dynamic-best history is missing: {path}"
        )
    data = np.genfromtxt(path, skip_header=5, names=True)
    required = ("star_age", "model_number", "chi2", "log_dt")
    missing = [name for name in required if name not in data.dtype.names]
    if missing:
        raise ValueError(f"{path} lacks required columns: {missing}")
    data = np.atleast_1d(data)
    for name in required:
        if not np.issubdtype(data[name].dtype, np.number):
            raise TypeError(f"{path}: column {name} is not numeric")
    if not np.all(np.diff(data["model_number"]) == 1):
        raise ValueError(f"{path}: model_number is not consecutive")
    if not np.all(np.diff(data["star_age"]) > 0):
        raise ValueError(f"{path}: star_age is not strictly increasing")
    return path, data


def validate_and_select(component, data):
    age = data["star_age"]
    model = data["model_number"]
    chi2 = data["chi2"]
    log_dt = data["log_dt"]
    finite_fit = np.isfinite(age) & np.isfinite(model) & np.isfinite(chi2) & (chi2 >= 0)
    fit_rows = np.flatnonzero(finite_fit)
    if len(fit_rows) == 0:
        raise ValueError(f"Alpha Cen {component}: no finite non-negative chi2 values")
    best = fit_rows[np.argmin(chi2[fit_rows])]
    approaching = np.flatnonzero(finite_fit & (chi2 <= 10.0))
    if len(approaching) == 0:
        raise ValueError(f"Alpha Cen {component}: chi2 never reaches 10")
    start = max(int(approaching[0]) - MARGIN_POINTS, int(fit_rows[0]))
    end = int(fit_rows[-1])

    # Row i's residual controls the timestep stored on row i+1. The final
    # history row has no following accepted model and is never included here.
    transition_rows = np.arange(len(data) - 1)
    next_dt = np.full(len(data) - 1, np.nan)
    next_dt[:] = np.power(10.0, log_dt[1:])
    aligned = finite_fit[:-1] & np.isfinite(next_dt) & (next_dt > 0)
    aligned_rows = transition_rows[aligned]

    violations = []
    tier_counts = {}
    for threshold, cap in CHI_CAPS:
        tier = aligned_rows[chi2[aligned_rows] <= threshold]
        tier_violations = tier[next_dt[tier] > cap * (1.0 + 1e-10)]
        tier_counts[threshold] = tier
        violations.extend((threshold, int(i), float(model[i]), float(chi2[i]),
                           float(next_dt[i]), cap) for i in tier_violations)

    chosen = np.arange(start, end + 1)
    chosen = chosen[finite_fit[chosen]]
    chosen_transitions = chosen[chosen < len(data) - 1]
    chosen_transitions = chosen_transitions[aligned[chosen_transitions]]

    report = {
        "best_index": int(best),
        "best_model": int(model[best]),
        "best_age": float(age[best]),
        "min_chi2": float(chi2[best]),
        "aligned_rows": aligned_rows,
        "next_dt": next_dt,
        "tier_counts": tier_counts,
        "violations": violations,
        "display_rows": chosen,
        "display_transitions": chosen_transitions,
        "first_chi2_le_10_model": int(model[approaching[0]]),
        "display_start_model": int(model[start]),
        "display_end_model": int(model[end]),
    }
    return report


def print_validation(component, data, report):
    age_gyr = report["best_age"] / 1e9
    print(f"Alpha Cen {component}: {HISTORY[component]}")
    print(f"  best model={report['best_model']}, age={age_gyr:.9f} Gyr, "
          f"minimum chi2={report['min_chi2']:.9g}")
    print(f"  aligned model transitions={len(report['aligned_rows'])}")
    for threshold, cap in CHI_CAPS:
        tier = report["tier_counts"][threshold]
        max_dt = float(np.max(report["next_dt"][tier])) if len(tier) else float("nan")
        print(f"  chi2 <= {threshold:g}: points={len(tier)}, cap={cap:.0f} yr, "
              f"maximum next_dt={max_dt:.0f} yr")
    if report["violations"]:
        print("  APPARENT VIOLATIONS:")
        for threshold, index, model, chi2, next_dt, cap in report["violations"]:
            print(f"    chi2 <= {threshold:g}: row={index}, model={model:.0f}, "
                  f"chi2={chi2:.9g}, next_dt={next_dt:.9g} yr > {cap:.0f} yr")
    else:
        print("  apparent violations=none")
    print(f"  display models={report['display_start_model']}--"
          f"{report['display_end_model']}; first chi2<=10 model="
          f"{report['first_chi2_le_10_model']}")


def make_figure(histories, reports):
    plt.rcParams.update({
        "font.size": 8.3,
        "axes.labelsize": 8.4,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": True,
        "axes.grid": True,
        "grid.alpha": 0.16,
        "grid.linewidth": 0.45,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": False,
    })
    fig, ax_chi = plt.subplots(figsize=(7.1, 4.35))
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.15, top=0.97)
    ax_dt = ax_chi.twinx()
    ax_dt.spines["top"].set_visible(False)
    ax_dt.grid(False)

    for component in ("A", "B"):
        data = histories[component]
        report = reports[component]
        color = COLORS[component]
        best_age = report["best_age"]
        display = report["display_rows"]
        transitions = report["display_transitions"]
        time_myr = (data["star_age"] - best_age) / 1e6

        ax_chi.plot(time_myr[display], data["chi2"][display], color=color,
                    linewidth=1.05, solid_capstyle="round", zorder=3)
        ax_chi.scatter(time_myr[report["best_index"]], report["min_chi2"],
                       s=15, facecolor="white", edgecolor=color, linewidth=0.9,
                       zorder=5)
        ax_chi.set_yscale("log")
        ax_chi.set_ylim(1e-6, 20)
        ax_chi.set_yticks([1e-5, 1e-3, 1e-1, 1e1],
                          labels=[r"$10^{-5}$", r"$10^{-3}$", r"$10^{-1}$", r"$10^{1}$"])
        ax_chi.grid(False, which="minor")

        # A post-step begins at model i's age with the timestep applied to
        # produce i+1. Close its last segment at the next recorded age only.
        t = time_myr[transitions]
        dts = report["next_dt"][transitions]
        if len(transitions):
            last = transitions[-1]
            x_end = time_myr[last + 1]
            t_plot = np.append(t, x_end)
            dt_plot = np.append(dts, dts[-1])
            ax_dt.step(t_plot, dt_plot, where="post", color=color,
                       linewidth=1.05, linestyle="--", zorder=3)
        ax_dt.set_yscale("log")
        ax_dt.set_ylim(1e4, 2e6)
        ax_dt.set_yticks([1e4, 1e5, 1e6],
                         labels=[r"$10^{4}$", r"$10^{5}$", r"$10^{6}$"])
        ax_dt.grid(False, which="minor")
        ax_chi.axvline(0.0, color="0.38", linewidth=0.7, zorder=2)
        ax_chi.scatter(0.0, report["min_chi2"], s=15, facecolor="white",
                       edgecolor=color, linewidth=0.9, zorder=5)

    ax_chi.set_xlim(-1650, 55)
    ax_chi.set_xlabel("Time from best model (Myr)")
    ax_chi.set_ylabel("Photometric chi-square")
    ax_dt.set_ylabel("Subsequent timestep (yr)")
    ax_chi.grid(True, which="major", axis="y", color="0.86", linewidth=0.45)
    ax_chi.tick_params(axis="both", direction="out", length=2.6, width=0.65, pad=2)
    ax_dt.tick_params(axis="y", direction="out", length=2.6, width=0.65, pad=2)
    ax_chi.margins(x=0.01)

    handles = []
    for component, name in (("A", r"$\alpha$ Cen A"), ("B", r"$\alpha$ Cen B")):
        color = COLORS[component]
        handles.extend((
            Line2D([0], [0], color=color, lw=1.25, ls="-", label=f"{name}, $\\chi^2$"),
            Line2D([0], [0], color=color, lw=1.25, ls="--", label=f"{name}, timestep"),
        ))
    ax_chi.legend(handles=handles, loc="lower left", ncol=2, fontsize=7.4,
                  frameon=True, framealpha=0.92, facecolor="white",
                  edgecolor="0.86", borderpad=0.45, handlelength=2.0,
                  columnspacing=1.0, labelspacing=0.35)

    outdir = ROOT / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / "fig_alpha_cen_runtime_feedback.pdf"
    png = outdir / "fig_alpha_cen_runtime_feedback.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf}")
    print(f"Saved {png}")


def main():
    histories = {}
    reports = {}
    for component in ("A", "B"):
        path, histories[component] = load_history(component)
        print(f"Verified required columns in {path}: "
              "star_age, model_number, chi2, log_dt")
        reports[component] = validate_and_select(component, histories[component])
        print_validation(component, histories[component], reports[component])
    make_figure(histories, reports)


if __name__ == "__main__":
    main()
