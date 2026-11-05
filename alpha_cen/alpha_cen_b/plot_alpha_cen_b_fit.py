#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter


# Alpha Cen B simplex output and adopted resolved Johnson constraints.
RESULTS = Path("outputs/alpha_cen_b_simplex_results.data")
OUTDIR = Path("plots")

TARGET_MV = 5.71
TARGET_BV = 0.88
SIGMA_MV = 0.03
SIGMA_BV = 0.03
INJECTED_ALPHA = None


# MESA output has five header lines before the column names.
d = np.atleast_1d(
    np.genfromtxt(RESULTS, names=True, skip_header=5, dtype=None, encoding=None)
)

sample = d["sample"].astype(int)
chi2 = d["chi2"]
alpha = d["alpha"]
age = d["age"] / 1e9
mv = d["M_V"]
bv = d["B_minus_V"]

best = np.argmin(chi2)
by_alpha = np.argsort(alpha)
by_sample = np.argsort(sample)
best_so_far = np.minimum.accumulate(chi2[by_sample])
norm = LogNorm(chi2.min(), chi2.max())

plt.rcParams.update(
    {
        "font.size": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)

OUTDIR.mkdir(exist_ok=True)


# Main landscape figure -------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(11.2, 8.3), layout="constrained")
ax_loss, ax_convergence, ax_cmd, ax_age = ax.flat

# Each point is one complete MESA track. astero has already selected the
# lowest-chi2 age along that track.
ax_loss.plot(alpha[by_alpha], chi2[by_alpha], color="0.65")
ax_loss.scatter(alpha, chi2, color="tab:blue", edgecolor="white", s=42)
ax_loss.scatter(
    alpha[best], chi2[best], marker="*", s=180,
    color="#E69F00", edgecolor="black", zorder=5,
    label=rf"Best: $\alpha_{{\rm MLT}}={alpha[best]:.6f}$",
)
if INJECTED_ALPHA is not None:
    ax_loss.axvline(
        INJECTED_ALPHA, color="#D55E00", ls="--",
        label=rf"Injected $\alpha_{{\rm MLT}}={INJECTED_ALPHA:.6f}$",
    )
ax_loss.set_yscale("log")
ax_loss.set(
    xlabel=r"Mixing length $\alpha_{\rm MLT}$",
    ylabel=r"Profiled $\chi^2_{\rm phot}$",
)
ax_loss.legend()

# Chronological simplex convergence.
ax_convergence.plot(
    sample[by_sample], chi2[by_sample], "o-", color="#9CA3AF",
    markersize=4.5, label="Trial loss",
)
ax_convergence.step(
    sample[by_sample], best_so_far, where="post",
    color="#D55E00", linewidth=2, label="Best so far",
)
ax_convergence.scatter(
    sample[best], chi2[best], marker="*", s=160,
    color="#E69F00", edgecolor="black", zorder=5,
)
ax_convergence.set_yscale("log")
ax_convergence.set(
    xlabel="Track evaluation",
    ylabel=r"$\chi^2_{\rm phot}$",
)
ax_convergence.legend()

# CMD of the best-age model returned by each trial track.
points = ax_cmd.scatter(
    bv, mv, c=chi2, norm=norm, cmap="viridis",
    s=55, edgecolor="white",
)
ax_cmd.scatter(
    TARGET_BV, TARGET_MV, marker="*", s=190,
    color="white", edgecolor="black", zorder=5,
    label="Photometric target",
)
ax_cmd.scatter(
    bv[best], mv[best], s=110, facecolor="none",
    edgecolor="#E69F00", linewidth=2, label="Best model",
)
ax_cmd.invert_yaxis()
ax_cmd.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
ax_cmd.set(
    xlabel=r"$B-V$ (mag)",
    ylabel=r"$M_V$ (mag)",
)
ax_cmd.legend()

# Age is selected inside each track rather than being a simplex parameter.
ax_age.plot(alpha[by_alpha], age[by_alpha], color="0.7")
ax_age.scatter(
    alpha, age, c=chi2, norm=norm, cmap="viridis",
    s=55, edgecolor="white",
)
ax_age.scatter(
    alpha[best], age[best], marker="*", s=180,
    color="#E69F00", edgecolor="black", zorder=5,
)
if INJECTED_ALPHA is not None:
    ax_age.axvline(INJECTED_ALPHA, color="#D55E00", ls="--")
ax_age.set(
    xlabel=r"Mixing length $\alpha_{\rm MLT}$",
    ylabel="Best-fitting age (Gyr)",
)

cbar = fig.colorbar(points, ax=[ax_cmd, ax_age])
cbar.set_label(r"Profiled $\chi^2_{\rm phot}$")
fig.savefig(OUTDIR / "alpha_recovery_landscape.pdf", bbox_inches="tight")
fig.savefig(OUTDIR / "alpha_recovery_landscape.png", dpi=300, bbox_inches="tight")


# Why the loss selects alpha --------------------------------------------------
res_mv = (mv - TARGET_MV) / SIGMA_MV
res_bv = (bv - TARGET_BV) / SIGMA_BV

fig, ax = plt.subplots(1, 2, figsize=(10.8, 4.45), layout="constrained")

ax[0].axhline(0, color="0.3")
ax[0].plot(alpha[by_alpha], res_mv[by_alpha], "o-", label=r"$M_V$")
ax[0].plot(alpha[by_alpha], res_bv[by_alpha], "s-", label=r"$B-V$")
if INJECTED_ALPHA is not None:
    ax[0].axvline(INJECTED_ALPHA, color="#D55E00", ls="--")
ax[0].set(
    xlabel=r"Mixing length $\alpha_{\rm MLT}$",
    ylabel=r"Standardized residual $(M-O)/\sigma$",
)
ax[0].legend()

ax[1].plot(alpha[by_alpha], chi2[by_alpha], "o-", color="black", label="Total")
ax[1].plot(alpha[by_alpha], res_mv[by_alpha] ** 2, "o-", label=r"$M_V$")
ax[1].plot(alpha[by_alpha], res_bv[by_alpha] ** 2, "s-", label=r"$B-V$")
if INJECTED_ALPHA is not None:
    ax[1].axvline(INJECTED_ALPHA, color="#D55E00", ls="--")
ax[1].set_yscale("log")
ax[1].set(
    xlabel=r"Mixing length $\alpha_{\rm MLT}$",
    ylabel=r"Contribution to $\chi^2_{\rm phot}$",
)
ax[1].legend()

fig.savefig(OUTDIR / "alpha_recovery_constraints.pdf", bbox_inches="tight")
fig.savefig(OUTDIR / "alpha_recovery_constraints.png", dpi=300, bbox_inches="tight")


print(f"best sample = {sample[best]}")
print(f"best alpha  = {alpha[best]:.12f}")
print(f"best chi2   = {chi2[best]:.12e}")

