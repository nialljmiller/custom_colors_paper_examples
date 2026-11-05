#!/usr/bin/env python3
"""Redraw the timing figure (timing_feature_breakdown.pdf) at printed column width.

Bar heights are the per-step overheads (s/step relative to the module-disabled
baseline) from the benchmark run with test_timings.py (MESA custom_colors test
suite); they are the values printed on the original benchmark figure and agree
with Table 6 (Appendix J) to its rounding.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8.5, 'font.family': 'serif', 'axes.labelsize': 9,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8})
labels = ["Vega", "AB", "ST", "Vega\n+ CSV", "Vega + CSV\n+ per model"]
vals = [0.0024, 0.0012, 0.0018, 0.0430, 0.0389]   # delta s / step vs module disabled
cols = ['#dd8452', '#55a868', '#c44e52', '#8172b3', '#937860']

fig, ax = plt.subplots(figsize=(3.4, 2.3))
ax.bar(range(5), vals, color=cols, width=0.7, edgecolor='white')
for i, v in enumerate(vals):
    ax.text(i, v + 0.0008, f"{v:.4f} s", ha='center', va='bottom', fontsize=7.5)
ax.set_xticks(range(5))
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel(r"$\Delta$ s step$^{-1}$ vs module-disabled")
ax.set_ylim(0, 0.05)
ax.grid(axis='y', alpha=0.25, lw=0.5)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("timing_feature_breakdown.pdf")
fig.savefig("timing_feature_breakdown.png", dpi=200)
