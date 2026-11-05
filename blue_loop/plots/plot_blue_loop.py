#!/usr/bin/env python3
"""
plot_blue_loop.py - Analysis plots for MESA Colors Blue Loop demonstration

Generates publication-quality figures showing chromatic evolution during
blue loop phase of a 5 Msun intermediate-mass star.

Produces (saved to this directory):
1. cmd_blue_loop        - CMD track (g-r vs g) showing blue loop trajectory
2. magnitude_evolution  - Multi-band magnitude evolution vs time
3. color_evolution      - Color evolution vs time
4. hr_diagram           - HR diagram with color-coded evolutionary phase
5. teff_vs_color        - Teff vs color showing atmosphere interpolation

Usage:
    cd cepheid_blue_loop/plots
    python plot_blue_loop.py

Author: Niall Miller (2025)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Paths relative to this script's location (inside plots/)
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
DEFAULT_LOGS = PARENT_DIR / 'LOGS'
OUTPUT_DIR = SCRIPT_DIR  # Save plots right here


# =============================================================================
# Data Loading
# =============================================================================
def read_mesa_history(history_file):
    """
    Read MESA history.data file into a dictionary of arrays.
    """
    history_file = Path(history_file)

    with open(history_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 5:
                column_names = line.split()
                break

    data = np.genfromtxt(history_file, skip_header=6, names=column_names)
    history = {name: data[name] for name in data.dtype.names}

    return history


def find_blue_loop_phase(history):
    """
    Identify the blue loop phase in the evolution.
    Returns (start_idx, end_idx) indices bracketing the blue loop phase.
    """
    log_teff = history['log_Teff']
    teff = 10**log_teff

    # Find the minimum Teff (RGB tip region)
    rgb_tip_idx = np.argmin(teff[:len(teff)//2])
    start_idx = rgb_tip_idx

    return start_idx, len(teff) - 1


# =============================================================================
# Figure 1: CMD Track
# =============================================================================
def plot_cmd_track(history, output_dir, filename='cmd_blue_loop.pdf'):
    """Plot CMD track showing blue loop trajectory."""
    fig, ax = plt.subplots(figsize=(6, 7))

    g = history['g']
    r = history['r']
    color = g - r
    age = history['star_age'] / 1e6  # Myr

    scatter = ax.scatter(color, g, c=age, cmap='viridis', s=2, alpha=0.8)
    cbar = plt.colorbar(scatter, ax=ax, label='Age [Myr]')
    ax.plot(color, g, 'k-', lw=0.3, alpha=0.3, zorder=0)

    ax.scatter(color[0], g[0], c='green', s=100, marker='o',
               edgecolors='black', zorder=5, label='Start')
    ax.scatter(color[-1], g[-1], c='red', s=100, marker='s',
               edgecolors='black', zorder=5, label='End')

    ax.invert_yaxis()
    ax.set_xlabel(r'$g - r$ [mag]', fontsize=12)
    ax.set_ylabel(r'$g$ [mag]', fontsize=12)
    ax.set_title('Blue Loop CMD Track (LSST filters, AB)', fontsize=12)
    ax.legend(loc='lower left')
    ax.axvspan(0.25, 0.65, alpha=0.15, color='blue', label='Instability Strip')

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / filename}")


# =============================================================================
# Figure 2: Magnitude Evolution
# =============================================================================
def plot_magnitude_evolution(history, output_dir, filename='magnitude_evolution.pdf'):
    """Plot multi-band magnitude evolution vs time."""
    fig, ax = plt.subplots(figsize=(10, 6))

    age = history['star_age'] / 1e6
    bands = ['u', 'g', 'r', 'i', 'z', 'y']
    colors = ['purple', 'blue', 'green', 'orange', 'red', 'darkred']

    for band, color in zip(bands, colors):
        if band in history:
            ax.plot(age, history[band], label=band, color=color, lw=1.5)

    ax.invert_yaxis()
    ax.set_xlabel('Age [Myr]', fontsize=12)
    ax.set_ylabel('Magnitude [AB]', fontsize=12)
    ax.set_title('Multi-band Magnitude Evolution During Blue Loop', fontsize=12)
    ax.legend(loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / filename}")


# =============================================================================
# Figure 3: Color Evolution
# =============================================================================
def plot_color_evolution(history, output_dir, filename='color_evolution.pdf'):
    """Plot color evolution vs time."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    age = history['star_age'] / 1e6

    # g-r color
    ax1 = axes[0]
    g_r = history['g'] - history['r']
    ax1.plot(age, g_r, 'b-', lw=1.5)
    ax1.set_ylabel(r'$g - r$ [mag]', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhspan(0.25, 0.65, alpha=0.15, color='blue', label='Instability Strip')
    ax1.legend(loc='best')

    # r-i color
    ax2 = axes[1]
    r_i = history['r'] - history['i']
    ax2.plot(age, r_i, 'r-', lw=1.5)
    ax2.set_ylabel(r'$r - i$ [mag]', fontsize=12)
    ax2.set_xlabel('Age [Myr]', fontsize=12)
    ax2.grid(True, alpha=0.3)

    axes[0].set_title('Color Evolution During Blue Loop', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / filename}")


# =============================================================================
# Figure 4: HR Diagram
# =============================================================================
def plot_hr_diagram(history, output_dir, filename='hr_diagram.pdf'):
    """Plot HR diagram with evolutionary track."""
    fig, ax = plt.subplots(figsize=(8, 7))

    log_teff = history['log_Teff']
    log_l = history['log_L']
    g_r = history['g'] - history['r']

    scatter = ax.scatter(log_teff, log_l, c=g_r, cmap='coolwarm', s=3, alpha=0.8)
    cbar = plt.colorbar(scatter, ax=ax, label=r'$g - r$ [mag]')
    ax.plot(log_teff, log_l, 'k-', lw=0.3, alpha=0.3, zorder=0)

    ax.scatter(log_teff[0], log_l[0], c='green', s=100, marker='o',
               edgecolors='black', zorder=5, label='Start')
    ax.scatter(log_teff[-1], log_l[-1], c='red', s=100, marker='s',
               edgecolors='black', zorder=5, label='End')

    ax.invert_xaxis()
    ax.axvline(np.log10(5500), color='red', ls='--', alpha=0.5, label='IS Red Edge')
    ax.axvline(np.log10(6500), color='blue', ls='--', alpha=0.5, label='IS Blue Edge')

    ax.set_xlabel(r'$\log(T_{\rm eff}/{\rm K})$', fontsize=12)
    ax.set_ylabel(r'$\log(L/L_\odot)$', fontsize=12)
    ax.set_title('HR Diagram with Colors', fontsize=12)
    ax.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / filename}")


# =============================================================================
# Figure 5: Teff vs Color
# =============================================================================
def plot_teff_vs_color(history, output_dir, filename='teff_vs_color.pdf'):
    """Plot Teff vs color to show atmosphere interpolation performance."""
    fig, ax = plt.subplots(figsize=(8, 6))

    teff = 10**history['log_Teff']
    g_r = history['g'] - history['r']
    age = history['star_age'] / 1e6

    scatter = ax.scatter(teff, g_r, c=age, cmap='viridis', s=5, alpha=0.8)
    cbar = plt.colorbar(scatter, ax=ax, label='Age [Myr]')

    ax.set_xlabel(r'$T_{\rm eff}$ [K]', fontsize=12)
    ax.set_ylabel(r'$g - r$ [mag]', fontsize=12)
    ax.set_title('Color-Temperature Relation During Blue Loop', fontsize=12)
    ax.grid(True, alpha=0.3)

    teff_min, teff_max = teff.min(), teff.max()
    ax.axvline(teff_min, color='gray', ls=':', alpha=0.5)
    ax.axvline(teff_max, color='gray', ls=':', alpha=0.5)
    ax.text(0.02, 0.98, f'Teff range: {teff_min:.0f} - {teff_max:.0f} K',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / filename}")


# =============================================================================
# Summary Statistics
# =============================================================================
def print_summary_statistics(history):
    """Print summary statistics for the blue loop evolution."""
    print("\n" + "="*60)
    print("BLUE LOOP EVOLUTION SUMMARY")
    print("="*60)

    teff = 10**history['log_Teff']
    log_l = history['log_L']
    g_r = history['g'] - history['r']

    print(f"\nTeff range: {teff.min():.0f} - {teff.max():.0f} K")
    print(f"log(L) range: {log_l.min():.3f} - {log_l.max():.3f}")
    print(f"g-r color range: {g_r.min():.3f} - {g_r.max():.3f} mag")
    print(f"Color amplitude: delta(g-r) = {g_r.max() - g_r.min():.3f} mag")

    in_is = (g_r > 0.25) & (g_r < 0.65)
    crossings = np.sum(np.diff(in_is.astype(int)) != 0)
    print(f"Instability strip crossings: ~{crossings}")

    age_span = (history['star_age'][-1] - history['star_age'][0]) / 1e6
    print(f"Evolution time span: {age_span:.2f} Myr")
    print(f"Number of models: {len(history['model_number'])}")

    if 'Interp_rad' in history:
        max_interp = np.max(history['Interp_rad'])
        print(f"Max interpolation radius: {max_interp:.3f}")
        if max_interp > 1.0:
            print("  WARNING: Atmosphere grid extrapolation detected!")

    print("="*60 + "\n")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Generate figures for MESA Colors Blue Loop demonstration'
    )
    parser.add_argument('--logs_dir', type=str, default=str(DEFAULT_LOGS),
                       help='Path to MESA LOGS directory')
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_DIR),
                       help='Output directory for figures')
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    history_file = logs_dir / 'history.data'
    if not history_file.exists():
        print(f"ERROR: History file not found: {history_file}")
        print("Run MESA first, or check --logs_dir argument")
        return

    print(f"Reading: {history_file}")
    history = read_mesa_history(history_file)

    print_summary_statistics(history)

    print("Generating figures...")
    plot_cmd_track(history, output_dir)
    plot_magnitude_evolution(history, output_dir)
    plot_color_evolution(history, output_dir)
    plot_hr_diagram(history, output_dir)
    plot_teff_vs_color(history, output_dir)

    print(f"\nDone! Figures saved to: {output_dir}")


if __name__ == '__main__':
    main()
