#!/usr/bin/env python3
"""
plot_rsp.py - Analysis plots for MESA Colors RSP RR Lyrae demonstration

Generates publication-quality figures showing phase-resolved synthetic
photometry during radial stellar pulsation of an RR Lyrae variable.

Produces (saved to this directory):
1. rsp_lightcurve     - Multi-band light curves vs pulsation phase
2. rsp_colorloop      - Color-magnitude loop (B-V vs V)
3. rsp_amplitudes     - Pulsation amplitude vs wavelength

Usage:
    cd rsp_RR_Lyrae/plots
    python plot_rsp.py

Author: Niall Miller (2025) - for Miller, Joyce, Mocz et al.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path
import sys
import os

# Publication-quality settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (3.5, 3.5),  # Single column ApJ
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Paths relative to this script's location (inside plots/)
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
DEFAULT_LOGS = PARENT_DIR / 'LOGS'
OUTPUT_DIR = SCRIPT_DIR  # Save plots right here


# =============================================================================
# Data Loading
# =============================================================================
def load_mesa_history(filepath):
    """
    Load MESA history.data file.
    Returns structured numpy array with column names as attributes.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find header line (contains column names)
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('model_number') or 'model_number' in line:
            header_idx = i
            break

    if header_idx is None:
        # Try standard MESA format: line 6 has headers
        header_idx = 5

    cols = lines[header_idx].split()
    data = np.genfromtxt(filepath, skip_header=header_idx + 1, names=cols)

    return data


def find_magnitude_columns(data):
    """
    Find magnitude column names in the history data.
    MESA Colors may use different naming conventions.
    """
    colnames = data.dtype.names

    # Possible naming patterns for Johnson bands
    patterns = {
        'B': ['B', 'abs_mag_B', 'mag_B', 'Johnson_B'],
        'V': ['V', 'abs_mag_V', 'mag_V', 'Johnson_V'],
        'R': ['R', 'abs_mag_R', 'mag_R', 'Johnson_R'],
        'I': ['I', 'abs_mag_I', 'mag_I', 'Johnson_I'],
        'U': ['U', 'abs_mag_U', 'mag_U', 'Johnson_U'],
    }

    found = {}
    for band, candidates in patterns.items():
        for cand in candidates:
            if cand in colnames:
                found[band] = cand
                break

    return found


def get_column(data, name, default=None):
    """Safely get a column from data."""
    if name in data.dtype.names:
        return data[name]
    return default


# =============================================================================
# Figure 1: Multi-band Light Curves vs Phase
# =============================================================================
def plot_lightcurve(data, output_dir=OUTPUT_DIR):
    """
    Plot multi-band light curves vs pulsation phase.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    # Get time/phase data
    if 'rsp_phase' in data.dtype.names:
        phase = data['rsp_phase']
    else:
        # Compute phase from time
        time = get_column(data, 'star_age_day', get_column(data, 'star_age', None))
        if time is None:
            print("Error: No time column found")
            return
        period = 0.71  # days
        phase = ((time - time[0]) / period) % 1.0

    # Find magnitude columns
    mag_cols = find_magnitude_columns(data)

    if not mag_cols:
        print("Warning: No magnitude columns found.")
        print("Available columns:")
        print(data.dtype.names)
        return

    print(f"Found magnitude columns: {mag_cols}")

    # Define plotting parameters
    band_info = {
        'B': {'color': '#0066CC', 'offset': 0.0, 'label': '$B$'},
        'V': {'color': '#33AA33', 'offset': 1.2, 'label': '$V$'},
        'R': {'color': '#CC6600', 'offset': 2.4, 'label': '$R$'},
        'I': {'color': '#990033', 'offset': 3.6, 'label': '$I$'},
    }

    # Select data for ~2-3 complete cycles after initial settling
    # Skip first ~3 cycles to avoid transients
    if 'rsp_num_periods' in data.dtype.names:
        mask = (data['rsp_num_periods'] >= 3) & (data['rsp_num_periods'] <= 5)
    else:
        # Use time-based selection
        time = get_column(data, 'star_age_day', data['star_age'])
        mask = (time > 2.0) & (time < 4.5)  # ~3 cycles

    if mask.sum() < 100:
        print("Warning: Few points in selected range, using all data")
        mask = np.ones(len(data), dtype=bool)

    # Plot each band
    for band in ['B', 'V', 'R', 'I']:
        if band in mag_cols:
            col = mag_cols[band]
            mag = data[col][mask]
            ph = phase[mask]
            info = band_info[band]

            # Sort by phase for cleaner plotting
            sort_idx = np.argsort(ph)

            ax.plot(ph[sort_idx], mag[sort_idx] + info['offset'],
                    '.', ms=1.5, alpha=0.6, color=info['color'],
                    label=info['label'])

    ax.set_xlabel('Pulsation Phase')
    ax.set_ylabel('Magnitude + offset')
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.legend(loc='upper right', markerscale=4, framealpha=0.9)

    # Add annotation about offsets
    ax.text(0.02, 0.02, 'Offsets: +1.2 mag between bands',
            transform=ax.transAxes, fontsize=8, alpha=0.7)

    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    output_file = output_dir / 'rsp_lightcurve.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_file).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_file}")


# =============================================================================
# Figure 2: Color-Magnitude Loop
# =============================================================================
def plot_colorloop(data, output_dir=OUTPUT_DIR):
    """
    Plot color-magnitude loop (B-V vs V).
    Shows the hysteresis loop characteristic of pulsating variables.
    """

    from itertools import combinations
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))

    mag_cols = find_magnitude_columns(data)



    # ------------------------------------------------------------
    # Precompute mags, colors, phase, and mask once
    # ------------------------------------------------------------
    bands = ['U', 'B', 'V', 'R', 'I', 'J', 'M']

    # Magnitudes
    mags = {band: data[mag_cols[band]] for band in bands}

    # All unique colors, preserving order like U-B, U-V, ..., R-I
    colors = {}
    for b1, b2 in combinations(bands, 2):
        colors[(b1, b2)] = mags[b1] - mags[b2]

    # Get phase for coloring
    if 'rsp_phase' in data.dtype.names:
        phase = data['rsp_phase']
    else:
        time = get_column(data, 'star_age_day', data['star_age'])
        period = 0.71
        phase = ((time - time[0]) / period) % 1.0

    # Select one complete cycle (after settling)
    if 'rsp_num_periods' in data.dtype.names:
        mask = (data['rsp_num_periods'] >= 4)
    else:
        time = get_column(data, 'star_age_day', data['star_age'])
        mask = (time > 3.0)

    if mask.sum() < 50:
        # Fall back to using phase
        unique_phases = []
        unique_idx = []
        for i, p in enumerate(phase):
            if len(unique_phases) == 0 or abs(p - unique_phases[-1]) > 0.001:
                unique_phases.append(p)
                unique_idx.append(i)

        mask = np.zeros(len(data), dtype=bool)
        mask[unique_idx[:1000]] = True

    # ------------------------------------------------------------
    # Small helper to add direction arrow and phase markers
    # ------------------------------------------------------------
    def add_phase_annotations(ax, x_arr, y_arr, p_arr, invert_y=False):
        idx_arrow1 = np.argmin(np.abs(p_arr - 0.25))
        idx_arrow2 = np.argmin(np.abs(p_arr - 0.30))

        if idx_arrow1 != idx_arrow2:
            ax.annotate(
                '',
                xy=(x_arr[idx_arrow2], y_arr[idx_arrow2]),
                xytext=(x_arr[idx_arrow1], y_arr[idx_arrow1]),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
            )

        # Mark key phases
        for target_phase, label in [(0.0, 'max'), (0.5, 'min')]:
            idx = np.argmin(np.abs(p_arr - target_phase))
            ax.plot(x_arr[idx], y_arr[idx], 'k*', ms=10, zorder=10)

        if invert_y:
            ax.invert_yaxis()

        ax.grid(True, alpha=0.2)


    # ------------------------------------------------------------
    # 1) Color-Magnitude diagrams
    # ------------------------------------------------------------
    for (b1, b2), x_full in colors.items():
        for mag_band in bands:
            y_full = mags[mag_band]

            x = x_full[mask]
            y = y_full[mask]
            p = phase[mask]

            fig, ax = plt.subplots(figsize=(5, 5))

            sc = ax.scatter(
                x, y,
                c=p, cmap='twilight_shifted',
                s=8, alpha=0.8, edgecolors='none'
            )

            plt.colorbar(sc, ax=ax, label='Pulsation Phase', shrink=0.8)

            add_phase_annotations(ax, x, y, p, invert_y=True)

            ax.set_xlabel(f'${b1} - {b2}$ (mag)')
            ax.set_ylabel(f'${mag_band}$ (mag)')

            plt.tight_layout()

            stem = f'rsp_cmd_{b1}minus{b2}_vs_{mag_band}'
            pdf_file = output_dir / 'cmd' / f'{stem}.pdf'
            png_file = output_dir / 'cmd' /f'{stem}.png'

            plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
            plt.savefig(png_file, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Saved {pdf_file}")


    # ------------------------------------------------------------
    # 2) Color-Color diagrams
    # ------------------------------------------------------------
    color_items = list(colors.items())

    for i in range(len(color_items)):
        (x_b1, x_b2), x_full = color_items[i]
        for j in range(i + 1, len(color_items)):
            (y_b1, y_b2), y_full = color_items[j]

            x = x_full[mask]
            y = y_full[mask]
            p = phase[mask]

            fig, ax = plt.subplots(figsize=(5, 5))

            sc = ax.scatter(
                x, y,
                c=p, cmap='twilight_shifted',
                s=8, alpha=0.8, edgecolors='none'
            )

            plt.colorbar(sc, ax=ax, label='Pulsation Phase', shrink=0.8)

            add_phase_annotations(ax, x, y, p, invert_y=False)

            ax.set_xlabel(f'${x_b1} - {x_b2}$ (mag)')
            ax.set_ylabel(f'${y_b1} - {y_b2}$ (mag)')

            plt.tight_layout()

            stem = f'rsp_cc_{x_b1}minus{x_b2}_vs_{y_b1}minus{y_b2}'
            pdf_file = output_dir / 'cc' /f'{stem}.pdf'
            png_file = output_dir / 'cc' /f'{stem}.png'

            plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
            plt.savefig(png_file, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Saved {pdf_file}")




# =============================================================================
# Figure 3: Amplitude Spectrum
# =============================================================================
def plot_amplitude_spectrum(data, output_dir=OUTPUT_DIR):
    """
    Plot pulsation amplitude vs wavelength.
    Shows wavelength-dependent amplitude characteristic of pulsating stars.
    """
    fig, ax = plt.subplots(figsize=(5, 3.5))

    mag_cols = find_magnitude_columns(data)

    # Effective wavelengths (Angstroms)
    wavelengths = {'U': 3650, 'B': 4400, 'V': 5500, 'R': 6400, 'I': 8000}
    colors_plot = {'U': '#6600CC', 'B': '#0066CC', 'V': '#33AA33',
                   'R': '#CC6600', 'I': '#990033'}

    wl_plot = []
    amp_plot = []
    col_plot = []
    labels = []

    for band in ['U', 'B', 'V', 'R', 'I']:
        if band in mag_cols:
            mag = data[mag_cols[band]]
            amp = np.max(mag) - np.min(mag)
            wl_plot.append(wavelengths[band])
            amp_plot.append(amp)
            col_plot.append(colors_plot[band])
            labels.append(band)

    if not wl_plot:
        print("No magnitude data available for amplitude spectrum")
        plt.close()
        return

    ax.scatter(wl_plot, amp_plot, c=col_plot, s=100, zorder=5, edgecolors='k')
    ax.plot(wl_plot, amp_plot, 'k--', alpha=0.4, zorder=1)

    for w, a, l in zip(wl_plot, amp_plot, labels):
        ax.annotate(l, (w, a), xytext=(5, 5), textcoords='offset points', fontsize=10)

    ax.set_xlabel('Effective Wavelength (Å)')
    ax.set_ylabel('Pulsation Amplitude (mag)')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    output_file = output_dir / 'rsp_amplitudes.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_file).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_file}")


# =============================================================================
# Statistics
# =============================================================================
def print_statistics(data):
    """Print useful statistics about the run."""
    mag_cols = find_magnitude_columns(data)

    print("\n" + "=" * 60)
    print("RSP + Colors Run Statistics")
    print("=" * 60)

    n_models = len(data)
    print(f"Total models: {n_models}")

    if 'rsp_num_periods' in data.dtype.names:
        n_periods = int(np.max(data['rsp_num_periods']))
        print(f"Pulsation cycles completed: {n_periods}")
        print(f"Points per cycle: ~{n_models // max(n_periods, 1)}")

    if 'rsp_period_in_days' in data.dtype.names:
        periods = data['rsp_period_in_days']
        valid = periods > 0
        if valid.any():
            print(f"Mean period: {np.mean(periods[valid]):.5f} days")

    print("\nMagnitude ranges:")
    for band in ['B', 'V', 'R', 'I']:
        if band in mag_cols:
            mag = data[mag_cols[band]]
            print(f"  {band}: {np.min(mag):.3f} to {np.max(mag):.3f} "
                  f"(Δ = {np.max(mag) - np.min(mag):.3f} mag)")

    if 'B' in mag_cols and 'V' in mag_cols:
        color = data[mag_cols['B']] - data[mag_cols['V']]
        print(f"\n(B-V) color range: {np.min(color):.3f} to {np.max(color):.3f}")

    print("=" * 60 + "\n")


# =============================================================================
# Main
# =============================================================================
def main():
    """Main analysis routine."""
    print("=" * 60)
    print("MESA Colors RSP RR Lyrae Analysis")
    print("=" * 60)

    # Find history file
    history_file = DEFAULT_LOGS / 'history.data'

    if not history_file.exists():
        # Try command-line argument
        if len(sys.argv) >= 2:
            history_file = Path(sys.argv[1])
        else:
            print(f"ERROR: History file not found: {history_file}")
            print("Usage: python plot_rsp.py [LOGS/history.data]")
            sys.exit(1)

    if not history_file.exists():
        print(f"Error: File not found: {history_file}")
        sys.exit(1)

    print(f"Loading {history_file}...")
    data = load_mesa_history(str(history_file))

    print(f"Loaded {len(data)} models")
    print(f"Columns: {', '.join(data.dtype.names[:10])}...")

    # Print statistics
    print_statistics(data)

    # Generate figures
    print("Generating figures...")
    plot_lightcurve(data)
    plot_colorloop(data)
    plot_amplitude_spectrum(data)

    print(f"\nDone! Figures saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
