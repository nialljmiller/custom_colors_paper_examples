#!/usr/bin/env python3
"""
plot_rsp_radius_temp.py - Analysis plots for MESA Colors RSP RR Lyrae demonstration

Builds on the original plotting script and adds physics-driven diagnostics to show:
1. the near-IR flux more closely tracking radius changes, and
2. the optical/optical-IR colors more closely tracking temperature changes.

Produces (saved to this directory):
1. rsp_lightcurve                    - Multi-band light curves vs pulsation phase
2. rsp_colorloop_* / rsp_cc_*        - Color-magnitude and color-color loops
3. rsp_amplitudes                    - Pulsation amplitude vs wavelength
4. rsp_ir_radius_diagnostic          - J-band vs radius diagnostic figure
5. rsp_optical_temperature_diagnostic- optical color vs temperature diagnostic figure

Usage:
    cd rsp_RR_Lyrae/plots
    python plot_rsp_radius_temp.py

Author: Niall Miller (2025) - extended by ChatGPT (2026)
"""

from itertools import combinations
from pathlib import Path
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np


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
OUTPUT_DIR = SCRIPT_DIR

# Approximate effective wavelengths (Angstroms)
BAND_WAVELENGTHS = {
    'U': 3650,
    'B': 4400,
    'V': 5500,
    'R': 6400,
    'I': 8000,
    'J': 12200,
}

BAND_COLORS = {
    'U': '#6600CC',
    'B': '#0066CC',
    'V': '#33AA33',
    'R': '#CC6600',
    'I': '#990033',
    'J': '#6B1FB1',
}


# =============================================================================
# Data Loading
# =============================================================================
def load_mesa_history(filepath):
    """Load MESA history.data file into a structured numpy array."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('model_number') or 'model_number' in line:
            header_idx = i
            break

    if header_idx is None:
        header_idx = 5

    cols = lines[header_idx].split()
    data = np.genfromtxt(filepath, skip_header=header_idx + 1, names=cols)
    return data


# =============================================================================
# Column helpers
# =============================================================================
def find_magnitude_columns(data):
    """Find magnitude column names in the history data."""
    colnames = data.dtype.names
    patterns = {
        'U': ['U', 'abs_mag_U', 'mag_U', 'Johnson_U'],
        'B': ['B', 'abs_mag_B', 'mag_B', 'Johnson_B'],
        'V': ['V', 'abs_mag_V', 'mag_V', 'Johnson_V'],
        'R': ['R', 'abs_mag_R', 'mag_R', 'Johnson_R'],
        'I': ['I', 'abs_mag_I', 'mag_I', 'Johnson_I'],
        'M': ['M', 'abs_mag_M', 'mag_M', 'Johnson_M'],
        'J': ['J', 'abs_mag_J', 'mag_J', 'Johnson_J', '2MASS_J'],
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


def get_phase(data):
    """Return pulsation phase, using model output if available."""
    if 'rsp_phase' in data.dtype.names:
        return np.asarray(data['rsp_phase'])

    time = get_column(data, 'star_age_day', get_column(data, 'star_age', None))
    if time is None:
        raise ValueError('No phase or time column found in history.data')

    period = 0.71
    return ((time - time[0]) / period) % 1.0



def get_settled_cycle_mask(data, min_periods=4.0, single_cycle=False):
    """
    Select one settled pulsation cycle if possible.

    For RSP runs with rsp_num_periods, prefer a single complete settled cycle.
    Otherwise fall back to a simple time cut.
    """
    n = len(data)

    if 'rsp_num_periods' in data.dtype.names:
        periods = np.asarray(data['rsp_num_periods'], dtype=float)
        finite = np.isfinite(periods)
        if not finite.any():
            return np.ones(n, dtype=bool)

        if single_cycle:
            max_p = np.nanmax(periods[finite])
            cycle_floor = np.floor(max_p)
            candidate = (periods >= max(min_periods, cycle_floor - 1.0)) & (periods < cycle_floor)
            if candidate.sum() >= 20:
                return candidate

            candidate = (periods >= min_periods) & (periods < min_periods + 1.0)
            if candidate.sum() >= 20:
                return candidate

        candidate = periods >= min_periods
        if candidate.sum() >= 20:
            return candidate

    time = get_column(data, 'star_age_day', get_column(data, 'star_age', None))
    if time is not None:
        time = np.asarray(time, dtype=float)
        if single_cycle:
            candidate = (time > 3.0) & (time < 4.0)
            if candidate.sum() >= 20:
                return candidate
        candidate = time > 3.0
        if candidate.sum() >= 20:
            return candidate

    return np.ones(n, dtype=bool)



def finite_mask(*arrays):
    """Return mask for entries finite in all arrays."""
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask



def signed_unit_amplitude(x):
    """Center a series and scale it by its maximum absolute deviation."""
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    peak = np.nanmax(np.abs(x))
    if not np.isfinite(peak) or peak == 0.0:
        return np.zeros_like(x)
    return x / peak



def dln_flux_from_mag(mag):
    """
    Convert magnitudes to relative log-flux deviations.

    Since m = -2.5 log10(F) + const,
    ln F = -0.4 ln(10) * m + const.
    """
    mag = np.asarray(mag, dtype=float)
    dmag = mag - np.nanmean(mag)
    return -0.4 * np.log(10.0) * dmag



def pearson_r(x, y):
    """Simple Pearson correlation coefficient."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return np.nan
    x = x - np.nanmean(x)
    y = y - np.nanmean(y)
    denom = np.sqrt(np.nansum(x * x) * np.nansum(y * y))
    if denom == 0.0:
        return np.nan
    return np.nansum(x * y) / denom



def add_regression_line(ax, x, y, color='k', ls='--', lw=1.2):
    """Add least-squares regression line if possible."""
    if len(x) < 2:
        return
    coeffs = np.polyfit(x, y, 1)
    xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    yy = coeffs[0] * xx + coeffs[1]
    ax.plot(xx, yy, color=color, ls=ls, lw=lw, alpha=0.8)


# =============================================================================
# Figure 1: Multi-band Light Curves vs Phase
# =============================================================================
def plot_lightcurve(data, output_dir=OUTPUT_DIR):
    """Plot multi-band light curves vs pulsation phase."""
    fig, ax = plt.subplots(figsize=(7, 5))

    phase = get_phase(data)
    mag_cols = find_magnitude_columns(data)
    if not mag_cols:
        print('Warning: No magnitude columns found.')
        return

    print(f'Found magnitude columns: {mag_cols}')

    band_info = {
        'B': {'color': '#0066CC', 'offset': 0.0, 'label': '$B$'},
        'V': {'color': '#33AA33', 'offset': 1.2, 'label': '$V$'},
        'I': {'color': '#990033', 'offset': 2.4, 'label': '$I$'},
        'J': {'color': '#6B1FB1', 'offset': 3.6, 'label': '$J$'},
    }

    mask = get_settled_cycle_mask(data, min_periods=3.0, single_cycle=False)
    if mask.sum() < 20:
        mask = np.ones(len(data), dtype=bool)

    for band in ['B', 'V', 'I', 'J']:
        if band in mag_cols:
            mag = data[mag_cols[band]][mask]
            ph = phase[mask]
            sort_idx = np.argsort(ph)
            info = band_info[band]
            ax.plot(ph[sort_idx], mag[sort_idx] + info['offset'],
                    '.', ms=1.5, alpha=0.6, color=info['color'], label=info['label'])

    ax.set_xlabel('Pulsation Phase')
    ax.set_ylabel('Magnitude + offset')
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.legend(loc='upper right', markerscale=4, framealpha=0.9)
    ax.text(0.02, 0.02, 'Offsets: +1.2 mag between bands',
            transform=ax.transAxes, fontsize=8, alpha=0.7)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    output_file = output_dir / 'rsp_lightcurve.pdf'
    plt.savefig(output_file)
    plt.savefig(str(output_file).replace('.pdf', '.png'), dpi=150)
    plt.close()
    print(f'Saved {output_file}')




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

    # The plotting below writes into subdirectories. Create them
    # without changing any plot formatting or plotting logic.
    (output_dir / 'cmd').mkdir(parents=True, exist_ok=True)
    (output_dir / 'cc').mkdir(parents=True, exist_ok=True)



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

    if mask.sum() < 5:
        # Fall back to using phase
        unique_phases = []
        unique_idx = []
        for i, p in enumerate(phase):
            if len(unique_phases) == 0 or abs(p - unique_phases[-1]) > 0.001:
                unique_phases.append(p)
                unique_idx.append(i)

        mask = np.zeros(len(data), dtype=bool)
        mask[unique_idx[:10000]] = True

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
    """Plot pulsation amplitude vs wavelength, now including J when available."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    mag_cols = find_magnitude_columns(data)

    wl_plot, amp_plot, col_plot, labels = [], [], [], []
    for band in ['U', 'B', 'V', 'R', 'I', 'J']:
        if band in mag_cols and band in BAND_WAVELENGTHS:
            mag = np.asarray(data[mag_cols[band]], dtype=float)
            valid = np.isfinite(mag)
            if valid.sum() < 2:
                continue
            amp = np.nanmax(mag[valid]) - np.nanmin(mag[valid])
            wl_plot.append(BAND_WAVELENGTHS[band])
            amp_plot.append(amp)
            col_plot.append(BAND_COLORS[band])
            labels.append(band)

    if not wl_plot:
        print('No magnitude data available for amplitude spectrum')
        plt.close()
        return

    order = np.argsort(wl_plot)
    wl_plot = np.asarray(wl_plot)[order]
    amp_plot = np.asarray(amp_plot)[order]
    col_plot = np.asarray(col_plot, dtype=object)[order]
    labels = np.asarray(labels, dtype=object)[order]

    ax.scatter(wl_plot, amp_plot, c=col_plot, s=100, zorder=5, edgecolors='k')
    ax.plot(wl_plot, amp_plot, 'k--', alpha=0.4, zorder=1)

    for w, a, l in zip(wl_plot, amp_plot, labels):
        ax.annotate(l, (w, a), xytext=(5, 5), textcoords='offset points', fontsize=10)

    ax.set_xlabel('Effective Wavelength (Å)')
    ax.set_ylabel('Pulsation Amplitude (mag)')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    output_file = output_dir / 'rsp_amplitudes.pdf'
    plt.savefig(output_file)
    plt.savefig(str(output_file).replace('.pdf', '.png'), dpi=150)
    plt.close()
    print(f'Saved {output_file}')


# =============================================================================
# New diagnostic plots: radius vs IR, temperature vs optical color
# =============================================================================
def plot_ir_radius_diagnostic(data, output_dir=OUTPUT_DIR):
    """
    Compare radius sensitivity in J (expected stronger) against B (null comparison).

    Top row:  J-band flux vs radius.
    Bottom row: B-band flux vs radius.
    """
    mag_cols = find_magnitude_columns(data)
    missing = [band for band in ['B', 'J'] if band not in mag_cols]
    if missing:
        print(f"Skipping IR-radius diagnostic: missing bands {missing}.")
        return
    if 'photosphere_r' not in data.dtype.names:
        print('Skipping IR-radius diagnostic: no photosphere_r column found.')
        return

    phase = get_phase(data)
    mask = get_settled_cycle_mask(data, min_periods=4.0, single_cycle=True)

    radius = np.asarray(data['photosphere_r'], dtype=float)
    bmag = np.asarray(data[mag_cols['B']], dtype=float)
    jmag = np.asarray(data[mag_cols['J']], dtype=float)

    mask &= finite_mask(phase, radius, bmag, jmag)
    if mask.sum() < 20:
        print('Skipping IR-radius diagnostic: too few valid points.')
        return

    ph = phase[mask]
    r = radius[mask]
    b = bmag[mask]
    j = jmag[mask]

    order = np.argsort(ph)
    ph = ph[order]
    r = r[order]
    b = b[order]
    j = j[order]

    two_dlnR = 2.0 * (np.log(r) - np.mean(np.log(r)))
    flux_map = {
        'J': dln_flux_from_mag(j),
        'B': dln_flux_from_mag(b),
    }

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.8), sharex='col')
    phase_stats = {}
    direct_stats = {}

    for row, band in enumerate(['J', 'B']):
        dlnF = flux_map[band]
        phase_stats[band] = pearson_r(signed_unit_amplitude(two_dlnR), signed_unit_amplitude(dlnF))
        direct_stats[band] = pearson_r(two_dlnR, dlnF)

        ax = axes[row, 0]
        ax.plot(ph, signed_unit_amplitude(two_dlnR), '-', lw=2.0, color='black', label=r'$2\Delta\ln R$')
        ax.plot(ph, signed_unit_amplitude(dlnF), '-', lw=2.0, color=BAND_COLORS[band],
                label=rf'$\Delta\ln F_{{{band}}}$')
        ax.set_ylabel('Normalized variation')
        ax.set_title(f'{band}-band flux and radius vs phase')
        ax.grid(True, alpha=0.2)
        ax.legend(framealpha=0.9)
        ax.text(0.03, 0.04, f'phase-shape r = {phase_stats[band]:.3f}', transform=ax.transAxes, fontsize=9)
        if row == 1:
            ax.set_xlabel('Pulsation Phase')

        ax = axes[row, 1]
        sc = ax.scatter(two_dlnR, dlnF, c=ph, cmap='twilight_shifted', s=18, alpha=0.9, edgecolors='none')
        add_regression_line(ax, two_dlnR, dlnF, color='black')
        ax.set_ylabel(rf'$\Delta\ln F_{{{band}}}$')
        ax.set_title(f'{band}-band flux directly vs radius')
        ax.grid(True, alpha=0.2)
        ax.text(0.03, 0.04, f'Pearson r = {direct_stats[band]:.3f}', transform=ax.transAxes, fontsize=9)
        if row == 1:
            ax.set_xlabel(r'$2\Delta\ln R$')
        cbar = plt.colorbar(sc, ax=ax, shrink=0.88)
        cbar.set_label('Pulsation Phase')

    fig.suptitle('Radius diagnostic: J response compared against optical null (B)', y=0.995, fontsize=13)
    plt.tight_layout()
    output_file = output_dir / 'rsp_ir_radius_diagnostic.pdf'
    plt.savefig(output_file)
    plt.savefig(str(output_file).replace('.pdf', '.png'), dpi=150)
    plt.close()
    print(f'Saved {output_file}')



def plot_optical_temperature_diagnostic(data, output_dir=OUTPUT_DIR):
    """
    Compare temperature sensitivity in B (expected stronger) against J (null comparison).

    Top row:    temperature benchmark with B-band flux.
    Bottom row: same temperature benchmark with J-band flux.
    """
    mag_cols = find_magnitude_columns(data)
    missing = [band for band in ['B', 'J'] if band not in mag_cols]
    if missing:
        print(f"Skipping optical-temperature diagnostic: missing bands {missing}.")
        return
    if 'effective_T' not in data.dtype.names:
        print('Skipping optical-temperature diagnostic: no effective_T column found.')
        return

    phase = get_phase(data)
    mask = get_settled_cycle_mask(data, min_periods=4.0, single_cycle=True)

    teff = np.asarray(data['effective_T'], dtype=float)
    bmag = np.asarray(data[mag_cols['B']], dtype=float)
    jmag = np.asarray(data[mag_cols['J']], dtype=float)
    color_bj = bmag - jmag

    mask &= finite_mask(phase, teff, color_bj, bmag, jmag)
    if mask.sum() < 20:
        print('Skipping optical-temperature diagnostic: too few valid points.')
        return

    ph = phase[mask]
    teff = teff[mask]
    bmag = bmag[mask]
    jmag = jmag[mask]
    color_bj = color_bj[mask]

    order = np.argsort(ph)
    ph = ph[order]
    teff = teff[order]
    bmag = bmag[order]
    jmag = jmag[order]
    color_bj = color_bj[order]

    dlnT = np.log(teff) - np.mean(np.log(teff))
    minus_dcolor = -(color_bj - np.mean(color_bj))
    flux_map = {
        'B': dln_flux_from_mag(bmag),
        'J': dln_flux_from_mag(jmag),
    }

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.8), sharex='col')
    phase_stats = {}
    direct_stats = {}
    color_stats = pearson_r(teff, color_bj)

    for row, band in enumerate(['B', 'J']):
        dlnF = flux_map[band]
        phase_stats[band] = pearson_r(signed_unit_amplitude(dlnT), signed_unit_amplitude(dlnF))
        direct_stats[band] = pearson_r(dlnT, dlnF)

        ax = axes[row, 0]
        ax.plot(ph, signed_unit_amplitude(dlnT), '-', lw=2.0, color='black', label=r'$\Delta\ln T_{\rm eff}$')
        ax.plot(ph, signed_unit_amplitude(minus_dcolor), '-', lw=2.0,
                color=BAND_COLORS['B'], label=r'$-\Delta(B-J)$')
        ax.plot(ph, signed_unit_amplitude(dlnF), '--', lw=1.9,
                color=BAND_COLORS[band], label=rf'$\Delta\ln F_{{{band}}}$')
        ax.set_ylabel('Normalized variation')
        ax.set_title(f'Temperature benchmark with {band}-band response')
        ax.grid(True, alpha=0.2)
        ax.legend(framealpha=0.9)
        ax.text(0.03, 0.04,
                f'phase-shape r(T, {band}) = {phase_stats[band]:.3f}\n'
                f'color-temp r = {color_stats:.3f}',
                transform=ax.transAxes, fontsize=9)
        if row == 1:
            ax.set_xlabel('Pulsation Phase')

        ax = axes[row, 1]
        sc = ax.scatter(dlnT, dlnF, c=ph, cmap='twilight_shifted', s=18, alpha=0.9, edgecolors='none')
        add_regression_line(ax, dlnT, dlnF, color='black')
        ax.set_ylabel(rf'$\Delta\ln F_{{{band}}}$')
        ax.set_title(f'{band}-band flux directly vs temperature')
        ax.grid(True, alpha=0.2)
        ax.text(0.03, 0.04, f'Pearson r = {direct_stats[band]:.3f}', transform=ax.transAxes, fontsize=9)
        if row == 1:
            ax.set_xlabel(r'$\Delta\ln T_{\rm eff}$')
        cbar = plt.colorbar(sc, ax=ax, shrink=0.88)
        cbar.set_label('Pulsation Phase')

    fig.suptitle('Temperature diagnostic: B response compared against IR null (J)', y=0.995, fontsize=13)
    plt.tight_layout()
    output_file = output_dir / 'rsp_optical_temperature_diagnostic.pdf'
    plt.savefig(output_file)
    plt.savefig(str(output_file).replace('.pdf', '.png'), dpi=150)
    plt.close()
    print(f'Saved {output_file}')


# =============================================================================
# Statistics
# =============================================================================
def print_statistics(data):
    """Print useful statistics about the run."""
    mag_cols = find_magnitude_columns(data)

    print('\n' + '=' * 60)
    print('RSP + Colors Run Statistics')
    print('=' * 60)
    print(f'Total models: {len(data)}')

    if 'rsp_num_periods' in data.dtype.names:
        n_periods = int(np.nanmax(data['rsp_num_periods']))
        print(f'Pulsation cycles completed: {n_periods}')
        print(f'Points per cycle: ~{len(data) // max(n_periods, 1)}')

    if 'rsp_period_in_days' in data.dtype.names:
        periods = np.asarray(data['rsp_period_in_days'], dtype=float)
        valid = periods > 0
        if valid.any():
            print(f'Mean period: {np.mean(periods[valid]):.5f} days')

    print('\nMagnitude ranges:')
    for band in ['B', 'V', 'R', 'I', 'J']:
        if band in mag_cols:
            mag = np.asarray(data[mag_cols[band]], dtype=float)
            valid = np.isfinite(mag)
            if valid.any():
                print(f'  {band}: {np.nanmin(mag):.3f} to {np.nanmax(mag):.3f} '
                      f'(Δ = {np.nanmax(mag) - np.nanmin(mag):.3f} mag)')

    if 'B' in mag_cols and 'J' in mag_cols:
        color = np.asarray(data[mag_cols['B']], dtype=float) - np.asarray(data[mag_cols['J']], dtype=float)
        valid = np.isfinite(color)
        if valid.any():
            print(f'\n(B-J) color range: {np.nanmin(color):.3f} to {np.nanmax(color):.3f}')

    print('=' * 60 + '\n')




def print_post_settling_cycle_statistics(data, output_dir=OUTPUT_DIR, min_periods=4.0, min_points=20):
    """
    Print and save per-cycle statistics for all cycles after settling.

    This does not alter any plotting. It only summarizes the history columns
    already used by the plots, grouping rows by floor(rsp_num_periods).
    """
    if 'rsp_num_periods' not in data.dtype.names:
        print('Skipping per-cycle statistics: no rsp_num_periods column found.')
        return

    periods = np.asarray(data['rsp_num_periods'], dtype=float)
    base_mask = np.isfinite(periods) & (periods >= min_periods)
    if base_mask.sum() < min_points:
        print(f'Skipping per-cycle statistics: fewer than {min_points} post-settling points.')
        return

    cycle_id = np.floor(periods).astype(int)
    mag_cols = find_magnitude_columns(data)

    rows = []
    candidate_cycles = np.unique(cycle_id[base_mask])
    for cyc in candidate_cycles:
        mask = base_mask & (cycle_id == cyc)
        if mask.sum() < min_points:
            continue

        row = {
            'cycle': int(cyc),
            'n_points': int(mask.sum()),
            'rsp_num_periods_min': float(np.nanmin(periods[mask])),
            'rsp_num_periods_max': float(np.nanmax(periods[mask])),
        }

        if 'effective_T' in data.dtype.names:
            teff = np.asarray(data['effective_T'], dtype=float)[mask]
            good = np.isfinite(teff)
            if good.any():
                row['Teff_mean'] = float(np.nanmean(teff[good]))
                row['Teff_amp'] = float(np.nanmax(teff[good]) - np.nanmin(teff[good]))

        if 'photosphere_r' in data.dtype.names:
            radius = np.asarray(data['photosphere_r'], dtype=float)[mask]
            good = np.isfinite(radius)
            if good.any():
                row['R_mean'] = float(np.nanmean(radius[good]))
                row['R_amp'] = float(np.nanmax(radius[good]) - np.nanmin(radius[good]))

        for band in ['U', 'B', 'V', 'R', 'I', 'J', 'M']:
            if band not in mag_cols:
                continue
            mag = np.asarray(data[mag_cols[band]], dtype=float)[mask]
            good = np.isfinite(mag)
            if good.any():
                row[f'{band}_mean'] = float(np.nanmean(mag[good]))
                row[f'{band}_amp'] = float(np.nanmax(mag[good]) - np.nanmin(mag[good]))

        if 'B' in mag_cols and 'J' in mag_cols:
            bj = (np.asarray(data[mag_cols['B']], dtype=float)
                  - np.asarray(data[mag_cols['J']], dtype=float))[mask]
            good = np.isfinite(bj)
            if good.any():
                row['BminusJ_mean'] = float(np.nanmean(bj[good]))
                row['BminusJ_amp'] = float(np.nanmax(bj[good]) - np.nanmin(bj[good]))

        rows.append(row)

    print('\n' + '=' * 60)
    print(f'Post-settling per-cycle statistics: rsp_num_periods >= {min_periods:g}')
    print('=' * 60)

    if not rows:
        print(f'No complete post-settling cycles found with at least {min_points} points.')
        print('=' * 60 + '\n')
        return

    print(f'Cycles summarized: {rows[0]["cycle"]} to {rows[-1]["cycle"]} ({len(rows)} cycles)')
    print(f'Post-settling points used: {sum(row["n_points"] for row in rows)}')

    for key, label in [
        ('B_amp', 'B amplitude'),
        ('V_amp', 'V amplitude'),
        ('I_amp', 'I amplitude'),
        ('J_amp', 'J amplitude'),
        ('BminusJ_amp', '(B-J) color amplitude'),
        ('Teff_amp', 'Teff amplitude'),
        ('R_amp', 'Radius amplitude'),
    ]:
        vals = np.asarray([row[key] for row in rows if key in row], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            print(f'  {label}: {np.nanmean(vals):.6g} +/- {np.nanstd(vals):.3g}')

    stats_dir = output_dir / 'stats'
    stats_dir.mkdir(parents=True, exist_ok=True)
    csv_file = stats_dir / 'rsp_cycle_stats.csv'

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Saved {csv_file}')
    print('=' * 60 + '\n')

# =============================================================================
# Main
# =============================================================================
def main():
    """Main analysis routine."""
    print('=' * 60)
    print('MESA Colors RSP RR Lyrae Analysis')
    print('=' * 60)

    history_file = DEFAULT_LOGS / 'history.data'
    if not history_file.exists():
        if len(sys.argv) >= 2:
            history_file = Path(sys.argv[1])
        else:
            print(f'ERROR: History file not found: {history_file}')
            print('Usage: python plot_rsp_radius_temp.py [LOGS/history.data]')
            sys.exit(1)

    if not history_file.exists():
        print(f'Error: File not found: {history_file}')
        sys.exit(1)

    print(f'Loading {history_file}...')
    data = load_mesa_history(str(history_file))
    print(f'Loaded {len(data)} models')
    print(f'Columns: {", ".join(data.dtype.names[:12])}...')

    print_statistics(data)
    print_post_settling_cycle_statistics(data)

    print('Generating figures...')
    plot_ir_radius_diagnostic(data)
    plot_optical_temperature_diagnostic(data)
    plot_lightcurve(data)
    plot_colorloop(data)
    plot_amplitude_spectrum(data)

    print(f'\nDone! Figures saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
