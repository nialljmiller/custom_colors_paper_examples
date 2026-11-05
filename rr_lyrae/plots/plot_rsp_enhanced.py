#!/usr/bin/env python3
"""
plot_rsp.py - Analysis plots for MESA Colors RSP RR Lyrae demonstration

Generates publication-quality figures showing phase-resolved synthetic
photometry during radial stellar pulsation of an RR Lyrae variable.

Produces (saved to this directory):
1. rsp_lightcurve            - Multi-band light curves vs pulsation phase
2. rsp_colorloop             - Color-magnitude loop (B-V vs V)
3. rsp_amplitudes            - Pulsation amplitude vs wavelength
4. rsp_sed_movie.mp4 / .gif  - Two-panel movie:
                               left = history + Newton-iteration output
                               right = evolving SED

Usage:
    cd rsp_RR_Lyrae/plots
    python plot_rsp.py
    python plot_rsp.py --sed_dir ../SED --newton_file ../LOGS/rsp_newton.data

Author: Niall Miller (2025) - for Miller, Joyce, Mocz et al.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
DEFAULT_SED = PARENT_DIR / 'SED'
OUTPUT_DIR = SCRIPT_DIR  # Save plots right here

JOHNSON_FILTERS = {
    'U': {'wavelength': 3650, 'color': '#4B0082'},
    'B': {'wavelength': 4400, 'color': '#0066CC'},
    'V': {'wavelength': 5500, 'color': '#33AA33'},
    'R': {'wavelength': 6400, 'color': '#CC6600'},
    'I': {'wavelength': 8000, 'color': '#990033'},
}

EM_REGIONS = [
    (100, 2000, '#4B0082'),
    (2000, 4000, '#9400D3'),
    (4000, 4500, '#0000FF'),
    (4500, 5500, '#00FF00'),
    (5500, 7000, '#FFFF00'),
    (7000, 10000, '#FF0000'),
    (10000, 25000, '#8B0000'),
]

NEWTON_FILE_PATTERNS = [
    '*newton*.data', '*newton*.txt', '*newton*.csv',
    '*iter*.data', '*iter*.txt', '*iter*.csv',
    '*rsp*.data', '*rsp*.txt', '*rsp*.csv',
    '*hydro*.data', '*hydro*.txt', '*hydro*.csv',
]

NEWTON_Y_PRIORITY = [
    'rsp_surface_v', 'v_surf_km_s', 'v_surf', 'u_surf', 'velocity',
    'R', 'radius', 'photosphere_r', 'r_phot',
    'log_Teff', 'Teff', 'T_eff',
    'log_L', 'luminosity', 'L',
    'delta_r', 'dr', 'rel_E_err'
]


# =============================================================================
# Data Loading
# =============================================================================
def load_mesa_history(filepath):
    """
    Load MESA history.data file.
    Returns structured numpy array with column names as attributes.
    """
    return load_numeric_table(filepath)


def load_numeric_table(filepath):
    """Load a whitespace-delimited MESA-like table with a header row."""
    filepath = Path(filepath)
    with open(filepath, 'r') as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        tokens = stripped.split()
        # Look for typical MESA header style or anything that is obviously not data
        if 'model_number' in tokens or any(any(c.isalpha() for c in tok) for tok in tokens):
            header_idx = i
            break

    if header_idx is None:
        header_idx = 5 if len(lines) > 5 else 0

    cols = lines[header_idx].split()
    data = np.genfromtxt(filepath, skip_header=header_idx + 1, names=cols,
                         invalid_raise=False, autostrip=True)

    if getattr(data, 'shape', ()) == ():
        data = np.array([data], dtype=data.dtype)

    return data


def load_sed_csv(filepath):
    """
    Load SED data from CSV file.

    Reads wavelength/flux pairs row-by-row so arrays stay aligned.
    Rows where wavelength is zero (padding) are skipped.
    """
    try:
        wavelengths = []
        fluxes = []
        convolved = []

        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    wl = float(row['wavelengths'])
                    fl = float(row['fluxes'])
                except (ValueError, KeyError):
                    continue
                if wl == 0:
                    continue
                wavelengths.append(wl)
                fluxes.append(fl)
                try:
                    convolved.append(float(row['convolved_flux']))
                except (ValueError, KeyError):
                    convolved.append(0.0)

        return {
            'wavelengths': np.array(wavelengths),
            'fluxes': np.array(fluxes),
            'convolved_flux': np.array(convolved),
        }
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
        return {}


def load_all_seds(sed_dir, preferred_prefix='V'):
    """Load one SED CSV per model number from sed_dir."""
    sed_dir = Path(sed_dir)
    all_files = sorted(sed_dir.glob('*_SED_*.csv'))

    by_model = defaultdict(list)
    for f in all_files:
        parts = f.stem.split('_')
        try:
            model_num = int(parts[-1])
        except ValueError:
            continue
        by_model[model_num].append(f)

    seds = []
    for model_num in sorted(by_model):
        candidates = by_model[model_num]
        chosen = None
        if preferred_prefix:
            for c in candidates:
                if c.name.startswith(preferred_prefix + '_'):
                    chosen = c
                    break
        if chosen is None:
            chosen = candidates[0]

        data = load_sed_csv(chosen)
        if data and 'wavelengths' in data and len(data['wavelengths']) > 0:
            seds.append((model_num, data))

    return seds


def find_magnitude_columns(data):
    """
    Find magnitude column names in the history data.
    MESA Colors may use different naming conventions.
    """
    colnames = data.dtype.names

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


def get_time_days(data):
    """Return the best available time column in days."""
    if 'star_age_day' in data.dtype.names:
        return data['star_age_day']
    if 'time_days' in data.dtype.names:
        return data['time_days']
    if 'star_age' in data.dtype.names:
        return data['star_age']
    return None


def get_phase(data):
    """Return phase array, computing it from time if needed."""
    if 'rsp_phase' in data.dtype.names:
        return data['rsp_phase']

    time = get_time_days(data)
    if time is None or len(time) == 0:
        return np.linspace(0.0, 1.0, len(data), endpoint=False)

    period = 0.71
    if 'rsp_period_in_days' in data.dtype.names:
        valid = np.isfinite(data['rsp_period_in_days']) & (data['rsp_period_in_days'] > 0)
        if np.any(valid):
            period = np.nanmedian(data['rsp_period_in_days'][valid])

    return ((time - time[0]) / period) % 1.0


# =============================================================================
# Figure 1: Multi-band Light Curves vs Phase
# =============================================================================
def plot_lightcurve(data, output_dir=OUTPUT_DIR):
    """Plot multi-band light curves vs pulsation phase."""
    fig, ax = plt.subplots(figsize=(7, 5))

    phase = get_phase(data)
    mag_cols = find_magnitude_columns(data)

    if not mag_cols:
        print("Warning: No magnitude columns found.")
        print("Available columns:")
        print(data.dtype.names)
        return

    print(f"Found magnitude columns: {mag_cols}")

    band_info = {
        'B': {'color': '#0066CC', 'offset': 0.0, 'label': '$B$'},
        'V': {'color': '#33AA33', 'offset': 1.2, 'label': '$V$'},
        'R': {'color': '#CC6600', 'offset': 2.4, 'label': '$R$'},
        'I': {'color': '#990033', 'offset': 3.6, 'label': '$I$'},
    }

    if 'rsp_num_periods' in data.dtype.names:
        mask = (data['rsp_num_periods'] >= 3) & (data['rsp_num_periods'] <= 5)
    else:
        time = get_time_days(data)
        if time is None:
            mask = np.ones(len(data), dtype=bool)
        else:
            mask = (time > 2.0) & (time < 4.5)

    if mask.sum() < 100:
        print("Warning: Few points in selected range, using all data")
        mask = np.ones(len(data), dtype=bool)

    for band in ['B', 'V', 'R', 'I']:
        if band in mag_cols:
            col = mag_cols[band]
            mag = data[col][mask]
            ph = phase[mask]
            info = band_info[band]
            sort_idx = np.argsort(ph)

            ax.plot(ph[sort_idx], mag[sort_idx] + info['offset'],
                    '.', ms=1.5, alpha=0.6, color=info['color'],
                    label=info['label'])

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
    fig, ax = plt.subplots(figsize=(5, 5))

    mag_cols = find_magnitude_columns(data)

    if 'B' not in mag_cols or 'V' not in mag_cols:
        print("Error: Need B and V magnitudes for color loop")
        return

    B = data[mag_cols['B']]
    V = data[mag_cols['V']]
    color = B - V
    phase = get_phase(data)

    if 'rsp_num_periods' in data.dtype.names:
        mask = (data['rsp_num_periods'] >= 4)
    else:
        time = get_time_days(data)
        mask = (time > 3.0) if time is not None else np.ones(len(data), dtype=bool)

    if mask.sum() < 50:
        mask = np.ones(len(data), dtype=bool)
        unique_phases = []
        unique_idx = []
        for i, p in enumerate(phase):
            if len(unique_phases) == 0 or abs(p - unique_phases[-1]) > 0.001:
                unique_phases.append(p)
                unique_idx.append(i)

        mask = np.zeros(len(data), dtype=bool)
        mask[unique_idx[:1000]] = True

    sc = ax.scatter(color[mask], V[mask],
                    c=phase[mask], cmap='twilight_shifted',
                    s=8, alpha=0.8, edgecolors='none')

    plt.colorbar(sc, ax=ax, label='Pulsation Phase', shrink=0.8)

    c_arr = color[mask]
    v_arr = V[mask]
    p_arr = phase[mask]

    idx_arrow = np.argmin(np.abs(p_arr - 0.25))
    idx_arrow2 = np.argmin(np.abs(p_arr - 0.30))

    if idx_arrow != idx_arrow2:
        ax.annotate('',
                    xy=(c_arr[idx_arrow2], v_arr[idx_arrow2]),
                    xytext=(c_arr[idx_arrow], v_arr[idx_arrow]),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    ax.set_xlabel('$B - V$ (mag)')
    ax.set_ylabel('$V$ (mag)')
    ax.invert_yaxis()

    for target_phase, label in [(0.0, 'max'), (0.5, 'min')]:
        idx = np.argmin(np.abs(p_arr - target_phase))
        ax.plot(c_arr[idx], v_arr[idx], 'k*', ms=10, zorder=10)

    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    output_file = output_dir / 'rsp_colorloop.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_file).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_file}")


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
            amp = np.nanmax(mag) - np.nanmin(mag)
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
# Movie helpers
# =============================================================================
def add_em_spectrum_regions(ax, alpha=0.03):
    """Add faint colored background regions for the EM spectrum."""
    xlim = ax.get_xlim()
    for xmin, xmax, color in EM_REGIONS:
        if xmax < xlim[0] or xmin > xlim[1]:
            continue
        ax.axvspan(max(xmin, xlim[0]), min(xmax, xlim[1]),
                   color=color, alpha=alpha, zorder=0)


def select_movie_cycle(data, preferred_cycle=4):
    """Select one representative pulsation cycle for the movie."""
    phase = get_phase(data)
    time_days = get_time_days(data)
    n = len(data)

    if 'rsp_num_periods' in data.dtype.names:
        cycles = data['rsp_num_periods']
        full_cycles = sorted({int(np.floor(c)) for c in cycles if np.isfinite(c)})
        full_cycles = [c for c in full_cycles if np.sum((cycles >= c) & (cycles < c + 1)) > 10]
        if full_cycles:
            if preferred_cycle in full_cycles:
                target = preferred_cycle
            else:
                target = full_cycles[min(len(full_cycles) - 1, max(0, len(full_cycles) // 2))]
            mask = (cycles >= target) & (cycles < target + 1)
            idx = np.where(mask)[0]
            if len(idx) > 10:
                return idx

    # Fallback: use phase wrapping to find a clean cycle after the first ~25% of the run
    start_search = max(1, n // 4)
    wrap_idx = np.where(np.diff(phase) < -0.5)[0]
    wrap_idx = wrap_idx[wrap_idx >= start_search]
    if len(wrap_idx) >= 2:
        i0 = wrap_idx[min(1, len(wrap_idx) - 2)] + 1
        i1 = wrap_idx[min(2, len(wrap_idx) - 1)] + 1
        idx = np.arange(i0, i1)
        if len(idx) > 10:
            return idx

    # Last resort: take the last ~15% of the run
    i0 = max(0, int(0.85 * n) - max(50, n // 10))
    return np.arange(i0, n)


def find_newton_output_file(logs_dir, explicit=None):
    """Find a plausible Newton-iteration output file."""
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None

    search_dirs = [Path(logs_dir), PARENT_DIR, SCRIPT_DIR]
    candidates = []
    seen = set()

    for base in search_dirs:
        if not base.exists():
            continue
        for pattern in NEWTON_FILE_PATTERNS:
            for path in base.glob(pattern):
                if not path.is_file():
                    continue
                if path.name in {'history.data', 'profiles.index'}:
                    continue
                if path in seen:
                    continue
                seen.add(path)
                score = 0
                lower = path.name.lower()
                if 'newton' in lower:
                    score += 50
                if 'iter' in lower:
                    score += 30
                if 'rsp' in lower:
                    score += 10
                candidates.append((score, path))

    for _, path in sorted(candidates, key=lambda t: (-t[0], str(t[1]))):
        try:
            data = load_numeric_table(path)
        except Exception:
            continue
        if getattr(data, 'dtype', None) is None or data.dtype.names is None:
            continue
        if len(data.dtype.names) < 2 or len(data) < 10:
            continue
        return path

    return None


def detect_newton_axes(newton_data):
    """Pick a sensible x column and up to three y columns from a Newton table."""
    names = list(newton_data.dtype.names)
    lower_to_name = {name.lower(): name for name in names}

    x_candidates = [
        'star_age_day', 'time_days', 'time', 't', 'rsp_phase', 'phase',
        'iteration', 'iter', 'iters', 'i', 'model_number'
    ]
    x_name = None
    for cand in x_candidates:
        if cand in lower_to_name:
            x_name = lower_to_name[cand]
            break
    if x_name is None:
        x_name = names[0]

    y_names = []
    for cand in NEWTON_Y_PRIORITY:
        for name in names:
            if name.lower() == cand.lower() and name != x_name and name not in y_names:
                y_names.append(name)
                break
    if not y_names:
        for name in names:
            if name != x_name:
                y_names.append(name)
            if len(y_names) == 3:
                break

    return x_name, y_names[:3]


def prepare_newton_panel(newton_data, history_data, cycle_idx):
    """Prepare Newton-iteration data for plotting alongside a selected cycle."""
    if newton_data is None:
        return None

    x_name, y_names = detect_newton_axes(newton_data)
    if not y_names:
        return None

    x = np.array(newton_data[x_name], dtype=float)
    valid = np.isfinite(x)
    if not np.any(valid):
        return None

    x_kind = 'generic'
    if x_name.lower() in {'rsp_phase', 'phase'}:
        x_kind = 'phase'
    elif x_name.lower() in {'star_age_day', 'time_days', 'time', 't'}:
        x_kind = 'time'
    elif x_name.lower() in {'iteration', 'iter', 'iters', 'i'}:
        x_kind = 'iter'
    elif x_name.lower() == 'model_number':
        x_kind = 'model'

    hist_phase = get_phase(history_data)[cycle_idx]
    hist_time = get_time_days(history_data)
    hist_time = hist_time[cycle_idx] if hist_time is not None else None
    hist_models = history_data['model_number'][cycle_idx] if 'model_number' in history_data.dtype.names else None

    mask = valid.copy()
    if x_kind == 'phase':
        mask &= (x >= -0.05) & (x <= 1.05)
    elif x_kind == 'time' and hist_time is not None:
        xmin = np.nanmin(hist_time)
        xmax = np.nanmax(hist_time)
        pad = 0.05 * max(xmax - xmin, 1e-6)
        mask &= (x >= xmin - pad) & (x <= xmax + pad)
    elif x_kind == 'model' and hist_models is not None:
        xmin = np.nanmin(hist_models)
        xmax = np.nanmax(hist_models)
        mask &= (x >= xmin) & (x <= xmax)

    if np.sum(mask) < 10:
        mask = valid

    x_plot = x[mask]
    sort_idx = np.argsort(x_plot)
    x_plot = x_plot[sort_idx]

    y_plot = {}
    for name in y_names:
        y = np.array(newton_data[name], dtype=float)[mask][sort_idx]
        if np.isfinite(y).sum() >= 3:
            y_plot[name] = y

    if not y_plot:
        return None

    return {
        'x_name': x_name,
        'x_kind': x_kind,
        'x': x_plot,
        'ys': y_plot,
    }


def map_history_frame_to_newton_x(frame_in_cycle, history_data, cycle_idx, newton_panel):
    """Map the current history frame to the Newton panel x coordinate."""
    if newton_panel is None:
        return None

    hidx = cycle_idx[frame_in_cycle]
    x_kind = newton_panel['x_kind']

    if x_kind == 'phase':
        return float(get_phase(history_data)[hidx])
    if x_kind == 'time':
        time_days = get_time_days(history_data)
        return float(time_days[hidx]) if time_days is not None else None
    if x_kind == 'model' and 'model_number' in history_data.dtype.names:
        return float(history_data['model_number'][hidx])

    # Generic/iteration fallback: use fractional progress through movie.
    x = newton_panel['x']
    if len(x) == 0:
        return None
    frac = frame_in_cycle / max(len(cycle_idx) - 1, 1)
    j = int(round(frac * (len(x) - 1)))
    return float(x[j])


def make_rsp_sed_movie(data, sed_dir, output_dir=OUTPUT_DIR, newton_file=None,
                       filename='rsp_sed_movie.mp4', fps=20, stride=1,
                       preferred_cycle=4, logs_dir=DEFAULT_LOGS):
    """
    Create a two-panel movie:
      left  - stacked history + Newton-iteration output
      right - evolving SED

    The movie uses one representative post-settling pulsation cycle.
    """
    sed_dir = Path(sed_dir)
    if not sed_dir.exists():
        print(f"Warning: SED directory not found: {sed_dir}. Skipping movie.")
        return

    all_seds = load_all_seds(sed_dir, preferred_prefix='V')
    if not all_seds:
        print("Warning: No SED files found. Skipping movie.")
        return

    if 'model_number' not in data.dtype.names:
        print("Warning: history.data has no model_number column. Skipping movie.")
        return

    cycle_idx = select_movie_cycle(data, preferred_cycle=preferred_cycle)
    cycle_idx = cycle_idx[::max(1, int(stride))]
    if len(cycle_idx) < 5:
        print("Warning: selected cycle is too short for movie. Skipping.")
        return

    model_to_hist_idx = {int(m): i for i, m in enumerate(data['model_number'].astype(int))}
    cycle_models = set(int(m) for m in data['model_number'][cycle_idx].astype(int))
    cycle_seds = [(mn, sed) for mn, sed in all_seds if mn in cycle_models]

    if not cycle_seds:
        print("Warning: No SED files matched the selected RSP cycle. Skipping movie.")
        return

    cycle_lookup = {int(data['model_number'][i]): pos for pos, i in enumerate(cycle_idx)}
    cycle_seds = sorted(cycle_seds, key=lambda item: cycle_lookup.get(item[0], 1_000_000))

    mag_cols = find_magnitude_columns(data)
    phase = get_phase(data)
    time_days = get_time_days(data)

    # Representative history panel data for one cycle
    history_phase = phase[cycle_idx]
    hist_bands = {}
    for band in ['B', 'V', 'R', 'I']:
        if band in mag_cols:
            hist_bands[band] = data[mag_cols[band]][cycle_idx]

    if not hist_bands:
        print("Warning: No B/V/R/I magnitudes found for movie. Skipping.")
        return

    # Newton iteration output (optional but auto-detected if present)
    newton_path = find_newton_output_file(logs_dir if newton_file is None else Path(newton_file).parent,
                                          explicit=newton_file)
    newton_data = None
    newton_panel = None
    if newton_path is not None:
        try:
            newton_data = load_numeric_table(newton_path)
            newton_panel = prepare_newton_panel(newton_data, data, cycle_idx)
            if newton_panel is not None:
                print(f"Using Newton output file: {newton_path}")
            else:
                print(f"Found Newton-like file but could not plot it cleanly: {newton_path}")
        except Exception as e:
            print(f"Warning: Could not load Newton output file {newton_path}: {e}")
    else:
        print("No Newton-iteration file found automatically; movie will still render.")

    # SED scaling
    wl_min, wl_max = 3000, 30000
    peak_fluxes = []
    for _, sed in cycle_seds:
        wl = sed['wavelengths']
        fl = sed['fluxes']
        mask = (wl >= wl_min) & (wl <= wl_max) & np.isfinite(fl)
        if mask.any() and np.nanmax(fl[mask]) > 0:
            peak_fluxes.append(np.nanmax(fl[mask]))
    fl_max = np.percentile(peak_fluxes, 99) * 1.15 if peak_fluxes else 1.0
    fl_max = max(fl_max, 1.0)

    # Figure layout
    fig = plt.figure(figsize=(9.0, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28,
                          left=0.08, right=0.98, top=0.90, bottom=0.12)
    gs_left = gs[0].subgridspec(2, 1, hspace=0.08, height_ratios=[1.0, 0.9])

    ax_hist = fig.add_subplot(gs_left[0])
    ax_newt = fig.add_subplot(gs_left[1])
    ax_sed = fig.add_subplot(gs[1])

    # Top-left: converged history over one cycle
    band_offsets = {'B': 0.0, 'V': 1.0, 'R': 2.0, 'I': 3.0}
    history_markers = {}
    for band in ['B', 'V', 'R', 'I']:
        if band not in hist_bands:
            continue
        y = hist_bands[band] + band_offsets.get(band, 0.0)
        sort_idx = np.argsort(history_phase)
        ax_hist.plot(history_phase[sort_idx], y[sort_idx], '-',
                     color=JOHNSON_FILTERS[band]['color'], lw=1.0, alpha=0.9,
                     label=band)
        marker, = ax_hist.plot([], [], 'o', ms=5, color=JOHNSON_FILTERS[band]['color'],
                               markeredgecolor='k', markeredgewidth=0.35, zorder=5)
        history_markers[band] = marker

    ax_hist.set_xlim(0, 1)
    ax_hist.set_ylabel('Magnitude + offset')
    ax_hist.invert_yaxis()
    ax_hist.legend(loc='upper right', ncol=4, fontsize=7, framealpha=0.9)
    ax_hist.grid(True, alpha=0.2)
    ax_hist.set_title('Converged history output', fontsize=10)
    plt.setp(ax_hist.get_xticklabels(), visible=False)

    # Bottom-left: Newton output if available
    newton_lines = []
    newton_cursor = None
    if newton_panel is not None:
        colors = ['#444444', '#d95f02', '#7570b3']
        for color, (name, y) in zip(colors, newton_panel['ys'].items()):
            line, = ax_newt.plot(newton_panel['x'], y, '-', lw=0.9, alpha=0.95,
                                 color=color, label=name)
            newton_lines.append(line)
        newton_cursor = ax_newt.axvline(newton_panel['x'][0], color='red',
                                        lw=1.0, ls='--', alpha=0.85)
        ax_newt.set_title('Newton-iteration output', fontsize=10)
        ax_newt.legend(loc='best', fontsize=6.5, framealpha=0.9)
        xlabel = newton_panel['x_name']
        if newton_panel['x_kind'] == 'phase':
            ax_newt.set_xlim(-0.02, 1.02)
        else:
            ax_newt.set_xlim(np.nanmin(newton_panel['x']), np.nanmax(newton_panel['x']))
        ax_newt.set_xlabel(xlabel)
        ax_newt.set_ylabel('Value')
        ax_newt.grid(True, alpha=0.2)
    else:
        ax_newt.text(0.5, 0.5,
                     'No Newton-iteration file found\n(pass --newton_file to force one)',
                     ha='center', va='center', transform=ax_newt.transAxes, fontsize=9)
        ax_newt.set_title('Newton-iteration output', fontsize=10)
        ax_newt.set_xlabel('')
        ax_newt.set_ylabel('')
        ax_newt.set_xticks([])
        ax_newt.set_yticks([])

    # Right: evolving SED
    sed_line, = ax_sed.plot([], [], 'k-', lw=1.0)
    ax_sed.set_xscale('log')
    ax_sed.set_xlim(wl_min, wl_max)
    ax_sed.set_ylim(0, fl_max)
    ax_sed.set_xlabel(r'Wavelength ($\mathrm{\AA}$)')
    ax_sed.set_ylabel(r'$F_\lambda$')
    ax_sed.set_title('Evolving SED', fontsize=10)
    for filt in ['B', 'V', 'R', 'I']:
        props = JOHNSON_FILTERS[filt]
        ax_sed.axvline(props['wavelength'], color=props['color'], alpha=0.35, lw=1.0, zorder=0)
    add_em_spectrum_regions(ax_sed, alpha=0.03)
    info_txt = ax_sed.text(0.97, 0.95, '', transform=ax_sed.transAxes,
                           fontsize=7, ha='right', va='top', family='monospace',
                           bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                                     alpha=0.8, edgecolor='none'))

    fig.suptitle('MESA Colors: RSP RR Lyrae cycle + SED evolution', fontsize=11)

    # Dynamic frame mapping
    model_to_sed = {mn: sed for mn, sed in cycle_seds}
    ordered_models = [int(data['model_number'][i]) for i in cycle_idx if int(data['model_number'][i]) in model_to_sed]
    if not ordered_models:
        print("Warning: no overlapping history/SED frames after ordering. Skipping movie.")
        plt.close(fig)
        return

    def init():
        for marker in history_markers.values():
            marker.set_data([], [])
        if newton_cursor is not None:
            newton_cursor.set_xdata([newton_panel['x'][0], newton_panel['x'][0]])
        sed_line.set_data([], [])
        info_txt.set_text('')
        artists = list(history_markers.values()) + [sed_line, info_txt]
        if newton_cursor is not None:
            artists.append(newton_cursor)
        return artists

    def update(frame_idx):
        model_num = ordered_models[frame_idx]
        hist_idx = model_to_hist_idx[model_num]
        cycle_pos = cycle_lookup[model_num]

        ph_now = float(phase[hist_idx])
        for band, marker in history_markers.items():
            y_now = data[mag_cols[band]][hist_idx] + band_offsets.get(band, 0.0)
            if np.isfinite(y_now):
                marker.set_data([ph_now], [y_now])
            else:
                marker.set_data([], [])

        if newton_cursor is not None:
            x_now = map_history_frame_to_newton_x(cycle_pos, data, cycle_idx, newton_panel)
            if x_now is not None:
                newton_cursor.set_xdata([x_now, x_now])

        sed = model_to_sed[model_num]
        wl = sed['wavelengths']
        fl = sed['fluxes']
        mask = (wl >= wl_min) & (wl <= wl_max)
        sed_line.set_data(wl[mask], fl[mask])

        t_now = time_days[hist_idx] if time_days is not None else np.nan
        teff = 10 ** data['log_Teff'][hist_idx] if 'log_Teff' in data.dtype.names else np.nan
        logL = data['log_L'][hist_idx] if 'log_L' in data.dtype.names else np.nan
        info_txt.set_text(
            f"phase = {ph_now:5.3f}\n"
            f"time  = {t_now:7.4f} d\n"
            f"Teff  = {teff:7.0f} K\n"
            f"log L = {logL:5.2f}\n"
            f"Model = {model_num}"
        )

        artists = list(history_markers.values()) + [sed_line, info_txt]
        if newton_cursor is not None:
            artists.append(newton_cursor)
        return artists

    n_frames = len(ordered_models)
    print(f"Rendering {n_frames} movie frames at {fps} fps...")
    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  init_func=init, blit=False,
                                  interval=1000 / fps)

    out_path = output_dir / filename
    writer = animation.FFMpegWriter(fps=fps, bitrate=2200,
                                    metadata={'title': 'RSP RR Lyrae SED evolution'})
    try:
        ani.save(str(out_path), writer=writer)
        print(f"Saved {out_path}")
    except Exception as e:
        print(f"Warning: FFMpegWriter failed ({e}). Trying PillowWriter...")
        gif_path = output_dir / filename.replace('.mp4', '.gif')
        ani.save(str(gif_path), writer=animation.PillowWriter(fps=fps))
        print(f"Saved {gif_path}")

    plt.close(fig)


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
        n_periods = int(np.nanmax(data['rsp_num_periods']))
        print(f"Pulsation cycles completed: {n_periods}")
        print(f"Points per cycle: ~{n_models // max(n_periods, 1)}")

    if 'rsp_period_in_days' in data.dtype.names:
        periods = data['rsp_period_in_days']
        valid = np.isfinite(periods) & (periods > 0)
        if valid.any():
            print(f"Mean period: {np.mean(periods[valid]):.5f} days")

    print("\nMagnitude ranges:")
    for band in ['B', 'V', 'R', 'I']:
        if band in mag_cols:
            mag = data[mag_cols[band]]
            print(f"  {band}: {np.nanmin(mag):.3f} to {np.nanmax(mag):.3f} "
                  f"(Δ = {np.nanmax(mag) - np.nanmin(mag):.3f} mag)")

    if 'B' in mag_cols and 'V' in mag_cols:
        color = data[mag_cols['B']] - data[mag_cols['V']]
        print(f"\n(B-V) color range: {np.nanmin(color):.3f} to {np.nanmax(color):.3f}")

    print("=" * 60 + "\n")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Generate MESA Colors RSP RR Lyrae plots and movie'
    )
    parser.add_argument('history_file', nargs='?', default=None,
                        help='Path to LOGS/history.data (optional positional for backward compatibility)')
    parser.add_argument('--logs_dir', type=str, default=str(DEFAULT_LOGS),
                        help='Path to the MESA LOGS directory')
    parser.add_argument('--sed_dir', type=str, default=str(DEFAULT_SED),
                        help='Path to SED CSV directory')
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_DIR),
                        help='Directory where figures and movie are saved')
    parser.add_argument('--newton_file', type=str, default=None,
                        help='Optional path to Newton-iteration output file')
    parser.add_argument('--fps', type=int, default=20,
                        help='Movie frames per second')
    parser.add_argument('--stride', type=int, default=1,
                        help='Use every Nth frame from the selected cycle in the movie')
    parser.add_argument('--preferred_cycle', type=int, default=4,
                        help='Preferred pulsation cycle number for movie selection')
    parser.add_argument('--no_movie', action='store_true',
                        help='Skip movie generation and only make the static figures')
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MESA Colors RSP RR Lyrae Analysis")
    print("=" * 60)

    history_file = Path(args.history_file) if args.history_file else logs_dir / 'history.data'
    if not history_file.exists():
        print(f"ERROR: History file not found: {history_file}")
        print("Usage: python plot_rsp.py [LOGS/history.data]")
        sys.exit(1)

    print(f"Loading {history_file}...")
    data = load_mesa_history(str(history_file))

    print(f"Loaded {len(data)} models")
    print(f"Columns: {', '.join(data.dtype.names[:10])}...")

    print_statistics(data)

    print("Generating static figures...")
    plot_lightcurve(data, output_dir=output_dir)
    plot_colorloop(data, output_dir=output_dir)
    plot_amplitude_spectrum(data, output_dir=output_dir)

    if not args.no_movie:
        print("Generating two-panel history/Newton + SED movie...")
        make_rsp_sed_movie(
            data,
            sed_dir=Path(args.sed_dir),
            output_dir=output_dir,
            newton_file=args.newton_file,
            fps=args.fps,
            stride=args.stride,
            preferred_cycle=args.preferred_cycle,
            logs_dir=logs_dir,
        )

    print(f"\nDone! Outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
