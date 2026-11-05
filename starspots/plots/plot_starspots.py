#!/usr/bin/env python3
"""
Starspots CMD Grid Plotting
============================
Plot color-magnitude diagram and HR diagram from starspots grid models.
Recreates MESA VI Figure 15: LSST g-r vs M_g CMD.

Grid: 5 masses (0.3, 0.6, 0.7, 0.9, 1.1 Msun) x 4 fspot (0.2, 0.4, 0.6, 0.8)
      = 20 models, xspot = 0.85 fixed

Usage:
    cd starspots/plots
    python plot_starspots.py
    python plot_starspots.py --data-dir /path/to/starspots
"""

import os
import re
import sys
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR  # Save plots alongside this script

# ── Publication-quality plot style ─────────────────────────────────────
# Targets ApJ/AAS journal standards: serif fonts, STIX math, inward ticks,
# 300 dpi output, no grid lines on CMDs.
plt.rcParams.update({
    'font.family':           'serif',
    'font.serif':            ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset':      'stix',       # STIX matches Times-style math without requiring LaTeX
    'font.size':             10,
    'axes.labelsize':        12,
    'axes.titlesize':        12,
    'legend.fontsize':       9,
    'legend.framealpha':     0.9,
    'legend.edgecolor':      '0.7',
    'xtick.direction':       'in',
    'ytick.direction':       'in',
    'xtick.top':             False,
    'ytick.right':           False,
    'xtick.minor.visible':   False,
    'ytick.minor.visible':   False,
    'xtick.major.size':      5,
    'ytick.major.size':      5,
    'xtick.minor.size':      3,
    'ytick.minor.size':      3,
    'figure.dpi':            150,
    'savefig.dpi':           300,           # print-quality output
    'savefig.bbox':          'tight',
    'savefig.pad_inches':    0.05,
})

# ── Grid parameters ────────────────────────────────────────────────────
MASSES = [0.3, 0.6, 0.7, 0.9, 1.1]
FSPOTS = [0.2, 0.4, 0.6, 0.8]
XSPOT  = 0.85

# Colour scheme: viridis dark-purple -> yellow, matching MESA VI Figure 15
MASS_CMAP   = plt.cm.viridis(np.linspace(0.1, 0.9, len(MASSES)))
MASS_COLORS = {m: c for m, c in zip(MASSES, MASS_CMAP)}

# Linestyles that survive print reproduction.
# ':' (dotted) is excluded — it breaks up badly at 300 dpi in print.
FSPOT_LINESTYLES = {
    0.2: '-',
    0.4: '--',
    0.6: '-.',
    0.8: (0, (5, 2)),   # long-dash, distinct from '--' and prints cleanly
}


# ── Data I/O ───────────────────────────────────────────────────────────

def read_history(filepath):
    """Read MESA history.data file (manual parser, no mesa_reader needed)."""
    filepath = str(filepath)
    with open(filepath, 'r') as f:
        for _ in range(5):
            f.readline()
        header = f.readline().split()
    data = np.genfromtxt(filepath, skip_header=6, names=header)
    return data


def discover_models(data_dir):
    """Find all LOGS_M*_f* directories and parse mass/fspot from names.

    Returns list of (mass, fspot, history_path) tuples.
    """
    log_dirs = sorted(glob.glob(os.path.join(str(data_dir), "LOGS_M*_f*")))
    models = []
    for logdir in log_dirs:
        dirname = os.path.basename(logdir)
        try:
            parts = dirname.replace("LOGS_M", "").replace("_f", " ").split()
            mass  = float(parts[0])
            fspot = float(parts[1])
        except (IndexError, ValueError):
            print(f"  Warning: could not parse {dirname}, skipping")
            continue

        history_file = os.path.join(logdir, "history.data")
        if os.path.exists(history_file):
            models.append((mass, fspot, history_file))
        else:
            print(f"  Warning: no history.data in {dirname}")

    return models


def load_model(history_path, ms_only=True):
    """Load a single model's history data.

    Parameters
    ----------
    history_path : str
        Path to history.data file.
    ms_only : bool
        If True, filter to main-sequence only (center_h1 > 0.01).

    Returns
    -------
    dict with keys: g_mag, r_mag, g_r, log_Teff, log_L (all arrays)
    Returns None if data cannot be loaded.
    """
    try:
        data = read_history(history_path)
    except Exception as e:
        print(f"  Error reading {history_path}: {e}")
        return None

    try:
        g_mag    = data['g']
        r_mag    = data['r']
        log_teff = data['log_Teff']
        log_l    = data['log_L']
    except (ValueError, KeyError) as e:
        print(f"  Missing columns in {history_path}: {e}")
        return None

    # Main-sequence filter
    if ms_only and 'center_h1' in data.dtype.names:
        mask   = data['center_h1'] > 0.01
        g_mag  = g_mag[mask]
        r_mag  = r_mag[mask]
        log_teff = log_teff[mask]
        log_l    = log_l[mask]

    return {
        'g_mag':   g_mag,
        'r_mag':   r_mag,
        'g_r':     g_mag - r_mag,
        'log_Teff': log_teff,
        'log_L':   log_l,
    }


# ── Plotting utilities ─────────────────────────────────────────────────

def _save_figure(fig, name):
    """Save figure as PDF (primary journal submission) and PNG (preview)."""
    base = OUTPUT_DIR / name
    fig.savefig(str(base) + '.pdf')
    fig.savefig(str(base) + '.png')
    print(f"  Saved: {base}.pdf / .png")
    plt.close(fig)


def _add_legends(ax, loc_mass='lower left', loc_fspot='lower left', flag = 1):
    """Add mass (colour) and fspot (linestyle) legends with proper math labels."""
        
    if flag == 1:
        mass_handles = [
            Line2D([0], [0], color=MASS_COLORS[m], linewidth=2,
                   label=rf'${m}\,{{\rm M}}_\odot$')
            for m in MASSES
        ]
        leg1 = ax.legend(handles=mass_handles, loc=loc_mass,
                         title='Mass', title_fontsize=9,
                         fontsize=9, framealpha=0.9)
        ax.add_artist(leg1)

    else:

        fspot_handles = [
            Line2D([0], [0], color='gray', linestyle=FSPOT_LINESTYLES[f],
                   linewidth=2, label=rf'$f_{{\rm spot}} = {f}$')
            for f in FSPOTS
        ]
        ax.legend(handles=fspot_handles, loc=loc_fspot,
                  title=r'$f_{\rm spot}$', title_fontsize=9,
                  fontsize=9, framealpha=0.9)


def _xspot_annotation(ax):
    """Add the xspot annotation box in the style of MESA VI Figure 15."""
    ax.text(0.65, 0.96,
            rf'$x_{{\rm spot}} = {XSPOT}$',
            transform=ax.transAxes, fontsize=9,
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='honeydew',
                      edgecolor='0.6', linewidth=0.8, alpha=0.9))


# ── Main figure: CMD (Figure 15 analogue) ─────────────────────────────

def plot_cmd(models_data):
    """Color-magnitude diagram: LSST g-r vs M_g.

    This is the CMD analogue of MESA VI Figure 15.  All 20 models are shown
    (5 masses x 4 fspot values), with colour encoding mass and linestyle
    encoding fspot at fixed xspot = 0.85.
    """
    # Single-column ApJ width: 3.5 in.  Height chosen for CMD proportions.
    fig, ax = plt.subplots(figsize=(3.5, 4.5))

    for (mass, fspot, _), model in models_data:
        if model is None:
            continue
        ax.plot(
            model['g_r'], model['g_mag'],
            color=MASS_COLORS[mass],
            linestyle=FSPOT_LINESTYLES[fspot],
            linewidth=1.5, alpha=0.9
        )

    _xspot_annotation(ax)

    ax.invert_yaxis()
    ax.set_xlabel(r'$g - r$ [mag]')
    ax.set_ylabel(r'$M_g$ [mag]')

    _add_legends(ax)
    fig.tight_layout()
    _save_figure(fig, 'starspots_cmd')


# ── Supporting figure: HR diagram ─────────────────────────────────────

def plot_hr(models_data):
    """HR diagram: log L vs log Teff (structural counterpart to the CMD)."""
    fig, ax = plt.subplots(figsize=(3.5, 4.5))

    for (mass, fspot, _), model in models_data:
        if model is None:
            continue
        ax.plot(
            10**model['log_Teff'], model['log_L'],
            color=MASS_COLORS[mass],
            linestyle=FSPOT_LINESTYLES[fspot],
            linewidth=1.5, alpha=0.9
        )

    _xspot_annotation(ax)

    ax.invert_xaxis()
    ax.set_xlabel(r'$T_{\mathrm{eff}}$ [K]')
    ax.set_ylabel(r'$\log\, L / L_\odot$')

    _add_legends(ax)
    fig.tight_layout()
    _save_figure(fig, 'starspots_hr')


# ── Supporting figure: combined CMD + HR ──────────────────────────────

def plot_cmd_hr_combined(models_data):
    """Side-by-side CMD + HR diagram."""
    fig, (ax_cmd, ax_hr) = plt.subplots(1, 2, figsize=(7.0, 4.5))

    for (mass, fspot, _), model in models_data:
        if model is None:
            continue

        style = dict(
            color=MASS_COLORS[mass],
            linestyle=FSPOT_LINESTYLES[fspot],
            linewidth=1.5, alpha=0.9
        )
        ax_cmd.plot(model['g_r'], model['g_mag'], **style)
        ax_hr.plot(10**model['log_Teff'], model['log_L'], **style)

    # CMD formatting
    ax_cmd.invert_yaxis()
    ax_cmd.set_xlabel(r'$g - r$ [mag]')
    ax_cmd.set_ylabel(r'$M_g$ [mag]')
    #_xspot_annotation(ax_cmd)
    _add_legends(ax_cmd)

    # HR formatting
    ax_hr.invert_xaxis()
    ax_hr.set_xlabel(r'$T_{\mathrm{eff}}$ [K]')
    ax_hr.set_ylabel(r'$\log\, L / L_\odot$')
    ax_hr.yaxis.set_label_position("right")
    ax_hr.yaxis.tick_right()
    _xspot_annotation(ax_hr)
    _add_legends(ax_hr, flag = 0)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.0)
    _save_figure(fig, 'starspots_cmd_hr')

# ── Supporting figure: per-mass fspot comparison ──────────────────────

def plot_fspot_comparison(models_data):
    """One panel per mass showing the effect of fspot on CMD position."""
    fig, axes = plt.subplots(1, len(MASSES),
                             figsize=(3.5 * len(MASSES), 4.5),
                             sharey=True)

    for i, target_mass in enumerate(MASSES):
        ax = axes[i]
        for (mass, fspot, _), model in models_data:
            if model is None or mass != target_mass:
                continue
            ax.plot(
                model['g_r'], model['g_mag'],
                linestyle=FSPOT_LINESTYLES[fspot],
                linewidth=2, alpha=0.9,
                label=rf'$f_{{\rm spot}} = {fspot}$'
            )

        ax.invert_yaxis()
        ax.set_xlabel(r'$g - r$ [mag]')
        if i == 0:
            ax.set_ylabel(r'$M_g$ [mag]')
        ax.set_title(rf'${target_mass}\,{{\rm M}}_\odot$')
        ax.legend(fontsize=8)

    fig.tight_layout()
    _save_figure(fig, 'starspots_fspot_comparison')


# ── SED I/O ───────────────────────────────────────────────────────────

SED_FLUX_THRESHOLD = 1e-2


def discover_sed_files(logdir, filter_name='g'):
    """Find all SED CSVs for a given filter in logdir/SED/."""
    sed_dir = os.path.join(str(logdir), 'SED')
    if not os.path.isdir(sed_dir):
        return []
    files = glob.glob(os.path.join(sed_dir, f'{filter_name}*SED_*.csv'))
    results = []
    for f in files:
        match = re.search(r'(\d{8})\.csv$', os.path.basename(f))
        if match:
            results.append((int(match.group(1)), f))
    return sorted(results)


def read_sed(filepath):
    """Read one SED CSV. Returns dict or None on failure."""
    try:
        data = np.genfromtxt(filepath, delimiter=',', names=True)
        data.dtype.names = tuple(n.strip() for n in data.dtype.names)
        return {
            'wavelengths':        data['wavelengths'],
            'fluxes':             data['fluxes'],
            'convolved_flux':     data['convolved_flux'],
            'filter_wavelengths': data['filter_wavelengths'],
            'filter_trans':       data['filter_trans'],
        }
    except Exception as e:
        print(f"  Error reading SED {filepath}: {e}")
        return None


def load_sed_timeseries(logdir, filter_name='g', max_steps=None):
    """Load all (or subsampled) SED timesteps for one model/filter."""
    files = discover_sed_files(logdir, filter_name)
    if not files:
        return []
    if max_steps is not None and len(files) > max_steps:
        idx   = np.linspace(0, len(files) - 1, max_steps, dtype=int)
        files = [files[i] for i in idx]
    result = []
    for step, path in files:
        sed = read_sed(path)
        if sed is not None:
            result.append((step, sed))
    return result


def _significant_mask(sed, threshold=SED_FLUX_THRESHOLD):
    """Boolean mask selecting wavelengths with significant flux."""
    peak = sed['fluxes'].max()
    if peak <= 0:
        return np.zeros(len(sed['fluxes']), dtype=bool)
    return sed['fluxes'] >= peak * threshold


def _valid_wav_range(sed_list, pad_frac=0.05):
    """(wmin, wmax) of the significant-flux region across all SEDs."""
    lo, hi = np.inf, -np.inf
    for sed in sed_list:
        mask = _significant_mask(sed)
        if mask.sum() < 2:
            continue
        lo = min(lo, sed['wavelengths'][mask].min())
        hi = max(hi, sed['wavelengths'][mask].max())
    if not np.isfinite(lo):
        return None
    span = hi - lo
    return lo - pad_frac * span, hi + pad_frac * span


def _integrated_convolved_flux(sed):
    """Trapezoid-integrate convolved_flux over the significant wavelength grid."""
    mask = _significant_mask(sed)
    if mask.sum() < 2:
        return np.nan
    return np.trapezoid(sed['convolved_flux'][mask], sed['wavelengths'][mask])


# ── SED plotting functions ─────────────────────────────────────────────

def plot_sed_evolution(models_data, data_dir, filter_name='g', n_snapshots=6,
                       example_mass=0.6, example_fspot=0.4):
    """SED at N evenly-spaced timesteps for one representative model."""
    logdir = None
    for (mass, fspot, history_path), _ in models_data:
        if mass == example_mass and fspot == example_fspot:
            logdir = os.path.dirname(history_path)
            break
    if logdir is None:
        for (mass, fspot, history_path), model in models_data:
            if model is not None:
                logdir = os.path.dirname(history_path)
                example_mass, example_fspot = mass, fspot
                break
    if logdir is None:
        print("  plot_sed_evolution: no valid model, skipping")
        return

    timeseries = load_sed_timeseries(logdir, filter_name, max_steps=n_snapshots)
    if not timeseries:
        print(f"  plot_sed_evolution: no SED files in {logdir}/SED/, skipping")
        return

    fig, ax  = plt.subplots(figsize=(7.0, 4.0))
    cmap_col = plt.cm.plasma(np.linspace(0.1, 0.9, len(timeseries)))
    sed_list = [sed for _, sed in timeseries]

    for i, (step, sed) in enumerate(timeseries):
        mask = _significant_mask(sed)
        if mask.sum() < 2:
            continue
        ax.plot(sed['wavelengths'][mask], sed['fluxes'][mask],
                color=cmap_col[i], linewidth=1.2, alpha=0.85,
                label=f'step {step}')

    xlim = _valid_wav_range(sed_list)
    if xlim is not None:
        ax.set_xlim(xlim)

    ax2 = ax.twinx()
    last_sed = sed_list[-1]
    fw, ft = last_sed['filter_wavelengths'], last_sed['filter_trans']
    if xlim is not None:
        fmask = (fw >= xlim[0]) & (fw <= xlim[1])
        fw, ft = fw[fmask], ft[fmask]
    if len(fw) > 1:
        ax2.fill_between(fw, ft, alpha=0.15, color='steelblue')
    ax2.set_ylabel('Filter transmission', color='steelblue', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='steelblue')
    ax2.set_ylim(bottom=0)
    if xlim is not None:
        ax.set_xlim(xlim)

    ax.set_xlabel(r'Wavelength [$\AA$]')
    ax.set_ylabel(r'Flux [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]')
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    _save_figure(fig, f'starspots_sed_evolution_{filter_name}')


def plot_sed_fspot_comparison(models_data, filter_name='g'):
    """At the final timestep, compare SEDs across fspot for each mass."""
    fig, axes = plt.subplots(1, len(MASSES),
                             figsize=(3.0 * len(MASSES), 4.0),
                             sharey=False)
    fspot_cmap   = plt.cm.YlOrRd(np.linspace(0.25, 0.9, len(FSPOTS)))
    fspot_colors = {f: c for f, c in zip(FSPOTS, fspot_cmap)}

    for i, target_mass in enumerate(MASSES):
        ax        = axes[i]
        panel_seds = []
        last_sed  = None

        for (mass, fspot, history_path), model in models_data:
            if mass != target_mass or model is None:
                continue
            files = discover_sed_files(os.path.dirname(history_path), filter_name)
            if not files:
                continue
            sed = read_sed(files[-1][1])
            if sed is None:
                continue
            mask = _significant_mask(sed)
            if mask.sum() < 2:
                continue
            ax.plot(sed['wavelengths'][mask], sed['fluxes'][mask],
                    color=fspot_colors[fspot], linewidth=1.5, alpha=0.85,
                    label=rf'$f_{{\rm spot}}={fspot}$')
            panel_seds.append(sed)
            last_sed = sed

        xlim = _valid_wav_range(panel_seds)
        if xlim is not None:
            ax.set_xlim(xlim)

        if last_sed is not None:
            ax2 = ax.twinx()
            fw, ft = last_sed['filter_wavelengths'], last_sed['filter_trans']
            if xlim is not None:
                fmask = (fw >= xlim[0]) & (fw <= xlim[1])
                fw, ft = fw[fmask], ft[fmask]
            if len(fw) > 1:
                ax2.fill_between(fw, ft, alpha=0.12, color='steelblue')
            ax2.set_ylim(bottom=0)
            ax2.set_yticklabels([])
            if xlim is not None:
                ax.set_xlim(xlim)

        ax.set_xlabel(r'Wavelength [$\AA$]')
        if i == 0:
            ax.set_ylabel(r'Flux [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]')
        ax.set_title(rf'${target_mass}\,{{\rm M}}_\odot$')
        ax.legend(fontsize=7)

    fig.tight_layout()
    _save_figure(fig, f'starspots_sed_fspot_comparison_{filter_name}')


def plot_sed_flux_timeseries(models_data, filter_name='g'):
    """Integrated convolved flux vs timestep for all models."""
    fig, axes = plt.subplots(1, len(MASSES),
                             figsize=(3.0 * len(MASSES), 4.0),
                             sharey=False)
    fspot_cmap   = plt.cm.YlOrRd(np.linspace(0.25, 0.9, len(FSPOTS)))
    fspot_colors = {f: c for f, c in zip(FSPOTS, fspot_cmap)}

    for i, target_mass in enumerate(MASSES):
        ax       = axes[i]
        any_data = False
        for (mass, fspot, history_path), model in models_data:
            if mass != target_mass or model is None:
                continue
            timeseries = load_sed_timeseries(
                os.path.dirname(history_path), filter_name)
            if not timeseries:
                continue
            steps  = np.array([s for s, _ in timeseries])
            fluxes = np.array([_integrated_convolved_flux(sed)
                               for _, sed in timeseries])
            valid  = np.isfinite(fluxes) & (fluxes > 0) & (steps < 40000)
            if valid.sum() < 2:
                continue
            ax.plot(steps[valid], fluxes[valid],
                    color=fspot_colors[fspot], linewidth=1.5, alpha=0.85,
                    label=rf'$f_{{\rm spot}}={fspot}$')
            any_data = True

        ax.set_xlabel('Timestep')
        if i == 0:
            ax.set_ylabel(
                r'$\int F_\lambda \cdot T_\lambda \, \mathrm{d}\lambda$'
                r' [erg s$^{-1}$ cm$^{-2}$]')
        ax.set_title(rf'${target_mass}\,{{\rm M}}_\odot$')
        if any_data:
            ax.legend(fontsize=7)

    fig.tight_layout()
    _save_figure(fig, f'starspots_sed_flux_timeseries_{filter_name}')


# ── Summary statistics ─────────────────────────────────────────────────

def print_summary(models_data):
    """Print summary statistics for all loaded models."""
    print("\n" + "=" * 60)
    print("STARSPOTS GRID SUMMARY")
    print("=" * 60)

    loaded = [(info, m) for info, m in models_data if m is not None]
    print(f"Models loaded: {len(loaded)} / {len(models_data)}")

    if not loaded:
        return

    print(f"\n{'Mass':>6s} {'fspot':>6s} {'g-r range':>14s} "
          f"{'Mg range':>14s} {'N_pts':>6s}")
    print("-" * 52)

    for (mass, fspot, _), model in loaded:
        gr = model['g_r']
        mg = model['g_mag']
        print(f"{mass:6.1f} {fspot:6.1f} "
              f"[{gr.min():+6.3f}, {gr.max():+6.3f}] "
              f"[{mg.min():+6.2f}, {mg.max():+6.2f}] "
              f"{len(gr):6d}")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot starspots CMD grid results')
    parser.add_argument('--data-dir', type=str, default=str(PARENT_DIR),
                        help='Directory containing LOGS_M*_f* folders')
    parser.add_argument('--no-ms-filter', action='store_true',
                        help='Include post-MS evolution (default: MS only)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Discover and load models
    model_info = discover_models(data_dir)
    if not model_info:
        print("ERROR: No LOGS_M*_f* directories found.")
        print(f"  Looked in: {data_dir}")
        sys.exit(1)

    print(f"Found {len(model_info)} model directories")

    ms_only    = not args.no_ms_filter
    models_data = []
    for mass, fspot, path in model_info:
        model = load_model(path, ms_only=ms_only)
        models_data.append(((mass, fspot, path), model))
        if model is not None:
            print(f"  Loaded M={mass}, fspot={fspot} ({len(model['g_r'])} points)")

    # Generate all plots
    print("\nGenerating plots...")
    plot_cmd(models_data)
    plot_hr(models_data)
    plot_cmd_hr_combined(models_data)
    plot_fspot_comparison(models_data)

    # SED-based plots (require sed_per_model = .true. in inlist)
    plot_sed_evolution(models_data, data_dir, filter_name='g')
    plot_sed_fspot_comparison(models_data, filter_name='g')
    plot_sed_flux_timeseries(models_data, filter_name='g')

    # Print summary
    print_summary(models_data)

    print("\nDone!")


if __name__ == "__main__":
    main()