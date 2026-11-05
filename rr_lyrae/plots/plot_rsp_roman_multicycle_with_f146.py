#!/usr/bin/env python3
"""
plot_rsp_roman.py
=================
Consolidated plotting for MESA Colors RSP RR Lyrae, Roman WFI edition.

Incorporates all functionality from plot_rsp.py, plot_rsp_enhanced.py,
and plot_rsp_radius_temp.py, updated for Roman WFI filters.

Figures produced:

  Standard:
    fig_lightcurves.pdf          Phased light curves, representative Roman bands stacked
    fig_amplitudes.pdf           Amplitude vs wavelength + Planck theory
    fig_phase_offset.pdf         Phase of max light vs wavelength
    fig_colorloop.pdf            (F062-F184) vs F106 colour-magnitude loop

  Physics diagnostics:
    fig_radius_sed.pdf           ** NEW ** 2x2:
                                   left  = R vs (F062-F184) colour, phase-coloured
                                   right = SED at max-R / min-R (top) +
                                           delta SED residual (bottom, shared x-axis)
    fig_teff_sed.pdf             ** NEW ** 2x2:
                                   left  = Teff vs (F062-F184) colour, phase-coloured
                                   right = SED at max-T / min-T (top) +
                                           delta SED residual (bottom, shared x-axis)
    fig_radius_diagnostic.pdf    Normalised flux vs 2*delta_lnR (phase + scatter)
    fig_temp_diagnostic.pdf      Normalised flux vs delta_lnT (phase + scatter)

  Optional (subfolders):
    cmd/                         All colour-magnitude combinations (--cmd)
    cc/                          All colour-colour combinations    (--cmd)

Usage:
    python plot_rsp_roman.py
    python plot_rsp_roman.py /path/to/LOGS/history.data
    python plot_rsp_roman.py --logs_dir /path --output_dir /path/plots
    python plot_rsp_roman.py --cmd        # also generate all CMD/CC plots
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'axes.labelsize': 12, 'axes.titlesize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 9, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

SCRIPT_DIR   = Path(__file__).resolve().parent
PARENT_DIR   = SCRIPT_DIR.parent
DEFAULT_LOGS = PARENT_DIR / 'LOGS_colors'
DEFAULT_OUT  = SCRIPT_DIR

# ---------------------------------------------------------------------------
# Roman WFI filters
# ---------------------------------------------------------------------------
ROMAN_FILTERS = {
    'F062': {'lambda_eff': 6200,  'lambda_lo': 4800,  'lambda_hi': 7600,  'color': '#CC2200'},
    'F087': {'lambda_eff': 8700,  'lambda_lo': 7600,  'lambda_hi': 10000, 'color': '#FF6600'},
    'F106': {'lambda_eff': 10600, 'lambda_lo': 9300,  'lambda_hi': 11900, 'color': '#FFAA00'},
    'F129': {'lambda_eff': 12900, 'lambda_lo': 11500, 'lambda_hi': 14500, 'color': '#99CC00'},
    'F146': {'lambda_eff': 14600, 'lambda_lo': 9270,  'lambda_hi': 20000, 'color': '#444444'},
    'F158': {'lambda_eff': 15800, 'lambda_lo': 13800, 'lambda_hi': 17700, 'color': '#33BB55'},
    'F184': {'lambda_eff': 18400, 'lambda_lo': 16800, 'lambda_hi': 20000, 'color': '#1166AA'},
    'F213': {'lambda_eff': 21300, 'lambda_lo': 19500, 'lambda_hi': 23000, 'color': '#660099'},
}
ROMAN_ORDER = ['F062', 'F087', 'F106', 'F129', 'F146', 'F158', 'F184', 'F213']

H   = 6.626e-27   # erg s
C   = 2.998e10    # cm/s
KB  = 1.381e-16   # erg/K
AA  = 1e-8        # cm per Angstrom


# ===========================================================================
# Physics helpers
# ===========================================================================

def planck_flam(wl_AA, T_K):
    """Planck B_lambda in arbitrary units. wl_AA array, T_K scalar."""
    lam = wl_AA * AA
    x   = np.clip(H * C / (lam * KB * T_K), 0.0, 500.0)
    return lam**(-5) / (np.exp(x) - 1.0)


def filter_mean(wl, flux, lo, hi):
    """Mean flux integrated over tophat [lo, hi] Angstroms."""
    m = (wl >= lo) & (wl <= hi)
    if m.sum() < 2:
        return np.nan
    return np.trapezoid(flux[m], wl[m]) / (hi - lo)


# ===========================================================================
# Data loading
# ===========================================================================

def load_mesa_history(filepath):
    filepath = Path(filepath)
    lines = open(filepath, encoding='utf-8').readlines()
    header_idx = None
    for i, line in enumerate(lines):
        if 'model_number' in line.split():
            header_idx = i
            break
    if header_idx is None:
        for c in (3, 5, 1):
            if c < len(lines):
                toks = lines[c].split()
                if sum(1 for t in toks if any(ch.isalpha() for ch in t)) >= 3:
                    header_idx = c
                    break
    if header_idx is None:
        raise ValueError(f'No column header found in {filepath}')
    cols = lines[header_idx].split()
    data = np.genfromtxt(filepath, skip_header=header_idx+1, names=cols,
                         invalid_raise=False, autostrip=True)
    if getattr(data, 'shape', ()) == ():
        data = np.array([data], dtype=data.dtype)
    return data


def find_roman_columns(data):
    colnames = list(data.dtype.names)
    print('\n--- All columns in history.data ---')
    for i in range(0, len(colnames), 8):
        print('  ' + '  '.join(colnames[i:i+8]))
    print()
    cs = set(colnames)
    found = {}
    for filt in ROMAN_ORDER:
        fl = filt.lower()
        matched = None
        for cand in [filt, 'Roman_'+filt, 'WFI_'+filt, 'Roman_WFI_'+filt,
                     fl, 'roman_'+fl, 'wfi_'+fl]:
            if cand in cs:
                matched = cand; break
        if matched is None:
            for col in colnames:
                if fl in col.lower():
                    matched = col; break
        if matched is not None:
            found[filt] = matched
    if found:
        print('Roman WFI columns found:')
        for k, v in found.items():
            print(f'  {k} -> {v}')
    else:
        print('WARNING: No Roman WFI columns found.')
    print()
    return found


# ===========================================================================
# Utility helpers
# ===========================================================================

def get_time_days(data):
    for col in ('star_age_day', 'time_days', 'star_age'):
        if col in data.dtype.names:
            return np.asarray(data[col], dtype=float)
    return None


def get_phase(data):
    if 'rsp_phase' in data.dtype.names:
        return np.asarray(data['rsp_phase'], dtype=float)
    t = get_time_days(data)
    if t is None:
        return np.linspace(0., 1., len(data), endpoint=False)
    period = 0.71
    if 'rsp_period_in_days' in data.dtype.names:
        p = np.asarray(data['rsp_period_in_days'], dtype=float)
        v = np.isfinite(p) & (p > 0)
        if v.any():
            period = float(np.nanmedian(p[v]))
    return ((t - t[0]) / period) % 1.0


def get_settled_mask(data, min_periods=4.0, single_cycle=True):
    if 'rsp_num_periods' in data.dtype.names:
        periods = np.asarray(data['rsp_num_periods'], dtype=float)
        finite  = np.isfinite(periods)
        if finite.any():
            if single_cycle:
                max_p = np.nanmax(periods[finite])
                cf    = np.floor(max_p)
                cand  = (periods >= max(min_periods, cf - 1.)) & (periods < cf)
                if cand.sum() >= 50: return cand
                cand  = (periods >= min_periods) & (periods < min_periods + 1.)
                if cand.sum() >= 50: return cand
            cand = periods >= min_periods
            if cand.sum() >= 50: return cand
    t = get_time_days(data)
    if t is not None:
        t    = np.asarray(t, dtype=float)
        cand = (t > 3.) & (t < 4.) if single_cycle else (t > 3.)
        if cand.sum() >= 50: return cand
    return np.ones(len(data), dtype=bool)


def finite_mask(*arrays):
    m = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        m &= np.isfinite(a)
    return m


def get_teff(data):
    if 'effective_T' in data.dtype.names:
        return np.asarray(data['effective_T'], dtype=float)
    if 'log_Teff' in data.dtype.names:
        return 10. ** np.asarray(data['log_Teff'], dtype=float)
    return None


def normalised(x):
    x  = np.asarray(x, dtype=float) - np.nanmean(x)
    pk = np.nanmax(np.abs(x))
    return x / pk if (np.isfinite(pk) and pk > 0) else np.zeros_like(x)


def dln_flux(mag):
    mag = np.asarray(mag, dtype=float)
    return -0.4 * np.log(10.) * (mag - np.nanmean(mag))


def pearson_r(x, y):
    xm = x - np.nanmean(x); ym = y - np.nanmean(y)
    d  = np.sqrt(np.nansum(xm**2) * np.nansum(ym**2))
    return np.nansum(xm * ym) / d if (np.isfinite(d) and d > 0) else np.nan


def add_regression(ax, x, y, color='k', ls='--', lw=1.2):
    coeffs = np.polyfit(x, y, 1)
    xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    ax.plot(xx, np.polyval(coeffs, xx), color=color, ls=ls, lw=lw, alpha=0.7)


def add_arrow(ax, ph, x, y, phi=0.25, dphi=0.05):
    i1 = np.argmin(np.abs(ph - phi))
    i2 = np.argmin(np.abs(ph - (phi + dphi)))
    if i1 != i2:
        ax.annotate('', xy=(x[i2], y[i2]), xytext=(x[i1], y[i1]),
                    arrowprops=dict(arrowstyle='->', color='k', lw=1.5))


def shade_roman(ax, alpha_span=0.07, alpha_line=0.28, label_ax=None):
    """Add Roman bandpass shading and dashed centre lines to ax."""
    for filt in ROMAN_ORDER:
        p = ROMAN_FILTERS[filt]
        ax.axvspan(p['lambda_lo'], p['lambda_hi'],
                   alpha=alpha_span, color=p['color'], lw=0, zorder=0)
        ax.axvline(p['lambda_eff'], color=p['color'],
                   alpha=alpha_line, lw=0.8, ls='--', zorder=1)
    if label_ax is not None:
        for filt in ROMAN_ORDER:
            p = ROMAN_FILTERS[filt]
            label_ax.text(p['lambda_eff'], 1.02, filt,
                          transform=label_ax.get_xaxis_transform(),
                          ha='center', va='bottom', fontsize=7,
                          color=p['color'], rotation=40)


def save(fig, output_dir, stem):
    for ext in ('pdf', 'png'):
        fig.savefig(output_dir / f'{stem}.{ext}',
                    dpi=(300 if ext == 'pdf' else 150), bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {stem}.pdf')


# ===========================================================================
# ── THE TWO NEW KEY FIGURES ──────────────────────────────────────────────
# ===========================================================================

def _sed_panel_pair(fig, gs_row, gs_col, snap_a, snap_b, label_a, label_b,
                    color_a, color_b, wl, title_top, ylabel_left_top,
                    add_filter_labels=True):
    """
    Build the right-hand two-panel column (SED top + residual bottom).

    snap_a, snap_b: dicts with keys 'T' (K) and 'R' (Rsun)
    Returns (ax_sed, ax_delta).
    """
    ax_sed   = fig.add_subplot(gs_row[0, gs_col])
    ax_delta = fig.add_subplot(gs_row[1, gs_col], sharex=ax_sed)

    # Planck SEDs
    F_a = planck_flam(wl, snap_a['T']) * snap_a['R']**2
    F_b = planck_flam(wl, snap_b['T']) * snap_b['R']**2
    norm = np.nanmax(F_a)
    F_a_n = F_a / norm
    F_b_n = F_b / norm

    shade_roman(ax_sed,   label_ax=(ax_sed if add_filter_labels else None))
    shade_roman(ax_delta)

    ax_sed.plot(wl, F_a_n, color=color_a, lw=2.2,
                label=(r'$T_{\rm eff}$=' + f'{snap_a["T"]:.0f} K, '
                       r'$R$=' + f'{snap_a["R"]:.4f}' + r' $R_\odot$  (' + label_a + ')'))
    ax_sed.plot(wl, F_b_n, color=color_b, lw=2.2, ls='--',
                label=(r'$T_{\rm eff}$=' + f'{snap_b["T"]:.0f} K, '
                       r'$R$=' + f'{snap_b["R"]:.4f}' + r' $R_\odot$  (' + label_b + ')'))
    ax_sed.fill_between(wl, F_b_n, F_a_n, alpha=0.10, color='gray')
    ax_sed.set_ylabel(ylabel_left_top)
    ax_sed.set_ylim(-0.02, 1.18)
    ax_sed.legend(loc='upper right', framealpha=0.92, fontsize=8)
    ax_sed.grid(True, alpha=0.10, lw=0.5)
    #ax_sed.set_title(title_top, fontsize=10)
    plt.setp(ax_sed.get_xticklabels(), visible=False)

    # Residual = F_a - F_b (same normalisation as above)
    delta = (F_a - F_b) / norm
    ax_delta.axhline(0, color='k', lw=0.7, alpha=0.4)
    ax_delta.plot(wl, delta, color='k', lw=1.8)
    ax_delta.fill_between(wl, 0, delta, where=(delta > 0),
                          alpha=0.18, color=color_a, label=f'{label_a} brighter')
    ax_delta.fill_between(wl, 0, delta, where=(delta < 0),
                          alpha=0.18, color=color_b, label=f'{label_b} brighter')

    # Filter-integrated residuals as coloured circles
    for filt in ROMAN_ORDER:
        p   = ROMAN_FILTERS[filt]
        val = filter_mean(wl, delta, p['lambda_lo'], p['lambda_hi'])
        if np.isfinite(val):
            ax_delta.scatter(p['lambda_eff'], val, color=p['color'],
                             s=70, zorder=6, edgecolors='k', lw=0.7)

    ax_delta.set_xlabel(r'Wavelength ($\mathrm{\AA}$)')
    ax_delta.set_ylabel(r'$\Delta F_\lambda$ (normalised)')
    ax_delta.legend(loc='upper right', framealpha=0.92, fontsize=8)
    ax_delta.grid(True, alpha=0.10, lw=0.5)

    return ax_sed, ax_delta



# ===========================================================================
# SED loading from MESA Colors output files
# ===========================================================================

def load_sed_for_model(sed_dir, model_number, preferred_filter='F062'):
    """
    Load the real stellar atmosphere SED for a given model number.

    Files are named {FILTER}_SED_{MODEL:08d}.csv and contain columns:
        wavelengths   -- wavelength in Angstroms
        fluxes        -- surface flux density (erg/s/cm^2/AA)
        convolved_flux -- flux after multiplying by filter transmission

    We use 'fluxes' (the raw atmospheric SED before filter convolution)
    since it is the same underlying spectrum regardless of which filter
    file we open.  We try preferred_filter first, then any available filter.

    Returns dict with 'wavelengths' and 'fluxes' arrays, or None on failure.
    """
    import csv as _csv
    sed_dir = Path(sed_dir)
    model_str = f'{int(model_number):08d}'

    # Try preferred filter first, then any filter
    candidates = [f'{preferred_filter}_SED_{model_str}.csv']
    for filt in ROMAN_ORDER:
        fname = f'{filt}_SED_{model_str}.csv'
        if fname not in candidates:
            candidates.append(fname)

    for fname in candidates:
        fpath = sed_dir / fname
        if not fpath.exists():
            continue
        wls = []; fls = []
        try:
            with open(fpath, newline='') as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    try:
                        wl = float(row['wavelengths'])
                        fl = float(row['fluxes'])
                    except (ValueError, KeyError):
                        continue
                    if wl == 0:
                        continue
                    wls.append(wl); fls.append(fl)
        except Exception as e:
            print(f'  Warning: could not read {fpath}: {e}')
            continue
        if wls:
            wls = np.array(wls); fls = np.array(fls)
            # Sort by wavelength (should already be sorted but be safe)
            srt = np.argsort(wls)
            return {'wavelengths': wls[srt], 'fluxes': fls[srt]}

    print(f'  Warning: no SED file found for model {model_number} in {sed_dir}')
    return None


# ===========================================================================
# THE TWO KEY FIGURES  (use real MESA Colors SEDs, not blackbodies)
# ===========================================================================
 
# ===========================================================================
# THE TWO KEY FIGURES  (use real MESA Colors SEDs, not blackbodies)
# ===========================================================================
 
def _right_sed_panels(fig, inner_gs, sed_a, sed_b, label_a, label_b,
                      color_a, color_b, title_top):
    """
    Right-hand two-panel column: SED comparison (top) + residual (bottom).
 
    Panels share a log-scaled x-axis clipped to the Roman WFI wavelength
    range (4500–24000 AA).  hspace is controlled by the caller's GridSpec
    (pass hspace=0 there); tick labels on the top panel are suppressed so
    the two panels appear butted together.
    """
    # Roman wavelength window: just outside F062 blue edge to F213 red edge
    WL_LO = 4500.0   # AA
    WL_HI = 24000.0  # AA
 
    ax_sed   = fig.add_subplot(inner_gs[0])
    ax_delta = fig.add_subplot(inner_gs[1], sharex=ax_sed)
 
    wl_a = sed_a['wavelengths'];  fl_a = sed_a['fluxes']
    wl_b = sed_b['wavelengths'];  fl_b = sed_b['fluxes']
 
    # Clip both SEDs to the Roman window before doing anything else
    clip_a = (wl_a >= WL_LO) & (wl_a <= WL_HI) & np.isfinite(fl_a) & (fl_a > 0)
    wl_a = wl_a[clip_a];  fl_a = fl_a[clip_a]
 
    clip_b = (wl_b >= WL_LO) & (wl_b <= WL_HI) & np.isfinite(fl_b) & (fl_b > 0)
    wl_b = wl_b[clip_b];  fl_b = fl_b[clip_b]
 
    if len(wl_a) < 2 or len(wl_b) < 2:
        print('  _right_sed_panels: too few points after clipping — skipping.')
        return ax_sed, ax_delta
 
    # Interpolate sed_b onto sed_a grid for the residual
    fl_b_interp = np.interp(wl_a, wl_b, fl_b, left=np.nan, right=np.nan)
 
    # Normalise to peak of sed_a in this window
    norm   = np.nanmax(fl_a)
    fl_a_n = fl_a / norm
    fl_b_n = fl_b_interp / norm
    delta  = fl_a_n - fl_b_n
 
    # ── Roman bandpass shading on both panels ───────────────────────────
    for filt in ROMAN_ORDER:
        p = ROMAN_FILTERS[filt]
        if p['lambda_hi'] < WL_LO or p['lambda_lo'] > WL_HI:
            continue
        for ax in (ax_sed, ax_delta):
            ax.axvspan(p['lambda_lo'], p['lambda_hi'],
                       alpha=0.07, color=p['color'], lw=0, zorder=0)
            ax.axvline(p['lambda_eff'], color=p['color'],
                       alpha=0.28, lw=0.8, ls='--', zorder=1)
        # Filter labels inside the top panel (at ~85% height so they don't clip)
        ax_sed.text(p['lambda_eff'], 0.93, filt,
                    transform=ax_sed.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=8,
                    color=p['color'], fontweight='bold')
 
    # ── Top panel: SED comparison ────────────────────────────────────────
    ax_sed.fill_between(wl_a, fl_b_n, fl_a_n, alpha=0.12, color='gray', zorder=2)
    ax_sed.plot(wl_a, fl_a_n, color=color_a, lw=2.2, zorder=3, label=label_a)
    ax_sed.plot(wl_a, fl_b_n, color=color_b, lw=2.2, ls='--', zorder=3, label=label_b)
 
    ax_sed.set_xscale('log')
    ax_sed.set_xlim(WL_LO, WL_HI)
    ax_sed.set_ylim(bottom=0.0)
    ax_sed.set_ylabel(r'$F_\lambda$ (normalised)')
    ax_sed.legend(loc='center right', framealpha=0.92, fontsize=8)
    ax_sed.grid(True, which='both', alpha=0.12, lw=0.5)
    #ax_sed.set_title(title_top, fontsize=10)
    # Hide x tick labels — panels are butted together; bottom panel carries them
    plt.setp(ax_sed.get_xticklabels(), visible=False)
    ax_sed.tick_params(axis='x', which='both', length=0)
 
    # ── Bottom panel: residual ───────────────────────────────────────────
    ax_delta.axhline(0, color='k', lw=0.8, alpha=0.5)
    ax_delta.plot(wl_a, delta, color='k', lw=1.8, zorder=3)
    ax_delta.fill_between(wl_a, 0, delta, where=(delta > 0),
                          alpha=0.20, color=color_a, zorder=2)
    ax_delta.fill_between(wl_a, 0, delta,
                          where=(delta < 0) & np.isfinite(delta),
                          alpha=0.20, color=color_b, zorder=2)
 
    # Filter-integrated residuals — circles at effective wavelength
    for filt in ROMAN_ORDER:
        p = ROMAN_FILTERS[filt]
        m = ((wl_a >= p['lambda_lo']) & (wl_a <= p['lambda_hi'])
             & np.isfinite(delta))
        if m.sum() < 2:
            continue
        val = np.trapezoid(delta[m], wl_a[m]) / (p['lambda_hi'] - p['lambda_lo'])
        if not np.isfinite(val):
            continue
        ax_delta.scatter(p['lambda_eff'], val, color=p['color'],
                         s=80, zorder=6, edgecolors='k', lw=0.8)
        # Label above or below the circle depending on sign
        offset = 9 if val >= 0 else -14
        ax_delta.annotate(filt, (p['lambda_eff'], val),
                          xytext=(0, offset), textcoords='offset points',
                          ha='center', fontsize=7, color=p['color'])
 
    ax_delta.set_xscale('log')
    ax_delta.set_xlim(WL_LO, WL_HI)
 
    # Human-readable x tick labels in Angstroms (log-spaced)
    import matplotlib.ticker as mticker
    ticks = [5000, 6000, 7000, 10000, 15000, 20000]
    ax_delta.set_xticks(ticks)
    ax_delta.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{int(x):,}'))
    #ax_delta.set_xlabel(r'Wavelength ($\mathrm{\AA}$)')
    ax_delta.set_ylabel(r'$\Delta F_\lambda$ (normalised)')
    ax_delta.grid(True, which='both', alpha=0.12, lw=0.5)
    ax_delta.set_xlabel(r'Wavelength $\lambda$')

    return ax_sed, ax_delta
 
 
 
def plot_radius_sed(data, output_dir, sed_dir, min_periods=4.0):
    """
    2-column figure.
 
    Left: photospheric radius vs (F062-F184) colour, phase-coloured.
          Stars mark the max-R and min-R moments.
 
    Right (two stacked panels, shared x-axis):
      Top:    Real MESA Colors SED at max-R moment (solid) +
              real SED at min-R moment (dashed).
      Bottom: Residual delta SED = F_maxR - F_minR, with Roman filter
              integrated values as coloured circles.
    """
    if 'photosphere_r' not in data.dtype.names:
        print('fig_radius_sed: no photosphere_r — skipping.'); return
 
    mag_cols = find_roman_columns(data)
    for need in ('F062', 'F184'):
        if need not in mag_cols:
            print(f'fig_radius_sed: missing {need} — skipping.'); return
 
    mask = get_settled_mask(data, min_periods=min_periods, single_cycle=False)
    if mask.sum() < 50:
        mask = np.ones(len(data), dtype=bool)
 
    radius   = np.asarray(data['photosphere_r'], dtype=float)[mask]
    f062     = np.asarray(data[mag_cols['F062']], dtype=float)[mask]
    f184     = np.asarray(data[mag_cols['F184']], dtype=float)[mask]
    phase    = get_phase(data)[mask]
    mod_nums = np.asarray(data['model_number'], dtype=int)[mask]
    color    = f062 - f184
 
    ok = finite_mask(radius, color, phase)
    radius   = radius[ok];  color    = color[ok]
    phase    = phase[ok];   mod_nums = mod_nums[ok]
 
    idx_maxR = np.argmax(radius)
    idx_minR = np.argmin(radius)
    mn_maxR  = mod_nums[idx_maxR]
    mn_minR  = mod_nums[idx_minR]
 
    print(f'  Max R: model={mn_maxR}, R={radius[idx_maxR]:.4f} Rsun, '
          f'phase={phase[idx_maxR]:.3f}')
    print(f'  Min R: model={mn_minR}, R={radius[idx_minR]:.4f} Rsun, '
          f'phase={phase[idx_minR]:.3f}')
 
    sed_maxR = load_sed_for_model(sed_dir, mn_maxR)
    sed_minR = load_sed_for_model(sed_dir, mn_minR)
    if sed_maxR is None or sed_minR is None:
        print('  fig_radius_sed: could not load SEDs — skipping.')
        return
 
    teff_arr = get_teff(data)
    if teff_arr is not None:
        T_maxR = teff_arr[mask][ok][idx_maxR]
        T_minR = teff_arr[mask][ok][idx_minR]
        label_maxR = (r'Max $R$: '
                      r'$T_{\rm eff}$=' + f'{T_maxR:.0f} K, '
                      r'$R$=' + f'{radius[idx_maxR]:.4f}' + r' $R_\odot$')
        label_minR = (r'Min $R$: '
                      r'$T_{\rm eff}$=' + f'{T_minR:.0f} K, '
                      r'$R$=' + f'{radius[idx_minR]:.4f}' + r' $R_\odot$')
    else:
        label_maxR = f'Max R = {radius[idx_maxR]:.4f} Rsun'
        label_minR = f'Min R = {radius[idx_minR]:.4f} Rsun'
 
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    fig    = plt.figure(figsize=(13, 6.5))
    outer  = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.5], wspace=0.38)
    ax_left = fig.add_subplot(outer[0, 0])
    inner_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 1], hspace=0.0)
 
    # Left panel
    sc = ax_left.scatter(color, radius, c=phase, cmap='twilight_shifted',
                         s=8, alpha=0.8, edgecolors='none', rasterized=True)
    plt.colorbar(sc, ax=ax_left, label='Pulsation Phase', shrink=1.0, pad=0.0)
    add_arrow(ax_left, phase, color, radius, phi=0.25)
    ax_left.scatter(color[idx_maxR], radius[idx_maxR],
                    marker='*', s=200, color='#CC2200', zorder=10,
                    label=f'Max R')# (model {mn_maxR})')
    ax_left.scatter(color[idx_minR], radius[idx_minR],
                    marker='s', s=100, color='#1166AA', zorder=10,
                    label=f'Min R')# (model {mn_minR})')
    ax_left.set_xlabel(r'$F_{062} - F_{184}$ (mag)')
    ax_left.set_ylabel(r'Photospheric Radius ($R_\odot$)')
    #ax_left.set_title('Radius vs Roman colour', fontsize=10)
    ax_left.legend(framealpha=0.9, fontsize=9)
    ax_left.grid(True, alpha=0.18, lw=0.5)
 
    # Right panels — real SEDs
    _right_sed_panels(fig, inner_gs,
                      sed_maxR, sed_minR,
                      label_maxR, label_minR,
                      '#CC2200', '#1166AA',
                      'Real atmosphere SED at max/min radius')
 
    #fig.suptitle('Radius variation: real MESA Colors SEDs at max/min R',
    #             fontsize=12, y=1.01)
    save(fig, output_dir, 'fig_radius_sed')
 
 
def plot_teff_sed(data, output_dir, sed_dir, min_periods=4.0):
    """
    2-column figure.
 
    Left: Teff vs (F062-F184) colour, phase-coloured.
          Marks at hottest and coldest moments.
 
    Right (two stacked panels, shared x-axis):
      Top:    Real MESA Colors SED at hottest moment (solid) +
              real SED at coldest moment (dashed).
      Bottom: Residual delta SED = F_hot - F_cold, with Roman filter
              integrated values as coloured circles.
    """
    teff_arr = get_teff(data)
    if teff_arr is None:
        print('fig_teff_sed: no Teff column — skipping.'); return
    if 'photosphere_r' not in data.dtype.names:
        print('fig_teff_sed: no photosphere_r — skipping.'); return
 
    mag_cols = find_roman_columns(data)
    for need in ('F062', 'F184'):
        if need not in mag_cols:
            print(f'fig_teff_sed: missing {need} — skipping.'); return
 
    mask = get_settled_mask(data, min_periods=min_periods, single_cycle=False)
    if mask.sum() < 50:
        mask = np.ones(len(data), dtype=bool)
 
    teff     = teff_arr[mask]
    radius   = np.asarray(data['photosphere_r'], dtype=float)[mask]
    f062     = np.asarray(data[mag_cols['F062']], dtype=float)[mask]
    f184     = np.asarray(data[mag_cols['F184']], dtype=float)[mask]
    phase    = get_phase(data)[mask]
    mod_nums = np.asarray(data['model_number'], dtype=int)[mask]
    color    = f062 - f184
 
    ok = finite_mask(teff, radius, color, phase)
    teff     = teff[ok];    radius   = radius[ok]
    color    = color[ok];   phase    = phase[ok]
    mod_nums = mod_nums[ok]
 
    idx_hot  = np.argmax(teff)
    idx_cold = np.argmin(teff)
    mn_hot   = mod_nums[idx_hot]
    mn_cold  = mod_nums[idx_cold]
 
    print(f'  Hottest:  model={mn_hot},  Teff={teff[idx_hot]:.0f} K, '
          f'R={radius[idx_hot]:.4f} Rsun, phase={phase[idx_hot]:.3f}')
    print(f'  Coldest:  model={mn_cold}, Teff={teff[idx_cold]:.0f} K, '
          f'R={radius[idx_cold]:.4f} Rsun, phase={phase[idx_cold]:.3f}')
 
    sed_hot  = load_sed_for_model(sed_dir, mn_hot)
    sed_cold = load_sed_for_model(sed_dir, mn_cold)
    if sed_hot is None or sed_cold is None:
        print('  fig_teff_sed: could not load SEDs — skipping.')
        return
 
    label_hot  = (r'Hottest: $T_{\rm eff}$=' + f'{teff[idx_hot]:.0f} K, '
                  r'$R$=' + f'{radius[idx_hot]:.4f}' + r' $R_\odot$')
    label_cold = (r'Coldest: $T_{\rm eff}$=' + f'{teff[idx_cold]:.0f} K, '
                  r'$R$=' + f'{radius[idx_cold]:.4f}' + r' $R_\odot$')
 
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    fig    = plt.figure(figsize=(13, 6.5))
    outer  = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.5], wspace=0.38)
    ax_left = fig.add_subplot(outer[0, 0])
    inner_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 1], hspace=0.0)
 
    # Left panel
    sc = ax_left.scatter(color, teff, c=phase, cmap='twilight_shifted',
                         s=8, alpha=0.8, edgecolors='none', rasterized=True)
    plt.colorbar(sc, ax=ax_left, label='Pulsation Phase', shrink=1., pad=0.0)
    add_arrow(ax_left, phase, color, teff, phi=0.25)
    ax_left.scatter(color[idx_hot], teff[idx_hot],
                    marker='*', s=200, color='#CC2200', zorder=10,
                    label=f'Hottest')# (model {mn_hot})')
    ax_left.scatter(color[idx_cold], teff[idx_cold],
                    marker='s', s=100, color='#1166AA', zorder=10,
                    label=f'Coldest')# (model {mn_cold})')
    ax_left.set_xlabel(r'$F_{062} - F_{184}$ (mag)')
    ax_left.set_ylabel(r'$T_{\rm eff}$ (K)')
    #ax_left.set_title('Temperature vs Roman colour', fontsize=10)
    ax_left.legend(framealpha=0.9, fontsize=9)
    ax_left.grid(True, alpha=0.18, lw=0.5)
 
    # Right panels — real SEDs
    _right_sed_panels(fig, inner_gs,
                      sed_hot, sed_cold,
                      label_hot, label_cold,
                      '#CC2200', '#1166AA',
                      r'Real atmosphere SED at hottest/coldest $T_{\rm eff}$')
 
    #fig.suptitle(r'Temperature variation: real MESA Colors SEDs at hottest/coldest $T_{\rm eff}$',
    #             fontsize=12, y=1.01)
    save(fig, output_dir, 'fig_teff_sed')

# ===========================================================================
# Standard figures
# ===========================================================================

def plot_lightcurves(data, output_dir, min_periods=4.0):
    mag_cols = find_roman_columns(data)
    show = [f for f in ['F062', 'F106', 'F146', 'F184', 'F213'] if f in mag_cols]
    if not show:
        print('fig_lightcurves: no suitable Roman columns — skipping.'); return

    phase = get_phase(data)
    if 'rsp_num_periods' in data.dtype.names:
        mask = np.asarray(data['rsp_num_periods'], dtype=float) >= min_periods
    else:
        t = get_time_days(data)
        mask = t > 3. if t is not None else np.ones(len(data), dtype=bool)
    if mask.sum() < 100:
        mask = np.ones(len(data), dtype=bool)

    ph = phase[mask]
    fig, axes = plt.subplots(len(show), 1, figsize=(7, 1.9*len(show)+0.5), sharex=True)
    if len(show) == 1:
        axes = [axes]

    for ax, filt in zip(axes, show):
        p   = ROMAN_FILTERS[filt]
        mag = np.asarray(data[mag_cols[filt]], dtype=float)[mask]
        ok  = np.isfinite(mag)
        srt = np.argsort(ph[ok])
        ax.plot(ph[ok][srt], mag[ok][srt], '.', ms=1.5, alpha=0.55,
                color=p['color'], rasterized=True)
        ax.invert_yaxis()
        amp = np.nanmax(mag[ok]) - np.nanmin(mag[ok])
        ax.set_ylabel(f'{filt}\n(mag)', fontsize=10)
        ax.text(0.98, 0.93, f'A = {amp:.2f} mag',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                color=p['color'],
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.8))
        ax.grid(True, alpha=0.15, lw=0.5)

    axes[-1].set_xlabel('Pulsation Phase')
    #axes[0].set_title('Roman WFI phased light curves — amplitude decreases with wavelength')
    fig.tight_layout()
    save(fig, output_dir, 'fig_lightcurves')


def plot_amplitudes(data, output_dir, min_periods=4.0):
    mag_cols = find_roman_columns(data)
    if not mag_cols:
        print('fig_amplitudes: no Roman columns — skipping.'); return

    if 'rsp_num_periods' in data.dtype.names:
        mask = np.asarray(data['rsp_num_periods'], dtype=float) >= min_periods
    else:
        t = get_time_days(data)
        mask = t > 3. if t is not None else np.ones(len(data), dtype=bool)
    if mask.sum() < 100:
        mask = np.ones(len(data), dtype=bool)

    wls=[]; amps=[]; cols=[]; labels=[]
    for filt in [f for f in ROMAN_ORDER if f in mag_cols]:
        mag = np.asarray(data[mag_cols[filt]], dtype=float)[mask]
        ok  = np.isfinite(mag)
        if ok.sum() < 2: continue
        amp = np.nanmax(mag[ok]) - np.nanmin(mag[ok])
        wls.append(ROMAN_FILTERS[filt]['lambda_eff'])
        amps.append(amp); cols.append(ROMAN_FILTERS[filt]['color'])
        labels.append(filt)
    if not wls:
        print('fig_amplitudes: no valid data.'); return

    wls_a = np.array(wls); amps_a = np.array(amps)

    T0=6800.; dR_R=0.06; dT_T=0.12
    wl_th = np.linspace(5000, 24000, 500)
    x_th  = np.clip(H*C/(wl_th*AA*KB*T0), 1e-3, 500)
    amp_th = 2.5/np.log(10)*np.abs(2*dR_R + (x_th*np.exp(x_th)/(np.exp(x_th)-1.)-5.)*dT_T)
    i0 = np.argmin(np.abs(wl_th - wls_a[0]))
    scale = amps_a[0]/amp_th[i0] if amp_th[i0]>0 else 1.

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(wl_th/1e4, amp_th*scale, 'k--', lw=1.5, alpha=0.55,
            label=r'Linearised Planck ($\delta T/T=0.12$, $\delta R/R=0.06$)')
    ax.scatter(wls_a/1e4, amps_a, c=cols, s=130, zorder=5, edgecolors='k', lw=0.8)
    for w, a, lab in zip(wls_a/1e4, amps_a, labels):
        ax.annotate(lab, (w, a), xytext=(6, 5),
                    textcoords='offset points', fontsize=9,
                    color=ROMAN_FILTERS[lab]['color'])
    ax.set_xlabel(r'Effective Wavelength ($\mu$m)')
    ax.set_ylabel('Peak-to-trough amplitude (mag)')
    ax.legend(framealpha=0.92, loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0.4, 2.5)
    fig.tight_layout()
    save(fig, output_dir, 'fig_amplitudes')


def plot_phase_offset(data, output_dir, min_periods=4.0):
    mag_cols = find_roman_columns(data)
    if not mag_cols:
        print('fig_phase_offset: no Roman columns — skipping.'); return

    if 'rsp_num_periods' in data.dtype.names:
        mask = np.asarray(data['rsp_num_periods'], dtype=float) >= min_periods
    else:
        t = get_time_days(data)
        mask = t > 3. if t is not None else np.ones(len(data), dtype=bool)
    if mask.sum() < 100:
        mask = np.ones(len(data), dtype=bool)

    phase = get_phase(data)
    wls=[]; phis=[]; cols=[]; labels=[]
    for filt in [f for f in ROMAN_ORDER if f in mag_cols]:
        mag = np.asarray(data[mag_cols[filt]], dtype=float)[mask]
        ph  = phase[mask]
        ok  = np.isfinite(mag)
        if ok.sum() < 10: continue
        idx = np.argmin(mag[ok])
        wls.append(ROMAN_FILTERS[filt]['lambda_eff'])
        phis.append(ph[ok][idx])
        cols.append(ROMAN_FILTERS[filt]['color'])
        labels.append(filt)
    if not wls:
        print('fig_phase_offset: no valid data.'); return

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.scatter(np.array(wls)/1e4, phis, c=cols, s=130, zorder=5, edgecolors='k', lw=0.8)
    ax.plot(np.array(wls)/1e4, phis, 'k--', alpha=0.3, lw=1.2)
    for lab, w, phi in zip(labels, np.array(wls)/1e4, phis):
        ax.annotate(lab, (w, phi), xytext=(6, 5),
                    textcoords='offset points', fontsize=9,
                    color=ROMAN_FILTERS[lab]['color'])
    ax.set_xlabel(r'Effective Wavelength ($\mu$m)')
    ax.set_ylabel('Phase of Maximum Light')
    ##ax.set_title('Max-light phase shifts redward with wavelength')
    ax.set_xlim(0.4, 2.5)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    save(fig, output_dir, 'fig_phase_offset')


def plot_colorloop(data, output_dir, min_periods=4.0):
    mag_cols = find_roman_columns(data)
    for need in ('F062', 'F106', 'F184'):
        if need not in mag_cols:
            print(f'fig_colorloop: missing {need} — skipping.'); return

    F062  = np.asarray(data[mag_cols['F062']], dtype=float)
    F106  = np.asarray(data[mag_cols['F106']], dtype=float)
    F184  = np.asarray(data[mag_cols['F184']], dtype=float)
    color = F062 - F184
    phase = get_phase(data)
    mask  = get_settled_mask(data, min_periods=min_periods, single_cycle=False)
    if mask.sum() < 50:
        mask = np.ones(len(data), dtype=bool)

    ok = mask & finite_mask(color, F106, phase)
    ph = phase[ok]; col = color[ok]; mag = F106[ok]

    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(col, mag, c=ph, cmap='twilight_shifted',
                    s=10, alpha=0.85, edgecolors='none', rasterized=True)
    plt.colorbar(sc, ax=ax, label='Pulsation Phase', shrink=0.85)
    add_arrow(ax, ph, col, mag, phi=0.22)
    for phi, mk, lab in [(0., '*', 'max'), (0.5, 's', 'min')]:
        idx = np.argmin(np.abs(ph - phi))
        ax.plot(col[idx], mag[idx], mk, color='k', ms=10, zorder=10,
                label=f'$\\phi={phi:.1f}$ ({lab} light)')
    ax.invert_yaxis()
    ax.set_xlabel(r'$F_{062} - F_{184}$ (mag)')
    ax.set_ylabel(r'$F_{106}$ (mag)')
    ##ax.set_title('Colour-magnitude hysteresis loop: Roman WFI')
    ax.legend(framealpha=0.92, fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    save(fig, output_dir, 'fig_colorloop')


# ===========================================================================
# Radius and temperature flux diagnostics (from plot_rsp_radius_temp.py)
# ===========================================================================
def plot_radius_diagnostic(data, output_dir, min_periods=4.0):
    """
    1x2: LHS = phase vs normalised radius + phase vs normalised colour (same axes).
         RHS = radius vs colour, phase-coloured scatter.
    """
    mag_cols = find_roman_columns(data)
    for need in ('F062', 'F184'):
        if need not in mag_cols:
            print(f'fig_radius_diagnostic: missing {need} — skipping.'); return
    if 'photosphere_r' not in data.dtype.names:
        print('fig_radius_diagnostic: no photosphere_r — skipping.'); return

    mask = get_settled_mask(data, min_periods=min_periods, single_cycle=False)
    if mask.sum() < 50:
        mask = np.ones(len(data), dtype=bool)

    phase  = get_phase(data)[mask]
    radius = np.asarray(data['photosphere_r'], dtype=float)[mask]
    f062   = np.asarray(data[mag_cols['F062']], dtype=float)[mask]
    f184   = np.asarray(data[mag_cols['F184']], dtype=float)[mask]
    color  = f062 - f184

    ok = finite_mask(phase, radius, color)
    phase = phase[ok]; radius = radius[ok]; color = color[ok]

    srt = np.argsort(phase)
    phase = phase[srt]; radius = radius[srt]; color = color[srt]

    r_val = pearson_r(radius, color)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # LHS: phase vs radius and phase vs colour, both normalised to [-1, 1]
    ax = axes[0]
    ax.plot(phase, normalised(radius), '-', lw=2., color='k',
            label=r'Radius ($R_\odot$)')
    ax.plot(phase, normalised(color),  '-', lw=2.,
            color=ROMAN_FILTERS['F062']['color'],
            label=r'$F_{062} - F_{184}$')
    ax.set_xlabel('Pulsation Phase')
    ax.set_ylabel('Normalised variation')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.text(0.03, 0.05, f'Pearson $r$ = {r_val:.3f}',
            transform=ax.transAxes, fontsize=10)

    # RHS: radius vs colour, phase-coloured
    ax = axes[1]
    sc = ax.scatter(color, radius, c=phase, cmap='twilight_shifted',
                    s=10, alpha=0.85, edgecolors='none', rasterized=True)
    plt.colorbar(sc, ax=ax, label='Pulsation Phase', shrink=1.0, pad=0.01)
    add_arrow(ax, phase, color, radius, phi=0.25)
    ax.set_xlabel(r'$F_{062} - F_{184}$ (mag)')
    ax.set_ylabel(r'Radius ($R_\odot$)')
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    save(fig, output_dir, 'fig_radius_diagnostic')


def plot_temp_diagnostic(data, output_dir, min_periods=4.0):
    """
    1x2: LHS = phase vs normalised Teff + phase vs normalised colour (same axes).
         RHS = Teff vs colour, phase-coloured scatter.
    """
    teff_arr = get_teff(data)
    if teff_arr is None:
        print('fig_temp_diagnostic: no Teff — skipping.'); return

    mag_cols = find_roman_columns(data)
    for need in ('F062', 'F184'):
        if need not in mag_cols:
            print(f'fig_temp_diagnostic: missing {need} — skipping.'); return

    mask = get_settled_mask(data, min_periods=min_periods, single_cycle=False)
    if mask.sum() < 50:
        mask = np.ones(len(data), dtype=bool)

    phase = get_phase(data)[mask]
    teff  = teff_arr[mask]
    f062  = np.asarray(data[mag_cols['F062']], dtype=float)[mask]
    f184  = np.asarray(data[mag_cols['F184']], dtype=float)[mask]
    color = f062 - f184

    ok = finite_mask(phase, teff, color)
    phase = phase[ok]; teff = teff[ok]; color = color[ok]

    srt = np.argsort(phase)
    phase = phase[srt]; teff = teff[srt]; color = color[srt]

    r_val = pearson_r(teff, color)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # LHS: phase vs Teff and phase vs colour, both normalised
    ax = axes[0]
    ax.plot(phase, normalised(teff),  '-', lw=2., color='k',
            label=r'$T_{\rm eff}$ (K)')
    ax.plot(phase, normalised(color), '-', lw=2.,
            color=ROMAN_FILTERS['F062']['color'],
            label=r'$F_{062} - F_{184}$')
    ax.set_xlabel('Pulsation Phase')
    ax.set_ylabel('Normalised variation')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.text(0.03, 0.05, f'Pearson $r$ = {r_val:.3f}',
            transform=ax.transAxes, fontsize=10)

    # RHS: Teff vs colour, phase-coloured
    ax = axes[1]
    sc = ax.scatter(color, teff, c=phase, cmap='twilight_shifted',
                    s=10, alpha=0.85, edgecolors='none', rasterized=True)
    plt.colorbar(sc, ax=ax, label='Pulsation Phase', shrink=1.0, pad=0.01)
    add_arrow(ax, phase, color, teff, phi=0.25)
    ax.set_xlabel(r'$F_{062} - F_{184}$ (mag)')
    ax.set_ylabel(r'$T_{\rm eff}$ (K)')
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    save(fig, output_dir, 'fig_temp_diagnostic')

# ===========================================================================
# CMD / CC comprehensive plots (all band combinations)
# ===========================================================================

def plot_cmd_cc(data, output_dir, min_periods=4.0):
    """
    Generate every colour-magnitude and colour-colour combination for all
    available Roman bands.  Saved into cmd/ and cc/ subdirectories.
    """
    mag_cols = find_roman_columns(data)
    available = [f for f in ROMAN_ORDER if f in mag_cols]
    if len(available) < 2:
        print('plot_cmd_cc: fewer than 2 Roman bands — skipping.'); return

    phase = get_phase(data)
    mask  = get_settled_mask(data, min_periods=min_periods, single_cycle=False)
    if mask.sum() < 50:
        mask = np.ones(len(data), dtype=bool)

    ph = phase[mask]

    def ann(ax, x, y, invert_y):
        i1 = np.argmin(np.abs(ph - 0.25))
        i2 = np.argmin(np.abs(ph - 0.30))
        if i1 != i2:
            ax.annotate('', xy=(x[i2], y[i2]), xytext=(x[i1], y[i1]),
                        arrowprops=dict(arrowstyle='->', color='k', lw=1.5))
        for tgt, mk in [(0., '*'), (0.5, 's')]:
            idx = np.argmin(np.abs(ph - tgt))
            ax.plot(x[idx], y[idx], mk, color='k', ms=9, zorder=10)
        if invert_y:
            ax.invert_yaxis()
        ax.grid(True, alpha=0.2)

    mags = {f: np.asarray(data[mag_cols[f]], dtype=float)[mask] for f in available}

    # CMD
    cmd_dir = output_dir / 'cmd'; cmd_dir.mkdir(exist_ok=True)
    for b1, b2 in combinations(available, 2):
        col_vals = mags[b1] - mags[b2]
        for mag_band in available:
            ok = np.isfinite(col_vals) & np.isfinite(mags[mag_band])
            fig, ax = plt.subplots(figsize=(5, 5))
            sc = ax.scatter(col_vals[ok], mags[mag_band][ok],
                            c=ph[ok], cmap='twilight_shifted',
                            s=8, alpha=0.8, edgecolors='none')
            plt.colorbar(sc, ax=ax, label='Phase', shrink=0.8)
            ann(ax, col_vals[ok], mags[mag_band][ok], invert_y=True)
            ax.set_xlabel(f'${b1} - {b2}$ (mag)')
            ax.set_ylabel(f'${mag_band}$ (mag)')
            stem = f'cmd_{b1}minus{b2}_vs_{mag_band}'
            fig.savefig(cmd_dir / f'{stem}.pdf', dpi=300, bbox_inches='tight')
            fig.savefig(cmd_dir / f'{stem}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved cmd/{stem}.pdf')

    # CC
    cc_dir  = output_dir / 'cc';  cc_dir.mkdir(exist_ok=True)
    color_pairs = list(combinations(available, 2))
    for i, (b1a, b1b) in enumerate(color_pairs):
        x_full = mags[b1a] - mags[b1b]
        for b2a, b2b in color_pairs[i+1:]:
            y_full = mags[b2a] - mags[b2b]
            ok = np.isfinite(x_full) & np.isfinite(y_full)
            fig, ax = plt.subplots(figsize=(5, 5))
            sc = ax.scatter(x_full[ok], y_full[ok],
                            c=ph[ok], cmap='twilight_shifted',
                            s=8, alpha=0.8, edgecolors='none')
            plt.colorbar(sc, ax=ax, label='Phase', shrink=0.8)
            ann(ax, x_full[ok], y_full[ok], invert_y=False)
            ax.set_xlabel(f'${b1a} - {b1b}$ (mag)')
            ax.set_ylabel(f'${b2a} - {b2b}$ (mag)')
            stem = f'cc_{b1a}minus{b1b}_vs_{b2a}minus{b2b}'
            fig.savefig(cc_dir / f'{stem}.pdf', dpi=300, bbox_inches='tight')
            fig.savefig(cc_dir / f'{stem}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved cc/{stem}.pdf')


# ===========================================================================
# Main
# ===========================================================================

def print_statistics(data, mag_cols, min_periods=4.0):
    """
    Print statistics for the full run and for the settled multi-cycle sample.

    The plotted analysis uses all models with rsp_num_periods >= min_periods
    when that column is available.  These statistics use the same selection.
    """
    print('\n=== Run statistics ===')
    print(f'Total models: {len(data)}')

    periods = None
    if 'rsp_num_periods' in data.dtype.names:
        periods = np.asarray(data['rsp_num_periods'], dtype=float)
        finite_periods = np.isfinite(periods)
        if finite_periods.any():
            n = int(np.nanmax(periods[finite_periods]))
            print(f'Pulsation cycles: {n},  ~{len(data)//max(n,1)} pts/cycle')

    settled = get_settled_mask(data, min_periods=min_periods, single_cycle=False)
    if settled.sum() < 50:
        settled = np.ones(len(data), dtype=bool)

    print(f'Settled sample: {settled.sum()} models '
          f'(rsp_num_periods >= {min_periods:g}, all later cycles)')

    cycle_ids = None
    if periods is not None:
        finite_settled = settled & np.isfinite(periods)
        if finite_settled.any():
            pmin = np.nanmin(periods[finite_settled])
            pmax = np.nanmax(periods[finite_settled])
            cycle_ids = np.floor(periods[finite_settled]).astype(int)
            cycle_ids = np.unique(cycle_ids)
            print(f'Settled period span: {pmin:.3f} to {pmax:.3f}')
            if len(cycle_ids) > 0:
                print(f'Cycle bins represented: {cycle_ids[0]} to {cycle_ids[-1]} '
                      f'({len(cycle_ids)} bins)')

    if 'rsp_period_in_days' in data.dtype.names:
        p = np.asarray(data['rsp_period_in_days'], dtype=float)
        v_all = np.isfinite(p) & (p > 0)
        v_set = v_all & settled
        if v_all.any():
            print(f'Mean period, full run:    {np.nanmean(p[v_all]):.5f} days')
        if v_set.any():
            print(f'Mean period, settled:     {np.nanmean(p[v_set]):.5f} days')

    def per_cycle_amp(values):
        if periods is None:
            return None
        p = np.asarray(periods, dtype=float)
        values = np.asarray(values, dtype=float)
        ok_base = settled & np.isfinite(p) & np.isfinite(values)
        if ok_base.sum() < 10:
            return None
        ids = np.floor(p[ok_base]).astype(int)
        vals = values[ok_base]
        amps = []
        for cid in np.unique(ids):
            m = ids == cid
            if m.sum() >= 10:
                amps.append(np.nanmax(vals[m]) - np.nanmin(vals[m]))
        if len(amps) == 0:
            return None
        amps = np.asarray(amps, dtype=float)
        return np.nanmean(amps), np.nanstd(amps), len(amps)

    print('\nSettled multi-cycle amplitudes:')
    for filt in ROMAN_ORDER:
        if filt in mag_cols:
            mag = np.asarray(data[mag_cols[filt]], dtype=float)
            ok = settled & np.isfinite(mag)
            if ok.any():
                amp = np.nanmax(mag[ok]) - np.nanmin(mag[ok])
                cyc = per_cycle_amp(mag)
                if cyc is None:
                    print(f'  {filt}: A = {amp:.3f} mag')
                else:
                    mean_amp, std_amp, ncyc = cyc
                    print(f'  {filt}: A = {amp:.3f} mag; '
                          f'cycle mean = {mean_amp:.3f} ± {std_amp:.3f} mag '
                          f'(N={ncyc})')

    if 'F062' in mag_cols and 'F184' in mag_cols:
        f062 = np.asarray(data[mag_cols['F062']], dtype=float)
        f184 = np.asarray(data[mag_cols['F184']], dtype=float)
        color = f062 - f184
        ok = settled & np.isfinite(color)
        if ok.any():
            amp = np.nanmax(color[ok]) - np.nanmin(color[ok])
            print(f'\nSettled colour F062-F184: '
                  f'{np.nanmin(color[ok]):.3f} to {np.nanmax(color[ok]):.3f} '
                  f'(Δ = {amp:.3f} mag)')

        if 'photosphere_r' in data.dtype.names:
            radius = np.asarray(data['photosphere_r'], dtype=float)
            ok_r = settled & np.isfinite(radius) & np.isfinite(color)
            if ok_r.any():
                print(f'Settled radius: {np.nanmin(radius[ok_r]):.4f} to '
                      f'{np.nanmax(radius[ok_r]):.4f} Rsun '
                      f'(Pearson r with F062-F184 = {pearson_r(radius[ok_r], color[ok_r]):.3f})')

        teff = get_teff(data)
        if teff is not None:
            ok_t = settled & np.isfinite(teff) & np.isfinite(color)
            if ok_t.any():
                print(f'Settled Teff: {np.nanmin(teff[ok_t]):.0f} to '
                      f'{np.nanmax(teff[ok_t]):.0f} K '
                      f'(Pearson r with F062-F184 = {pearson_r(teff[ok_t], color[ok_t]):.3f})')

    print('======================\n')

def main():
    parser = argparse.ArgumentParser(
        description='Roman WFI validation plots for MESA Colors RSP RR Lyrae')
    parser.add_argument('history_file', nargs='?', default=None)
    parser.add_argument('--logs_dir',    default=str(DEFAULT_LOGS))
    parser.add_argument('--sed_dir',     default=None,
                        help='Directory containing {FILTER}_SED_{MODEL}.csv files '
                             '(default: ../SED relative to this script)')
    parser.add_argument('--output_dir',  default=str(DEFAULT_OUT))
    parser.add_argument('--min_periods', type=float, default=4.0,
                        help='Settled-cycle threshold (default 4)')
    parser.add_argument('--cmd', action='store_true',
                        help='Also generate all CMD and CC combination plots')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hf = (Path(args.history_file) if args.history_file
          else Path(args.logs_dir) / 'history.data')
    if not hf.exists():
        print(f'ERROR: {hf} not found'); sys.exit(1)

    # SED directory: explicit arg > ../SED relative to this script
    sed_dir = Path(args.sed_dir) if args.sed_dir else PARENT_DIR / 'SED'
    if not sed_dir.exists():
        print(f'WARNING: SED directory not found: {sed_dir}')
        print('  SED-based figures will be skipped.')
        print('  Pass --sed_dir /path/to/SED to specify the location.')
    else:
        print(f'SED directory: {sed_dir}')

    print(f'Loading {hf}...')
    data = load_mesa_history(str(hf))
    print(f'Loaded {len(data)} models')

    mag_cols = find_roman_columns(data)
    print_statistics(data, mag_cols, args.min_periods)

    mp = args.min_periods

    print('--- fig_radius_sed ---')
    plot_radius_sed(data, output_dir, sed_dir, mp)

    print('--- fig_teff_sed ---')
    plot_teff_sed(data, output_dir, sed_dir, mp)

    print('--- fig_lightcurves ---')
    plot_lightcurves(data, output_dir, mp)

    print('--- fig_amplitudes ---')
    plot_amplitudes(data, output_dir, mp)

    print('--- fig_phase_offset ---')
    plot_phase_offset(data, output_dir, mp)

    print('--- fig_colorloop ---')
    plot_colorloop(data, output_dir, mp)

    print('--- fig_radius_diagnostic ---')
    plot_radius_diagnostic(data, output_dir, mp)

    print('--- fig_temp_diagnostic ---')
    plot_temp_diagnostic(data, output_dir, mp)

    if args.cmd:
        print('--- CMD / CC plots ---')
        plot_cmd_cc(data, output_dir, mp)

    print(f'\nDone. Figures saved to {output_dir}')


if __name__ == '__main__':
    main()
