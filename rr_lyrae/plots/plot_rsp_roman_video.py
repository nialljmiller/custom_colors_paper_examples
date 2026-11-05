#!/usr/bin/env python3
"""
plot_rsp_roman_video.py
=======================
Animated Roman WFI diagnostics for the MESA Colors RSP RR Lyrae run.

This is the video analogue of the static Roman SED/loop figures.
It produces animated movies in which the post-settling loop is traced in time,
while the right-hand SED panels compare a fixed reference state against the
current moving state.

Outputs:
  movie_radius_sed.mp4 (or .gif fallback)
  movie_teff_sed.mp4   (or .gif fallback)

Key behaviour:
  * Uses all settled cycles after min_periods.
  * Left panel: the loop is shown as a phase-coloured background, with the
    visited trajectory and current point animated in time order.
  * Top-right: reference SED (fixed) versus current SED (moving).
  * Bottom-right: live delta SED = current - reference, so it evolves frame by
    frame as the current point moves around the loop.
  * The fixed reference point can be chosen from a maximum or minimum.

Usage:
  python plot_rsp_roman_video.py
  python plot_rsp_roman_video.py /path/to/LOGS/history.data
  python plot_rsp_roman_video.py --logs_dir /path --output_dir /path/plots
  python plot_rsp_roman_video.py --min_periods 4 --stride 5 --fps 24
  python plot_rsp_roman_video.py --radius_ref min --teff_ref cold
"""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.ticker as mticker

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'axes.labelsize': 12, 'axes.titlesize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 9, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

SCRIPT_DIR   = Path(__file__).resolve().parent
PARENT_DIR   = SCRIPT_DIR.parent
DEFAULT_LOGS = PARENT_DIR / 'LOGS'
DEFAULT_OUT  = SCRIPT_DIR

ROMAN_FILTERS = {
    'F062': {'lambda_eff': 6200,  'lambda_lo': 4800,  'lambda_hi': 7600,  'color': '#CC2200'},
    'F087': {'lambda_eff': 8700,  'lambda_lo': 7600,  'lambda_hi': 10000, 'color': '#FF6600'},
    'F106': {'lambda_eff': 10600, 'lambda_lo': 9300,  'lambda_hi': 11900, 'color': '#FFAA00'},
    'F129': {'lambda_eff': 12900, 'lambda_lo': 11500, 'lambda_hi': 14500, 'color': '#99CC00'},
    'F158': {'lambda_eff': 15800, 'lambda_lo': 13800, 'lambda_hi': 17700, 'color': '#33BB55'},
    'F184': {'lambda_eff': 18400, 'lambda_lo': 16800, 'lambda_hi': 20000, 'color': '#1166AA'},
    'F213': {'lambda_eff': 21300, 'lambda_lo': 19500, 'lambda_hi': 23000, 'color': '#660099'},
}
ROMAN_ORDER = ['F062', 'F087', 'F106', 'F129', 'F158', 'F184', 'F213']
WL_LO = 4500.0
WL_HI = 24000.0


def load_mesa_history(filepath):
    filepath = Path(filepath)
    with open(filepath, encoding='utf-8') as f:
        lines = f.readlines()
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
    data = np.genfromtxt(filepath, skip_header=header_idx + 1, names=cols,
                         invalid_raise=False, autostrip=True)
    if getattr(data, 'shape', ()) == ():
        data = np.array([data], dtype=data.dtype)
    return data


def find_roman_columns(data):
    colnames = list(data.dtype.names)
    cs = set(colnames)
    found = {}
    for filt in ROMAN_ORDER:
        fl = filt.lower()
        matched = None
        for cand in [filt, 'Roman_'+filt, 'WFI_'+filt, 'Roman_WFI_'+filt,
                     fl, 'roman_'+fl, 'wfi_'+fl]:
            if cand in cs:
                matched = cand
                break
        if matched is None:
            for col in colnames:
                if fl in col.lower():
                    matched = col
                    break
        if matched is not None:
            found[filt] = matched
    return found


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
        return np.linspace(0.0, 1.0, len(data), endpoint=False)
    period = 0.71
    if 'rsp_period_in_days' in data.dtype.names:
        p = np.asarray(data['rsp_period_in_days'], dtype=float)
        v = np.isfinite(p) & (p > 0)
        if v.any():
            period = float(np.nanmedian(p[v]))
    return ((t - t[0]) / period) % 1.0


def get_settled_mask(data, min_periods=4.0):
    if 'rsp_num_periods' in data.dtype.names:
        periods = np.asarray(data['rsp_num_periods'], dtype=float)
        finite = np.isfinite(periods)
        cand = finite & (periods >= min_periods)
        if cand.sum() >= 50:
            return cand
    t = get_time_days(data)
    if t is not None:
        cand = np.asarray(t, dtype=float) > 3.0
        if cand.sum() >= 50:
            return cand
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
        return 10.0 ** np.asarray(data['log_Teff'], dtype=float)
    return None


def load_sed_for_model(sed_dir, model_number, preferred_filter='F062'):
    sed_dir = Path(sed_dir)
    model_str = f'{int(model_number):08d}'
    candidates = [f'{preferred_filter}_SED_{model_str}.csv']
    for filt in ROMAN_ORDER:
        fname = f'{filt}_SED_{model_str}.csv'
        if fname not in candidates:
            candidates.append(fname)
    for fname in candidates:
        fpath = sed_dir / fname
        if not fpath.exists():
            continue
        wls = []
        fls = []
        try:
            with open(fpath, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        wl = float(row['wavelengths'])
                        fl = float(row['fluxes'])
                    except (ValueError, KeyError):
                        continue
                    if wl == 0:
                        continue
                    wls.append(wl)
                    fls.append(fl)
        except Exception as e:
            print(f'Warning: could not read {fpath}: {e}')
            continue
        if wls:
            wls = np.array(wls, dtype=float)
            fls = np.array(fls, dtype=float)
            srt = np.argsort(wls)
            return {'wavelengths': wls[srt], 'fluxes': fls[srt]}
    return None


def shade_roman(ax, alpha_span=0.07, alpha_line=0.28, label_ax=None):
    for filt in ROMAN_ORDER:
        p = ROMAN_FILTERS[filt]
        if p['lambda_hi'] < WL_LO or p['lambda_lo'] > WL_HI:
            continue
        ax.axvspan(p['lambda_lo'], p['lambda_hi'],
                   alpha=alpha_span, color=p['color'], lw=0, zorder=0)
        ax.axvline(p['lambda_eff'], color=p['color'],
                   alpha=alpha_line, lw=0.8, ls='--', zorder=1)
        if label_ax is not None:
            label_ax.text(p['lambda_eff'], 0.93, filt,
                          transform=label_ax.get_xaxis_transform(),
                          ha='center', va='bottom', fontsize=8,
                          color=p['color'], fontweight='bold')


def interp_and_clip_sed(sed, wl_grid):
    wl = np.asarray(sed['wavelengths'], dtype=float)
    fl = np.asarray(sed['fluxes'], dtype=float)
    m = (wl >= WL_LO) & (wl <= WL_HI) & np.isfinite(fl) & (fl > 0)
    wl = wl[m]
    fl = fl[m]
    if wl.size < 2:
        return None
    return np.interp(wl_grid, wl, fl, left=np.nan, right=np.nan)


def filter_mean(wl, flux, lo, hi):
    m = (wl >= lo) & (wl <= hi) & np.isfinite(flux)
    if m.sum() < 2:
        return np.nan
    return np.trapezoid(flux[m], wl[m]) / (hi - lo)


def choose_indices(n, stride=1, max_frames=None):
    idx = np.arange(0, n, max(1, int(stride)), dtype=int)
    if idx[-1] != n - 1:
        idx = np.r_[idx, n - 1]
    if max_frames is not None and len(idx) > max_frames:
        idx = np.linspace(0, n - 1, max_frames).round().astype(int)
        idx = np.unique(idx)
        if idx[-1] != n - 1:
            idx = np.r_[idx, n - 1]
    return idx


def pick_writer(fps):
    if animation.writers.is_available('ffmpeg'):
        return animation.FFMpegWriter(fps=fps, bitrate=2500), '.mp4'
    return animation.PillowWriter(fps=fps), '.gif'


def save_animation(anim, output_dir, stem, fps):
    writer, ext = pick_writer(fps)
    out = Path(output_dir) / f'{stem}{ext}'
    print(f'Saving {out} ...')
    anim.save(out, writer=writer, dpi=150)
    print(f'  Saved {out}')
    return out


def build_radius_dataset(data, sed_dir, min_periods=4.0, stride=5, max_frames=None, reference='max'):
    mag_cols = find_roman_columns(data)
    for need in ('F062', 'F184'):
        if need not in mag_cols:
            raise ValueError(f'Missing Roman column {need}')
    if 'photosphere_r' not in data.dtype.names:
        raise ValueError('No photosphere_r column found')

    mask = get_settled_mask(data, min_periods=min_periods)
    phase = get_phase(data)[mask]
    model = np.asarray(data['model_number'], dtype=int)[mask]
    radius = np.asarray(data['photosphere_r'], dtype=float)[mask]
    f062 = np.asarray(data[mag_cols['F062']], dtype=float)[mask]
    f184 = np.asarray(data[mag_cols['F184']], dtype=float)[mask]
    teff = get_teff(data)
    teff = teff[mask] if teff is not None else np.full(mask.sum(), np.nan)
    color = f062 - f184
    t = get_time_days(data)
    t = t[mask] if t is not None else np.arange(mask.sum(), dtype=float)

    ok = finite_mask(phase, model, radius, color, t)
    phase = phase[ok]; model = model[ok]; radius = radius[ok]
    color = color[ok]; teff = teff[ok]; t = t[ok]

    order = np.argsort(model)
    phase = phase[order]; model = model[order]; radius = radius[order]
    color = color[order]; teff = teff[order]; t = t[order]

    ref_idx = int(np.nanargmax(radius) if reference == 'max' else np.nanargmin(radius))
    frame_idx = choose_indices(len(model), stride=stride, max_frames=max_frames)
    if ref_idx not in frame_idx:
        frame_idx = np.unique(np.r_[frame_idx, ref_idx])
        frame_idx.sort()

    sed_cache = {}
    used = []
    for i in frame_idx:
        sed = load_sed_for_model(sed_dir, model[i])
        if sed is not None:
            sed_cache[int(i)] = sed
            used.append(i)
    used = np.array(used, dtype=int)
    if ref_idx not in sed_cache:
        raise RuntimeError(f'Could not load reference SED for model {model[ref_idx]}')
    if used.size < 2:
        raise RuntimeError('Too few usable SED frames for animation')

    ref_label = 'Max R' if reference == 'max' else 'Min R'
    return {
        'name': 'radius',
        'title_left': 'Radius vs Roman colour',
        'y_label': r'Photospheric Radius ($R_\odot$)',
        'x_label': r'$F_{062} - F_{184}$ (mag)',
        'y': radius,
        'x': color,
        'phase': phase,
        'model': model,
        'teff': teff,
        'time': t,
        'frame_idx': used,
        'sed_cache': sed_cache,
        'ref_idx': ref_idx,
        'ref_label': ref_label,
        'ref_color': '#CC2200' if reference == 'max' else '#1166AA',
        'cur_color': '#1166AA' if reference == 'max' else '#CC2200',
        'movie_stem': 'movie_radius_sed',
        'reference_mode': reference,
    }


def build_teff_dataset(data, sed_dir, min_periods=4.0, stride=5, max_frames=None, reference='hot'):
    mag_cols = find_roman_columns(data)
    for need in ('F062', 'F184'):
        if need not in mag_cols:
            raise ValueError(f'Missing Roman column {need}')
    teff = get_teff(data)
    if teff is None:
        raise ValueError('No effective temperature column found')
    if 'photosphere_r' not in data.dtype.names:
        raise ValueError('No photosphere_r column found')

    mask = get_settled_mask(data, min_periods=min_periods)
    phase = get_phase(data)[mask]
    model = np.asarray(data['model_number'], dtype=int)[mask]
    teff = np.asarray(teff, dtype=float)[mask]
    radius = np.asarray(data['photosphere_r'], dtype=float)[mask]
    f062 = np.asarray(data[mag_cols['F062']], dtype=float)[mask]
    f184 = np.asarray(data[mag_cols['F184']], dtype=float)[mask]
    color = f062 - f184
    t = get_time_days(data)
    t = t[mask] if t is not None else np.arange(mask.sum(), dtype=float)

    ok = finite_mask(phase, model, teff, radius, color, t)
    phase = phase[ok]; model = model[ok]; teff = teff[ok]
    radius = radius[ok]; color = color[ok]; t = t[ok]

    order = np.argsort(model)
    phase = phase[order]; model = model[order]; teff = teff[order]
    radius = radius[order]; color = color[order]; t = t[order]

    ref_idx = int(np.nanargmax(teff) if reference == 'hot' else np.nanargmin(teff))
    frame_idx = choose_indices(len(model), stride=stride, max_frames=max_frames)
    if ref_idx not in frame_idx:
        frame_idx = np.unique(np.r_[frame_idx, ref_idx])
        frame_idx.sort()

    sed_cache = {}
    used = []
    for i in frame_idx:
        sed = load_sed_for_model(sed_dir, model[i])
        if sed is not None:
            sed_cache[int(i)] = sed
            used.append(i)
    used = np.array(used, dtype=int)
    if ref_idx not in sed_cache:
        raise RuntimeError(f'Could not load reference SED for model {model[ref_idx]}')
    if used.size < 2:
        raise RuntimeError('Too few usable SED frames for animation')

    ref_label = 'Hottest' if reference == 'hot' else 'Coldest'
    return {
        'name': 'teff',
        'title_left': 'Temperature vs Roman colour',
        'y_label': r'$T_{\rm eff}$ (K)',
        'x_label': r'$F_{062} - F_{184}$ (mag)',
        'y': teff,
        'x': color,
        'phase': phase,
        'model': model,
        'radius': radius,
        'time': t,
        'frame_idx': used,
        'sed_cache': sed_cache,
        'ref_idx': ref_idx,
        'ref_label': ref_label,
        'ref_color': '#CC2200' if reference == 'hot' else '#1166AA',
        'cur_color': '#1166AA' if reference == 'hot' else '#CC2200',
        'movie_stem': 'movie_teff_sed',
        'reference_mode': reference,
    }


def prepare_sed_products(dataset):
    ref_idx = int(dataset['ref_idx'])
    sed_ref = dataset['sed_cache'][ref_idx]
    wl_ref = np.asarray(sed_ref['wavelengths'], dtype=float)
    mref = (wl_ref >= WL_LO) & (wl_ref <= WL_HI) & np.isfinite(sed_ref['fluxes']) & (np.asarray(sed_ref['fluxes']) > 0)
    wl_grid = wl_ref[mref]
    if wl_grid.size < 2:
        raise RuntimeError('Reference SED has too few points in Roman wavelength window')

    ref_flux = interp_and_clip_sed(sed_ref, wl_grid)
    if ref_flux is None:
        raise RuntimeError('Failed to interpolate reference SED')
    ref_norm = np.nanmax(ref_flux)

    products = {'wl': wl_grid, 'ref_flux': ref_flux / ref_norm, 'frames': {}}
    ymax_top = np.nanmax(products['ref_flux'])
    delta_min = np.inf
    delta_max = -np.inf

    for i in dataset['frame_idx']:
        sed_cur = dataset['sed_cache'][int(i)]
        cur_flux = interp_and_clip_sed(sed_cur, wl_grid)
        if cur_flux is None:
            continue
        cur_norm = cur_flux / ref_norm
        delta = cur_norm - products['ref_flux']
        circles = {}
        for filt in ROMAN_ORDER:
            p = ROMAN_FILTERS[filt]
            circles[filt] = filter_mean(wl_grid, delta, p['lambda_lo'], p['lambda_hi'])
        products['frames'][int(i)] = {'cur_flux': cur_norm, 'delta': delta, 'circles': circles}
        ymax_top = max(ymax_top, np.nanmax(cur_norm[np.isfinite(cur_norm)]))
        if np.any(np.isfinite(delta)):
            delta_min = min(delta_min, np.nanmin(delta[np.isfinite(delta)]))
            delta_max = max(delta_max, np.nanmax(delta[np.isfinite(delta)]))

    if not np.isfinite(delta_min):
        delta_min, delta_max = -0.1, 0.1

    pad = 0.08 * max(abs(delta_min), abs(delta_max), 1e-6)
    products['ymax_top'] = 1.05 * ymax_top
    products['delta_ylim'] = (delta_min - pad, delta_max + pad)
    return products


def make_movie(dataset, output_dir, fps=24):
    sed_products = prepare_sed_products(dataset)
    wl = sed_products['wl']
    ref_flux = sed_products['ref_flux']
    phase = dataset['phase']
    x = dataset['x']
    y = dataset['y']
    frame_idx = [int(i) for i in dataset['frame_idx'] if int(i) in sed_products['frames']]
    ref_idx = int(dataset['ref_idx'])
    ref_color = dataset['ref_color']
    cur_color = dataset['cur_color']

    fig = plt.figure(figsize=(13, 6.5))
    outer = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.5], wspace=0.38)
    ax_left = fig.add_subplot(outer[0, 0])
    inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 1], hspace=0.0)
    ax_sed = fig.add_subplot(inner[0])
    ax_delta = fig.add_subplot(inner[1], sharex=ax_sed)

    # Left panel: static background of the full multi-cycle loop
    sc = ax_left.scatter(x, y, c=phase, cmap='twilight_shifted',
                         s=8, alpha=0.25, edgecolors='none', rasterized=True)
    plt.colorbar(sc, ax=ax_left, label='Pulsation Phase', shrink=1.0, pad=0.0)
    ax_left.set_xlabel(dataset['x_label'])
    ax_left.set_ylabel(dataset['y_label'])
    ax_left.grid(True, alpha=0.18, lw=0.5)

    ref_marker_style = '*' if dataset['ref_label'] in ('Max R', 'Hottest') else 's'
    ax_left.scatter(x[ref_idx], y[ref_idx], marker=ref_marker_style,
                    s=200 if ref_marker_style == '*' else 100,
                    color=ref_color, zorder=10, label=dataset['ref_label'])
    ax_left.legend(framealpha=0.9, fontsize=9, loc='best')

    trail_line, = ax_left.plot([], [], '-', lw=1.4, color='k', alpha=0.75)
    current_point, = ax_left.plot([], [], 'o', ms=7, color=cur_color,
                                  mec='k', mew=0.6, zorder=11)

    txt_left = ax_left.text(1.1, 0.1, '', transform=ax_left.transAxes,
                            ha='left', va='top', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.85))

    # Right panels
    shade_roman(ax_sed, label_ax=ax_sed)
    shade_roman(ax_delta)

    ref_label = dataset['ref_label']
    ref_info = []
    if 'teff' in dataset:
        ref_info.append(r'$T_{\rm eff}$=' + f'{dataset["teff"][ref_idx]:.0f} K')
    if 'radius' in dataset:
        ref_info.append(r'$R$=' + f'{dataset["radius"][ref_idx]:.4f}' + r' $R_\odot$')
    elif dataset['name'] == 'radius':
        ref_info.append(r'$R$=' + f'{y[ref_idx]:.4f}' + r' $R_\odot$')
    ref_label_full = ref_label + ': ' + ', '.join(ref_info)

    ref_line, = ax_sed.plot(wl, ref_flux, color=ref_color, lw=2.2, label=ref_label_full)
    cur_line, = ax_sed.plot([], [], color=cur_color, lw=2.2, ls='--', label='Current state')
    sed_fill = [None]

    ax_sed.set_xscale('log')
    ax_sed.set_xlim(WL_LO, WL_HI)
    ax_sed.set_ylim(0.0, sed_products['ymax_top'])
    ax_sed.set_ylabel(r'$F_\lambda$ (normalised)')
    ax_sed.legend(loc='center right', framealpha=0.92, fontsize=8)
    ax_sed.grid(True, which='both', alpha=0.12, lw=0.5)
    plt.setp(ax_sed.get_xticklabels(), visible=False)
    ax_sed.tick_params(axis='x', which='both', length=0)

    zero_line = ax_delta.axhline(0, color='k', lw=0.8, alpha=0.5)
    delta_line, = ax_delta.plot([], [], color='k', lw=1.8, zorder=3)
    delta_fill_pos = [None]
    delta_fill_neg = [None]
    circle_artists = {}
    circle_labels = {}
    for filt in ROMAN_ORDER:
        p = ROMAN_FILTERS[filt]
        circle_artists[filt] = ax_delta.scatter([], [], color=p['color'], s=80,
                                                zorder=6, edgecolors='k', lw=0.8)
        circle_labels[filt] = ax_delta.annotate(
            filt, (p['lambda_eff'], 0.0), xytext=(0, 9), textcoords='offset points',
            ha='center', fontsize=7, color=p['color'], visible=False)

    ax_delta.set_xscale('log')
    ax_delta.set_xlim(WL_LO, WL_HI)
    ticks = [5000, 6000, 7000, 10000, 15000, 20000]
    ax_delta.set_xticks(ticks)
    ax_delta.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax_delta.set_ylim(*sed_products['delta_ylim'])
    ax_delta.set_xlabel(r'Wavelength $\lambda$')
    ax_delta.set_ylabel(r'$\Delta F_\lambda$ (current - ref)')
    ax_delta.grid(True, which='both', alpha=0.12, lw=0.5)

    txt_right = ax_delta.text(0.53, 0.22, '', transform=ax_delta.transAxes,
                              ha='left', va='top', fontsize=9,
                              bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.85))

    def init():
        trail_line.set_data([], [])
        current_point.set_data([], [])
        cur_line.set_data([], [])
        delta_line.set_data([], [])
        txt_left.set_text('')
        txt_right.set_text('')
        return [trail_line, current_point, cur_line, delta_line, txt_left, txt_right, ref_line, zero_line]

    def update(frame_no):
        i = frame_idx[frame_no]
        j = np.searchsorted(frame_idx, i, side='right')
        trail_idx = np.array(frame_idx[:j], dtype=int)
        trail_line.set_data(x[trail_idx], y[trail_idx])
        current_point.set_data([x[i]], [y[i]])

        prod = sed_products['frames'][i]
        cur_flux = prod['cur_flux']
        delta = prod['delta']
        cur_line.set_data(wl, cur_flux)
        delta_line.set_data(wl, delta)

        if sed_fill[0] is not None:
            sed_fill[0].remove()
        sed_fill[0] = ax_sed.fill_between(wl, ref_flux, cur_flux, alpha=0.12, color='gray', zorder=2)

        if delta_fill_pos[0] is not None:
            delta_fill_pos[0].remove()
        if delta_fill_neg[0] is not None:
            delta_fill_neg[0].remove()
        delta_fill_pos[0] = ax_delta.fill_between(wl, 0, delta, where=(delta > 0),
                                                  alpha=0.20, color=cur_color, zorder=2)
        delta_fill_neg[0] = ax_delta.fill_between(wl, 0, delta,
                                                  where=(delta < 0) & np.isfinite(delta),
                                                  alpha=0.20, color=ref_color, zorder=2)

        for filt in ROMAN_ORDER:
            val = prod['circles'][filt]
            if np.isfinite(val):
                circle_artists[filt].set_offsets(np.array([[ROMAN_FILTERS[filt]['lambda_eff'], val]]))
                circle_labels[filt].xy = (ROMAN_FILTERS[filt]['lambda_eff'], val)
                circle_labels[filt].set_y(val)
                circle_labels[filt].xyann = (0, 9 if val >= 0 else -14)
                circle_labels[filt].set_visible(True)
            else:
                circle_artists[filt].set_offsets(np.empty((0, 2)))
                circle_labels[filt].set_visible(False)

        if dataset['name'] == 'radius':
            txt_left.set_text(
                f'Model {dataset["model"][i]}\n'
                f'Phase = {dataset["phase"][i]:.3f}\n'
                f'R = {dataset["y"][i]:.4f} $R_\\odot$\n'
                f'Colour = {dataset["x"][i]:.4f} mag\n'
                f'$T_{{\\rm eff}}$ = {dataset["teff"][i]:.0f} K')
        else:
            txt_left.set_text(
                f'Model {dataset["model"][i]}\n'
                f'Phase = {dataset["phase"][i]:.3f}\n'
                f'$T_{{\\rm eff}}$ = {dataset["y"][i]:.0f} K\n'
                f'Colour = {dataset["x"][i]:.4f} mag\n'
                f'R = {dataset["radius"][i]:.4f} $R_\\odot$')

        txt_right.set_text(
            f'Reference = {dataset["ref_label"]} (model {dataset["model"][ref_idx]})\n'
            f'Current = model {dataset["model"][i]}\n'
            f'Age = {dataset["time"][i]:.4f} d')
        return [trail_line, current_point, cur_line, delta_line, txt_left, txt_right, ref_line, zero_line]

    anim = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=len(frame_idx), interval=1000 / fps,
                                   blit=False)
    out = save_animation(anim, output_dir, dataset['movie_stem'], fps)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(
        description='Animated Roman WFI loop + SED diagnostics for MESA Colors RSP RR Lyrae')
    parser.add_argument('history_file', nargs='?', default=None)
    parser.add_argument('--logs_dir', default=str(DEFAULT_LOGS))
    parser.add_argument('--sed_dir', default=None,
                        help='Directory containing {FILTER}_SED_{MODEL}.csv files '
                             '(default: ../SED relative to this script)')
    parser.add_argument('--output_dir', default=str(DEFAULT_OUT))
    parser.add_argument('--min_periods', type=float, default=4.0,
                        help='Settled-cycle threshold (default 4)')
    parser.add_argument('--stride', type=int, default=5,
                        help='Use every Nth settled point as a frame (default 5)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Optional hard cap on frame count after striding')
    parser.add_argument('--fps', type=int, default=24,
                        help='Movie frame rate (default 24)')
    parser.add_argument('--radius_ref', choices=['max', 'min'], default='max',
                        help='Fixed radius reference point for the radius movie')
    parser.add_argument('--teff_ref', choices=['hot', 'cold'], default='hot',
                        help='Fixed temperature reference point for the temperature movie')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_file = Path(args.history_file) if args.history_file else Path(args.logs_dir) / 'history.data'
    if not history_file.exists():
        print(f'ERROR: {history_file} not found')
        sys.exit(1)

    sed_dir = Path(args.sed_dir) if args.sed_dir else Path(args.logs_dir).parent / 'SED'
    if not sed_dir.exists():
        print(f'ERROR: SED directory not found: {sed_dir}')
        sys.exit(1)

    print(f'Loading {history_file} ...')
    data = load_mesa_history(history_file)
    print(f'Loaded {len(data)} models')

    mag_cols = find_roman_columns(data)
    if not mag_cols:
        print('ERROR: no Roman WFI columns found in history.data')
        sys.exit(1)
    print('Roman WFI columns found:')
    for k, v in mag_cols.items():
        print(f'  {k} -> {v}')

    print('Building radius movie dataset ...')
    radius_ds = build_radius_dataset(data, sed_dir, min_periods=args.min_periods,
                                     stride=args.stride, max_frames=args.max_frames,
                                     reference=args.radius_ref)
    print(f'  {len(radius_ds["frame_idx"])} frames, reference = {radius_ds["ref_label"]}, model {radius_ds["model"][radius_ds["ref_idx"]]}')

    print('Building temperature movie dataset ...')
    teff_ds = build_teff_dataset(data, sed_dir, min_periods=args.min_periods,
                                 stride=args.stride, max_frames=args.max_frames,
                                 reference=args.teff_ref)
    print(f'  {len(teff_ds["frame_idx"])} frames, reference = {teff_ds["ref_label"]}, model {teff_ds["model"][teff_ds["ref_idx"]]}')

    out1 = make_movie(radius_ds, output_dir, fps=args.fps)
    out2 = make_movie(teff_ds, output_dir, fps=args.fps)

    print('\nDone.')
    print(f'  {out1}')
    print(f'  {out2}')


if __name__ == '__main__':
    main()
