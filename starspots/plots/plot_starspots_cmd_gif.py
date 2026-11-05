#!/usr/bin/env python3
"""
Make a GIF of the starspots CMD evolving one track at a time.

This is a standalone script derived from the user's existing starspots plotting
workflow, but it only does one thing: create a cumulative CMD GIF in which each
track is drawn progressively, one model at a time.

Usage:
    python plot_starspots_cmd_gif.py
    python plot_starspots_cmd_gif.py --data-dir /path/to/starspots
    python plot_starspots_cmd_gif.py --fps 12 --points-per-track 40
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import numpy as np


# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR


# ── Plot style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.7',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': False,
    'ytick.right': False,
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


# ── Grid parameters ────────────────────────────────────────────────────
MASSES = [0.3, 0.6, 0.7, 0.9, 1.1]
FSPOTS = [0.2, 0.4, 0.6, 0.8]
XSPOT = 0.85

MASS_CMAP = plt.cm.viridis(np.linspace(0.1, 0.9, len(MASSES)))
MASS_COLORS = {m: c for m, c in zip(MASSES, MASS_CMAP)}

FSPOT_LINESTYLES = {
    0.2: '-',
    0.4: '--',
    0.6: '-.',
    0.8: (0, (5, 2)),
}


# ── Data I/O ───────────────────────────────────────────────────────────
def read_history(filepath: str | Path) -> np.ndarray:
    """Read a MESA history.data file without mesa_reader."""
    filepath = str(filepath)
    with open(filepath, 'r', encoding='utf-8') as handle:
        for _ in range(5):
            handle.readline()
        header = handle.readline().split()
    return np.genfromtxt(filepath, skip_header=6, names=header)


def discover_models(data_dir: str | Path) -> list[tuple[float, float, str]]:
    """Find LOGS_M*_f* directories and parse mass/fspot from their names."""
    data_dir = str(data_dir)
    log_dirs = sorted(glob.glob(os.path.join(data_dir, 'LOGS_M*_f*')))
    models: list[tuple[float, float, str]] = []

    for logdir in log_dirs:
        dirname = os.path.basename(logdir)
        try:
            parts = dirname.replace('LOGS_M', '').replace('_f', ' ').split()
            mass = float(parts[0])
            fspot = float(parts[1])
        except (IndexError, ValueError):
            print(f'Warning: could not parse {dirname}, skipping')
            continue

        history_file = os.path.join(logdir, 'history.data')
        if os.path.exists(history_file):
            models.append((mass, fspot, history_file))
        else:
            print(f'Warning: no history.data in {dirname}')

    return models


def load_model(history_path: str | Path, ms_only: bool = True) -> dict[str, np.ndarray] | None:
    """Load the CMD columns for one model."""
    try:
        data = read_history(history_path)
    except Exception as exc:
        print(f'Error reading {history_path}: {exc}')
        return None

    required = ('g', 'r')
    missing = [name for name in required if name not in data.dtype.names]
    if missing:
        print(f'Error: missing columns {missing} in {history_path}')
        return None

    g_mag = data['g']
    r_mag = data['r']

    if ms_only and 'center_h1' in data.dtype.names:
        mask = data['center_h1'] > 0.01
        g_mag = g_mag[mask]
        r_mag = r_mag[mask]

    if len(g_mag) == 0 or len(r_mag) == 0:
        print(f'Warning: no usable CMD points in {history_path}')
        return None

    return {
        'g_mag': g_mag,
        'r_mag': r_mag,
        'g_r': g_mag - r_mag,
    }


# ── Plot helpers ───────────────────────────────────────────────────────
def add_legends(ax: plt.Axes) -> None:
    """Add mass and fspot legends."""
    mass_handles = [
        Line2D([0], [0], color=MASS_COLORS[m], linewidth=2,
               label=rf'${m}\,{{\rm M}}_\odot$')
        for m in MASSES
    ]
    mass_legend = ax.legend(
        handles=mass_handles,
        loc='lower left',
        title='Mass',
        title_fontsize=9,
        fontsize=9,
        framealpha=0.9,
    )
    ax.add_artist(mass_legend)

    fspot_handles = [
        Line2D([0], [0], color='gray', linestyle=FSPOT_LINESTYLES[f], linewidth=2,
               label=rf'$f_{{\rm spot}} = {f}$')
        for f in FSPOTS
    ]
    ax.legend(
        handles=fspot_handles,
        loc='upper right',
        title=r'$f_{\rm spot}$',
        title_fontsize=9,
        fontsize=9,
        framealpha=0.9,
    )


def add_xspot_annotation(ax: plt.Axes) -> None:
    """Add fixed xspot annotation."""
    ax.text(
        0.13,
        0.96,
        rf'$x_{{\rm spot}} = {XSPOT}$',
        transform=ax.transAxes,
        fontsize=9,
        va='top',
        ha='left',
        bbox=dict(
            boxstyle='round,pad=0.35',
            facecolor='honeydew',
            edgecolor='0.6',
            linewidth=0.8,
            alpha=0.9,
        ),
    )


# ── GIF creation ───────────────────────────────────────────────────────
def make_cmd_track_gif(
    models_data: list[tuple[tuple[float, float, str], dict[str, np.ndarray] | None]],
    output_name: str = 'starspots_cmd_track_build.gif',
    points_per_track: int = 40,
    hold_frames: int = 8,
    fps: int = 12,
) -> Path:
    """Create a cumulative CMD GIF with one track building at a time."""
    valid_models = [item for item in models_data if item[1] is not None]
    if not valid_models:
        raise RuntimeError('No valid models found to animate.')

    all_gr = np.concatenate([model['g_r'] for _, model in valid_models])
    all_g = np.concatenate([model['g_mag'] for _, model in valid_models])

    xspan = all_gr.max() - all_gr.min()
    yspan = all_g.max() - all_g.min()
    xpad = 0.05 * xspan if xspan > 0 else 0.1
    ypad = 0.05 * yspan if yspan > 0 else 0.1

    xmin = all_gr.min() - xpad
    xmax = all_gr.max() + xpad
    ymin = all_g.min() - ypad
    ymax = all_g.max() + ypad

    fig, ax = plt.subplots(figsize=(3.5, 4.5))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_xlabel(r'$g - r$ [mag]')
    ax.set_ylabel(r'$M_g$ [mag]')

    add_xspot_annotation(ax)
    add_legends(ax)

    line_artists = []
    for (mass, fspot, _), _model in valid_models:
        line, = ax.plot(
            [], [],
            color=MASS_COLORS.get(mass, 'k'),
            linestyle=FSPOT_LINESTYLES.get(fspot, '-'),
            linewidth=1.5,
            alpha=0.9,
        )
        line_artists.append(line)

    frame_plan: list[tuple[int, int]] = []
    for idx, ((_mass, _fspot, _path), model) in enumerate(valid_models):
        npts = len(model['g_r'])
        draw_counts = np.linspace(1, npts, points_per_track, dtype=int)
        draw_counts = np.unique(draw_counts)

        for n in draw_counts:
            frame_plan.append((idx, int(n)))
        for _ in range(hold_frames):
            frame_plan.append((idx, npts))

    def init():
        for line in line_artists:
            line.set_data([], [])
        return line_artists

    def update(frame_idx: int):
        active_track, active_n = frame_plan[frame_idx]

        for j, (((_mass, _fspot, _path), model), line) in enumerate(zip(valid_models, line_artists)):
            xvals = model['g_r']
            yvals = model['g_mag']

            if j < active_track:
                line.set_data(xvals, yvals)
            elif j == active_track:
                line.set_data(xvals[:active_n], yvals[:active_n])
            else:
                line.set_data([], [])

        return line_artists

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_plan),
        init_func=init,
        blit=True,
        interval=1000 / fps,
    )

    output_path = OUTPUT_DIR / output_name
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Create a GIF of the starspots CMD evolving one track at a time.'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='.',
        help='Directory containing LOGS_M*_f* folders',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='starspots_cmd_track_build.gif',
        help='Output GIF filename',
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=12,
        help='Frames per second for the GIF',
    )
    parser.add_argument(
        '--points-per-track',
        type=int,
        default=40,
        help='Number of progressive draw steps per track',
    )
    parser.add_argument(
        '--hold-frames',
        type=int,
        default=8,
        help='How many frames to hold after each full track is drawn',
    )
    parser.add_argument(
        '--no-ms-filter',
        action='store_true',
        help='Include post-main-sequence evolution',
    )
    args = parser.parse_args()

    model_info = discover_models(args.data_dir)
    if not model_info:
        raise SystemExit(f'No LOGS_M*_f* directories found in: {args.data_dir}')

    ms_only = not args.no_ms_filter
    models_data = []
    for mass, fspot, path in model_info:
        model = load_model(path, ms_only=ms_only)
        models_data.append(((mass, fspot, path), model))
        if model is not None:
            print(f'Loaded M={mass}, fspot={fspot} ({len(model["g_r"])} points)')

    output_path = make_cmd_track_gif(
        models_data=models_data,
        output_name=args.output,
        points_per_track=args.points_per_track,
        hold_frames=args.hold_frames,
        fps=args.fps,
    )
    print(f'Saved: {output_path}')


if __name__ == '__main__':
    main()
