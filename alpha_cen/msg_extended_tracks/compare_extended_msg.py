"""Export accepted MESA continuation models and compare their exact Colors inputs."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pymsg

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('component', choices=['a', 'b'])
args = parser.parse_args()
name = f'alpha_cen_{args.component}'
work = ROOT / name
history = np.atleast_1d(np.genfromtxt(work / 'LOGS/history.data', skip_header=5, names=True))
required = ['model_number', 'star_age', 'effective_T', 'colors_log_g',
            'colors_radius_cm', 'metallicity_dex', 'fixed_alpha_MLT', 'B', 'V']
assert all(key in history.dtype.names for key in required), history.dtype.names
# MESA histories can contain superseded rows after a retry/restart: keep the
# final occurrence, in chronological model order.
_, reverse_indices = np.unique(history['model_number'][::-1], return_index=True)
history = history[np.sort(len(history) - 1 - reverse_indices)]
assert np.all(np.diff(history['model_number']) == 1)
assert np.all(np.diff(history['star_age']) > 0)
alpha = {'a': 1.8982421875, 'b': 2.041796875}[args.component]
assert np.allclose(history['fixed_alpha_MLT'], alpha, rtol=0, atol=1e-12)
mass = {'a': 1.0788, 'b': 0.9092}[args.component]
assert np.allclose(history['star_mass'], mass, rtol=0, atol=1e-10)
def model_header(path, field):
    text = '\n'.join(path.read_text().splitlines()[:30])
    match = re.search(r'^\s*' + re.escape(field) + r'\s+([\d.EeDd+\-]+)\s*$', text, re.M)
    assert match, (path, field)
    return float(match[1].replace('D', 'E').replace('d', 'e'))
start_age = model_header(work / 'start.mod', 'star_age')
start_model = model_header(work / 'start.mod', 'model_number')
assert history['model_number'][0] == start_model + 1
assert history['star_age'][0] > start_age
track = np.column_stack([history[k] for k in required[:6]] +
                        [history['B'], history['V'], history['B'] - history['V']])
assert np.all(np.isfinite(track))
export = work / f'{name}_extended_colors_track.csv'
np.savetxt(export, track, delimiter=',', comments='', fmt='%.16e',
           header='model_number,star_age_yr,Teff_K,log_g,radius_cm,metallicity_dex,Colors_B,Colors_V,Colors_B_minus_V')
msg = Path(os.environ['MSG_DIR'])
grid = msg / 'data/grids/sg-CAP18-coarse.h5'
phot = [pymsg.PhotGrid(str(grid), str(msg / f'data/passbands/pb-Generic-Johnson.{band}-Vega.h5'))
        for band in ['B', 'V']]
rows = []
skipped = []
distance = 10 * 3.085677581491367e18  # Same parsec convention as archived MSG scripts.
for index, row in enumerate(track):
    x = dict(zip(['Teff', '[Fe/H]', 'log(g)'], row[[2, 5, 3]]))
    if not all(p.axis_x_min[k] <= x[k] <= p.axis_x_max[k] for p in phot for k in x):
        skipped.append(int(row[0]))
        continue
    scale = (row[4] / distance)**2
    b, v = [-2.5 * np.log10(scale * p.flux(x)) for p in phot]
    rows.append([*row, b, v, b-v, b-row[6], v-row[7], (b-v)-row[8]])
    if index == 0 or (index+1) % 250 == 0 or index+1 == len(track):
        print(f'{name}: {index+1}/{len(track)} models', flush=True)
rows = np.asarray(rows)
assert len(rows) + len(skipped) == len(track)
assert np.all(np.isfinite(rows))
output = work / f'{name}_extended_colors_msg.csv'
np.savetxt(output, rows, delimiter=',', comments='', fmt='%.16e',
           header='model_number,star_age_yr,Teff_K,log_g,radius_cm,FeH,Colors_B,Colors_V,Colors_B_minus_V,MSG_B,MSG_V,MSG_B_minus_V,delta_B,delta_V,delta_B_minus_V')
log = (work / 'run.log').read_text()
termination = [line.strip() for line in log.splitlines() if 'termination code' in line.lower()]
assert termination, 'No MESA termination code in run.log'
assert (work / 'final.mod').is_file()
assert any(key in ' '.join(termination) for key in
           ['Teff_lower_limit', 'log_g_lower_limit', 'CAP18 photosphere', 'max_age', 'phase_TP_AGB']), termination
assert model_header(work / 'final.mod', 'model_number') == track[-1, 0]
assert np.isclose(model_header(work / 'final.mod', 'star_age'), track[-1, 1], rtol=1e-12)
summary = {'accepted_history_models': len(history), 'first_model': int(track[0, 0]),
           'starting_age_yr': start_age, 'starting_model': int(start_model),
           'final_model': int(track[-1, 0]), 'final_age_yr': track[-1, 1],
           'final_Teff_K': track[-1, 2], 'final_log_g': track[-1, 3],
           'alpha_MLT': alpha, 'termination': termination, 'MSG_models': len(rows),
           'outside_CAP18_models': skipped, 'statistics': {}}
for label, values in zip(['delta_B', 'delta_V', 'delta_B_minus_V'], rows[:, 12:].T):
    stats = dict(median=float(np.median(values)), rms=float(np.sqrt(np.mean(values**2))),
                 min=float(np.min(values)), max=float(np.max(values)))
    summary['statistics'][label] = stats
    print(label, ', '.join(f'{k}={v:+.6f}' for k, v in stats.items()))
(work / 'validation_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
for path, expected in json.loads((ROOT / 'archive_sha256_before.json').read_text()).items():
    assert hashlib.sha256((ROOT.parent / path).read_bytes()).hexdigest() == expected, path
print(json.dumps(summary, indent=2))
print('Wrote:', export, output, sep='\n')
print('All archived file SHA256 hashes unchanged.')
