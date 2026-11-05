# Custom Colors paper examples

Code, MESA inlists and analysis scripts behind the demonstrations in

> Miller, Joyce & Mocz, *Custom Colors: A Module for Computing Synthetic Photometry
> On-the-Fly in Stellar Evolution Calculations, Integrated with MESA and Usable
> with Other Stellar Evolution Codes* (ApJS, submitted).

This repository holds the code only. The Zenodo archive of the same name adds
the input data (flux grids, filters, observational samples, CALSPEC spectra) and
the output data (MESA `history.data` files, result tables, and SED dumps for
selected models). Unpack it over this repository, or use it on its own; the
directory layout is identical.

Related software:

* **MESA Custom Colors**, the `colors` module distributed with MESA (release r26.04.1 and later).
* **SED_Tools** (https://github.com/nialljmiller/SED_Tools), which downloads and preprocesses flux grids and filters.
* **SED_Model** (https://github.com/nialljmiller/SED_Model), which computes stand-alone synthetic photometry with the same Fortran engine.

## Layout

| Folder | Paper section | Contents |
|---|---|---|
| `extinction/` | Sec. 3.6, Fig. 1 | Extinction coefficients from SED_Model at grid nodes |
| `alpha_cen/` | Sec. 4.1, Table 3, Figs. 2, 18-21, Table 4, App. G-H | `astero` work directories for alpha Cen A and B, archived best runs, MSG comparison |
| `tp_agb/` | Sec. 4.2, Figs. 3-4 | TP-AGB evolution with UBVRIJM photometry |
| `rr_lyrae/` | Sec. 4.3, Figs. 5-8 | RSP RR Lyrae model with Roman WFI photometry |
| `rr_lyrae/FN_Lyr/` | Sec. 4.3, App. E, Figs. 9, 16 | FN Lyr Kepler fit, grid searches and optimizer |
| `starspots/` | Sec. 4.4, Fig. 10 | SPOTS grid (5 masses x 4 filling factors), LSST ugrizy |
| `white_dwarfs/` | Sec. 4.5, App. F, Figs. 11, 17 | WD cooling with the Koester grid against the Gaia DR3 WD sample |
| `blue_loop/` | Sec. 4.6, Figs. 12-13 | Cepheid blue loop, LSST ugrizy |
| `yrec/` | Sec. 4.7, Figs. 14-15 | Two-component spotted SEDs for YREC SPOTS tracks |
| `timing/` | App. J, Table 6, Fig. 23 | Overhead benchmark and figure |
| `calspec/` | App. K, Table 7, Fig. 24 | Synthetic vs published photometry of CALSPEC standards |
| `figures/` | all | The 24 figure PDFs as published |
| `input_data/` | (Zenodo only) | Flux cubes and lookup tables, Vega SED, filter curves |

## Figure and table map

| Item | Script (run from its folder) | Data |
|---|---|---|
| Fig. 1 | `extinction/make_extinction_figure.py` | writes `ext_coeffs_sedmodel.csv` (SED_Model, Kurucz2003all) |
| Fig. 2 | `alpha_cen/plot_alpha_cen_ab_main.py` | `alpha_cen/alpha_cen_{a,b}/archive/alpha_cen_{a,b}_dynamic_best/` |
| Table 3 | `alpha_cen/alpha_cen_{a,b}/archive/*_dynamic_best/` | best-fit `LOGS/history.data` and search records |
| Figs. 3-4 | `tp_agb/plots/plot_tpagb.py` | `tp_agb/LOGS/history.data` |
| Figs. 5, 6, 8 | `rr_lyrae/plots/plot_rsp_print.py` | `rr_lyrae/LOGS_colors/history.data`, `rr_lyrae/SED/` |
| Fig. 7 | `rr_lyrae/plots/plot_rsp_roman.py` | `rr_lyrae/LOGS_colors/history.data` |
| Fig. 9 | `rr_lyrae/FN_Lyr/plot_fn_lyr_fit.py` | `optimize_best/history.data`, `FN_Lyr_KIC6936115_kepler_binned.csv` |
| Fig. 10 | `starspots/plots/plot_starspots.py` | `starspots/LOGS_M*_f*/history.data` |
| Fig. 11 | `white_dwarfs/GAIA_obs/plot_gaia_cmd.py` | `GAIA_obs/LOGS_wd_cool/`, Gaia WD sample files in `GAIA_obs/` |
| Figs. 12-13 | annotated plotting script not retained; `blue_loop/plots/plot_blue_loop.py` makes the equivalent unannotated CMD and light-curve plots | `blue_loop/LOGS/history.data`, `blue_loop/SED/` |
| Figs. 14-15 | `yrec/yrec_sed_interface.py`, then `yrec/plot_yrec.py` | `yrec/tracks/` (input), `yrec/m*_f*/` (output) |
| Fig. 16 | `rr_lyrae/FN_Lyr/survey_fn_lyr.py` | FN Lyr grid summaries, `fn_lyr_campaign_scores.csv` |
| Fig. 17 | `white_dwarfs/GAIA_obs/quantify_wd_locus_offset.py` | as Fig. 11; `wd_locus_offsets_summary.txt` |
| Figs. 18-19 | `alpha_cen/alpha_cen_a/plot_alpha_cen_a_fit.py` | alpha Cen A archived search |
| Fig. 20 | `alpha_cen/plot_alpha_cen_runtime_feedback.py` | archived dynamic-best histories |
| Fig. 21, Table 4 | `alpha_cen/plot_alpha_cen_msg_comparison.py`; `alpha_cen/msg_extended_tracks/compare_*.py` | `*_colors_msg_comparison.csv`, `msg_extended_tracks/` |
| Fig. 22, Table 5 | not included; the cross-validation inputs and script were not retained | |
| Fig. 23, Table 6 | `timing/test_timings.py` (runs MESA); `timing/plot_timing_feature_breakdown.py` | values in the plotting script |
| Fig. 24, Table 7 | `calspec/synphot_pipeline.py`, then `calspec/make_figure_and_table.py` | `calspec/spectra/` (Zenodo), `published_photometry.csv` |

## Running the MESA examples

Each demonstration folder is a standard MESA work directory: `./mk`, then `./rn`.
The `&colors` namelist in each inlist sets the flux grid (`stellar_atm`), the
filter set (`instrument`), the magnitude system and the distance. Paths that
begin with `data/colors_data/` are resolved relative to `$MESA_DIR`.

Some inlists and scripts point to grids, filters or run folders by absolute paths
on the authors' machine (for example `/home/njm/SED_Tools/data/stellar_models/...`).
The grids and filters they need are in `input_data/` in the Zenodo archive, and
SED_Tools can regenerate them. Edit `stellar_atm` and `instrument`, or the path
constants at the top of a script, to point at your copies.

The runs were made with MESA development builds from 2026 that contain the
Custom Colors module. A different MESA version can give slightly different
numbers.

## Python requirements

Python 3.10 or later, with `numpy`, `scipy`, `pandas`, `matplotlib`, `astropy`,
`h5py`, `mesa_reader` and `sed-model`. The MSG comparison (App. H) needs `pymsg`.
The Gaia, Kepler and CALSPEC download scripts need `astroquery` and `lightkurve`.

## Notes on the Zenodo data

* SED dumps are included only for selected models:
  * RR Lyrae: F062 files for the last four pulsation periods (models 201289-204185);
  * TP-AGB: V-band files for every model;
  * white-dwarf cooling: all files (`white_dwarfs/GAIA_obs/SED/`);
  * blue loop: all files;
  * YREC: model `m110_f034`, shown in Fig. 14.

  Rerunning a work directory with `sed_per_model = .true.` regenerates the rest.
* For the FN Lyr grid searches, only the per-model `summary.txt` files are kept. The best and optimizer runs keep their full histories.
* Of the flux grids, only `flux_cube.bin` and `lookup_table.csv` are included, which is what MESA Custom Colors reads.

## License

MIT; see `LICENSE`.
