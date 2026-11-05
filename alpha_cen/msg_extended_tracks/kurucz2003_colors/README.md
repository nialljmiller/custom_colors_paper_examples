# MSG comparison using the MESA Colors Kurucz atmosphere grid

This diagnostic MSG grid was built from the 3,808 Kurucz2003all flux spectra
shipped with the installed MESA Colors package. It uses the same atmosphere
SED data as the Colors calculations and covers all Alpha Cen A/B continuation
models. The input spectra are flux-only, so MSG represents them with its
constant angular-intensity law; this is suitable for comparing disk-integrated
photometry, but it does not reproduce angle-dependent limb darkening.

Build and compare from the repository root with:

```sh
MSG_DIR=$HOME/MESA/msg $HOME/MESA/msg/.venv/bin/python msg_extended_tracks/build_kurucz_msg_grid.py
MSG_DIR=$HOME/MESA/msg $HOME/MESA/msg/.venv/bin/python msg_extended_tracks/compare_kurucz_msg.py
```

The comparison deliberately keeps the same MSG Johnson B/V passbands and Vega
zero points as the CAP18 validation. Therefore the remaining small offset
includes passband integration, interpolation, and zero-point differences in
addition to any numerical differences between the two interpolation paths.
The original CAP18 comparison files remain alongside the continuation runs.
