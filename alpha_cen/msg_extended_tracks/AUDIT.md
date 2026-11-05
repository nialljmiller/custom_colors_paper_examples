# Extended Custom Colors / MSG validation continuations

Working root: `/home/njm/MESA/alpha_cen_colors`.
MESA installation: `/home/njm/MESA/mesa_filter_interp_leak_fix`.
Archived model version identifier: `06842d5a-dirty`.

## Copied files

For each component, from its protected `archive/alpha_cen_*_dynamic_best`:
`inlist`, `inlist_example_astero`, `history_columns.list`,
`profile_columns.list`, `run.f90` and `run_star_extras.f90` (into `src/`),
and `final.mod` (renamed `start.mod`). The Makefile came from the live A
work directory and includes the installed MESA work Makefile.
No live fit executable, fitting output, or LOGS was reused.
`archive_sha256_before.json` records hashes of every protected archived file.

## Continuation controls

Load `start.mod`; do not create a pre-main-sequence model, change its
network, reset cumulative energy error, or relax its mass/composition.
Save the terminating model as local `final.mod`.
Fixed `mixing_length_alpha`: A = 1.8982421875, B = 2.041796875.
Mass: A = 1.0788, B = 0.9092 solar masses.
Initial Y = 0.26 and initial [Fe/H] = +0.24 are inherited in the
archived evolved abundance profiles. The model-file `initial_z` header
is not a substitute for those actual profiles.
Element diffusion is explicitly disabled; overshoot remains disabled.

Ordinary `do_run_star` replaces `run_star_astero`. The new extras contain
only photometry-input diagnostics and a photospheric gravity guard.
There are no astero callbacks, chi2 evaluations, chi2 timestep caps,
simplex searches, or fit termination rules.

Retain fitted gs98/lowT_fa05_gs98 opacities, opacity blending,
photosphere atmosphere table, TDC convection, mesh_delta_coeff = 0.5,
dedt energy equation, gold2 tolerances, nuclear network and special rate.
Retain max_years_for_timestep = 1e8; explicitly use ordinary
varcontrol_target = 1e-3. Set max_model_number = -1, max_age = 30e9.
Keep history_interval = 1 and preserve the complete local LOGS directory.
After a laptop interruption, photo_interval was changed from 1000000
to the installed ordinary default of 50 for restart resilience.
The interrupted A run reached model 8407 at 12.588543219 Gyr without
a restart photo. Its full LOGS and run.log are retained under
`alpha_cen_a/interrupted_attempt_1/`; A is repeated from unchanged start.mod.
Set `warn_run_star_extras = .false.` so unused optional hooks can use
MESA's default no-op routines without the example driver aborting.

Retain Kurucz2003all atmospheres, Generic/Johnson filters, Vega SED,
Vega magnitude system, and Colors distance = 3.0857e19 cm.
B and V are automatically appended to history by Colors.
Extras record exact photospheric radius, gravity, atmosphere metallicity,
and alpha_MLT on every history row. Standard surface radius/gravity are
also retained for diagnostics.

## Stopping conditions verified from the installed source

`star/defaults/controls.defaults` defines `Teff_lower_limit`,
`log_g_lower_limit`, `stop_at_phase_TP_AGB`, and `max_age`.
`star/private/do_one_utils.f90` implements the temperature and gravity
comparisons with <=. Native gravity is outermost-cell surface gravity.
Use Teff <= 3600 K OR surface log_g <= 0.1; extras additionally stop
at photospheric log_g <= 0.1, matching the Colors/MSG input.
Stop at entry into TP-AGB or at 30 Gyr if reached first.

CAP18 grid axes inspected through pymsg: Teff 3500--30000 K,
[Fe/H] -5--+0.5, log(g) 0--5. Every continuation history model within
these bounds is evaluated; any excluded model IDs are reported.
MSG uses each model's atmosphere metallicity, including changes from
evolutionary dredge-up. Initial [Fe/H] remains fixed.
MSG distance uses the archived comparison scripts' 10 parsec convention
(3.085677581491367e19 cm); the tiny difference from the fitted Colors
distance is retained and documented.

## Commands

From the working root, in order:

```bash
bash msg_extended_tracks/run_continuation.sh a
MSG_DIR=$HOME/MESA/msg $HOME/MESA/msg/.venv/bin/python msg_extended_tracks/compare_extended_msg.py a
bash msg_extended_tracks/run_continuation.sh b
MSG_DIR=$HOME/MESA/msg $HOME/MESA/msg/.venv/bin/python msg_extended_tracks/compare_extended_msg.py b
```

The wrapper sets four OpenMP threads and local MESA cache directories,
runs `make > build.log 2>&1`, and runs `/usr/bin/time -p ./build/bin/star`
with UTC start/end timestamps and all output in `run.log`.
It refuses to overwrite an existing run.log.

A required two initialization corrections before evolution began:
the installed diagnostic column is `co_core_mass`, and unused extras
hooks require disabling the driver warning. Both zero-step attempt
logs are preserved as `run_initialization_attempt.log` and
`run_optional_hooks_attempt.log`; `run.log` captures the actual run.

Each directory receives an exported Colors track CSV, the requested
Colors/MSG comparison CSV, and `validation_summary.json` with final
state, termination reason, accepted history count, exclusions, and
delta median/RMS/minimum/maximum. Delta always means MSG minus Colors.
