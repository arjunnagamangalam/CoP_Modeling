# Dual-Pathway Postural Sway Model

A physics-based dual-pathway stochastic oscillator model of center-of-pressure (CoP) postural sway. Two independent, additively-summed damped stochastic oscillators (a fast and a slow pathway, one pair per ML/AP axis) are calibrated per vision/feedback configuration via Simulated Method of Moments (SMM), then validated against held-out real trials using a 20-metric ensemble-level goodness-of-fit analysis.

## Repository structure

\`\`\`
final_model/
├── data_loading/
│   ├── load_raw_sessions.py      # Loads and cleans raw CoP recordings (10 Hz zero-phase Butterworth filter)
│   ├── find_outlier_trials.py    # Flags/removes outlier trials prior to calibration
│   └── verify_parquet.py         # Sanity-checks the cleaned parquet output
├── scripts/
│   ├── dual_pathway_matrix_system.py    # Dual-pathway SDE system definition (state matrices)
│   ├── kalman_discretize.py             # Discretizes the continuous-time SDE for simulation
│   ├── dual_pathway_smm_calibration.py  # SMM calibration: fits per-condition parameters to target statistics
│   ├── model_tools.py                   # Shared utilities: batch simulation, train/test split, config grouping
│   └── run_dual_pathway_all_configs.py  # Loads cleaned data per (vision, feedback) config; CONFIGS registry
├── goodness_of_fit/
│   ├── goodness_of_fit_dual_pathway.py  # Main GOF driver: ensemble comparison, composite score, KS tests
│   ├── goodness_of_fit_helpers.py       # Metric computation and scoring helpers
│   ├── plot_rms_variability.py          # RMS dispersion (real vs. simulated) comparison plots
│   └── rms_low_high_freq.py             # Low-/high-frequency RMS band-split analysis and plots
└── README.md
\`\`\`

Data (`cleaned_balance_all.parquet`) lives one level up, at `../data/processed/`, and is not tracked in this repo.

## Setup

```bash
conda activate cop_env
pip install -r requirements.txt
```

## Pipeline

The intended order of operations:

1. **Clean raw data** — `data_loading/load_raw_sessions.py` rebuilds `cleaned_balance_all.parquet` from the raw per-trial force-plate CSVs (`{subject}_{vision}_{feedback}_{trial}.csv`, motion-capture-suite ASCII export). Per trial it: auto-detects the active force plate by mean |Fz| (falling back to the dominant plate if both read a real load), converts the vendor-computed CoP (Cx/Cy, mm) to meters, applies a 10 Hz 4th-order zero-phase Butterworth low-pass filter (an unconfirmed-cutoff default — see the script's docstring), and subtracts each trial's own first sample so every cleaned trial starts at exactly 0. Calibration-trial files (`{subject} Cal {n}.csv`) are detected and excluded, not loaded.

```bash
   python3 data_loading/load_raw_sessions.py /path/to/raw/session/files [output.parquet] [cutoff_hz]
```
2. **Verify** — `data_loading/verify_parquet.py` runs 6 sanity checks against the freshly built `cleaned_balance_all.parquet`: schema (expected columns/dtypes), missing/non-finite values, trial/sample-count structure (subjects, trials per condition, ragged trials, sampling rates present), the per-trial baseline-subtraction property (every trial's first sample should be exactly 0), RMS/range magnitude plausibility (flags trials outside a ~0.01–10cm RMS range, which would indicate a unit/scaling error), and a directional sanity check against known postural-control tendencies (eyes-closed sway > eyes-open; auditory feedback modestly reducing sway, weaker effect).

```bash
   python3 data_loading/verify_parquet.py ../data/processed/cleaned_balance_all.parquet
```
3. **Investigate outliers** — `data_loading/find_outlier_trials.py` is a follow-up diagnostic to `verify_parquet.py` (triggered by anything section 5/6 above flags): it computes per-trial RMS/range for ML and AP, prints the worst trials by AP sway plus a mean-vs-median Close-vs-Open comparison (median is robust to a handful of outliers), and saves trajectory plots of the worst few trials as PNGs for manual review — it does not itself drop or exclude any trials.

```bash
   python3 data_loading/find_outlier_trials.py ../data/processed/cleaned_balance_all.parquet [n_plots]
```
4. **Calibrate** — `scripts/dual_pathway_smm_calibration.py` exposes `calibrate_dual_pathway_smm(train_df, test_df, fs, ...)`, which fits the dual-pathway model's 12 parameters (per axis: `omega_slow_sq`, `gap` [fast-slow frequency separation], `gamma_fast`, `gamma_slow`, `D_fast`, `D_slow`) to a training split's target statistics via Nelder–Mead SMM, with identifiability/stability guards (positivity, minimum damping ratios, minimum slow-pathway variance share) enforced during the search.

   `scripts/run_dual_pathway_all_configs.py` is the actual driver: it runs `N_RESTARTS` independent, randomly-initialized calibrations per configuration (16 tasks total across the 4 vision × feedback configs), in parallel via `ProcessPoolExecutor`, then per config picks the best non-ringing restart by loss (falling back to least-ringing, clearly flagged, if every restart for a config rings). Three startup self-checks (`_verify_floor_is_enforced`, `_verify_fast_floor_is_enforced`, `_verify_hf_threshold_is_enforced`) confirm the deployed module-level constraints are actually being enforced before committing to a full run.

```bash
   python scripts/run_dual_pathway_all_configs.py
```

   For each config, saves a validation-table CSV and calibrated-params JSON to `results/metrics/<hf_threshold_tag>/` (e.g. `hf0p3hz` for `HF_THRESHOLD_HZ=0.3`) and generates predicted-vs-real / PSD comparison plots to `results/figures/<hf_threshold_tag>/`.

5. **Evaluate goodness-of-fit** — `goodness_of_fit/goodness_of_fit_dual_pathway.py` loads a config's calibrated params and its held-out test split (the same split `run_dual_pathway_all_configs.py` validated against), simulates a fresh batch of trials with fresh, disjoint seeds (`--n-sim-seeds`, default 30; the paper's N=60 figure comes from an explicit override of this default), optionally with between-subject variance injection (see Methodology below), and compares them against the real held-out trials across 20 summary statistics via `evaluate_goodness_of_fit`, reporting composite relative error + KS significance per metric. Runs all 4 configs by default, or a single one via `--vision`/`--feedback`.

```bash
   python goodness_of_fit/goodness_of_fit_dual_pathway.py \
       --params-dir results/metrics/hf0p3hz \
       --out-dir goodness_of_fit/gof_output_all_configs \
       --n-sim-seeds 60 \
       --exclude-from-composite ml_skew ap_skew ml_kurtosis ap_kurtosis \
       --subject-effect-baseline-gof goodness_of_fit/gof_output_baseline \
       --trials-per-virtual-subject 3
```

   Other flags: `--seed-offset` (start of the simulated-trial seed range, default 100, disjoint from calibration/validation seeds), `--duration-seconds` (default 20.0), `--filter-cutoff-hz` (Butterworth low-pass applied to simulated trials to match the filtering already baked into the real parquet, default 10.0), and `--use-bootstrap` (a superseded, OOM-prone alternative to `--subject-effect-*` that draws each trial's params from a separate bootstrap-recalibration replicate instead of injecting variance into one fixed params dict — see Methodology).

6. **Plot supplementary comparisons** — `goodness_of_fit/plot_rms_variability.py` reads real/simulated RMS mean+std back out of each condition's `gof_summary.csv` (from step 5) and plots the across-trial RMS SD, real vs. simulated, per axis and condition, annotated with the real/sim ratio — the diagnostic for how much the simulated ensemble under-disperses relative to real subjects (and how much subject-effect injection closes that gap). `goodness_of_fit/rms_low_high_freq.py` is the companion for the low-/high-frequency RMS split (`ml_rms_lf_cm`/`ml_rms_hf_cm`/`ap_rms_lf_cm`/`ap_rms_hf_cm`, from the same `gof_summary.csv` files): a 2×2 grid (ML/AP × low/high band) of real-vs-simulated bar charts with `rel_error_pct` noted per bar pair. Both scripts are referenced in the paper's GOF section.

```bash
   python goodness_of_fit/plot_rms_variability.py --gof-dir goodness_of_fit/gof_output_all_configs --out rms_variability.png
   python goodness_of_fit/rms_low_high_freq.py --gof-dir goodness_of_fit/gof_output_all_configs --out rms_lowhigh_freq.png --cutoff-hz 0.3
```

## Methodology summary

Because the model is stochastic, fit is assessed at the ensemble level (distributions of summary statistics over trials), not by point-wise comparison against any single real trajectory. For each of the 4 configurations:

- 20 metrics per condition (10 per axis): RMS, low-/high-frequency RMS, peak-to-peak, mean velocity, median power frequency (F50), skewness, kurtosis, and short-/long-scale DFA exponents.
- N=60 simulated trials (disjoint seeds from calibration) are compared against held-out real trials using relative error of means and a two-sample KS test.
- A composite score (mean relative error across the 16 non-zero-weighted metrics; skewness/kurtosis excluded as unstable near-zero statistics) summarizes fit per condition — composite error ranges 5.64–10.27% across conditions.
- Simulated trials can incorporate a between-subject variance-injection step (`--subject-effect-*` flags): consecutive groups of `--trials-per-virtual-subject` simulated trials ("virtual subjects") share one lognormal noise-scale factor `c = exp(z)`, `z ~ N(-σz²/4, σz²)`, applied multiplicatively to both pathways' D parameters for that axis. This scales RMS by exactly `sqrt(c)` while leaving damping ratios, PSD shape, and DFA exponents untouched (D doesn't enter the damping ratios), and the `-σz²/4` mean-shift keeps E[RMS] unchanged, so it only adds dispersion. The target coefficient of variation can be set manually (`--subject-effect-cv-ml`/`--subject-effect-cv-ap`) or derived in closed form from a prior baseline GOF run's real-vs-simulated RMS variance gap (`--subject-effect-baseline-gof`). A superseded alternative (`--use-bootstrap`) instead draws each simulated trial's parameters from a separate bootstrap-recalibration replicate, but was dropped for being OOM-prone on Oscar.

The model's one consistent, honestly-reported limitation: the short-timescale AP DFA exponent is KS-significant in all four conditions, consistent with the fast AP pathway being mildly under-damped relative to the real system.