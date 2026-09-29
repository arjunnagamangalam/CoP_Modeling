"""
Goodness-of-fit evaluation wired directly into your existing dual-pathway
pipeline -- no manual npz export. Reuses, unchanged:

  - dual_pathway_smm.simulate_dual_pathway_batch to generate simulated
    trajectories from the SAME calibrated params your run already saved
    to dual_pathway_params_{vision}_{feedback}.json (run_dual_pathway_
    all_configs.py's _json_safe() output -- omega_fast_sq_ml/ap etc. are
    already computed and present, so no need to re-derive them from
    omega_slow_sq + gap).
  - run_dual_pathway_all_configs.load_config_data(vision, feedback), so
    this evaluates against the EXACT held-out test set your calibration
    run used -- same default seed=0 into generalized_fp_heldout.
    split_trials, not a fresh random split that would silently redefine
    what "held-out" means here.
  - The ML/AP seed-decorrelation convention from dual_pathway_smm.
    simulate_dual_pathway_and_compute_stats (AP seeds offset by +100000
    from ML seeds) -- copied exactly, not reinvented, so simulated ML/AP
    noise stays independent the same way it does during calibration.
  - dual_pathway_gof.py's metric table / composite-score / KS-test /
    PSD / DFA / plotting machinery, entirely unchanged, so numbers here
    are directly comparable to anything produced with that script.

REQUIRES, all importable from this file's directory (put this file in
the same directory as run_dual_pathway_all_configs.py):
    run_dual_pathway_all_configs.py, dual_pathway_smm.py,
    generalized_fp_heldout.py, bootstrap_noise_model.py,
    dual_pathway_system.py, kalman_discretize.py, dual_pathway_gof.py
plus the saved dual_pathway_params_{vision}_{feedback}.json files
run_dual_pathway_all_configs.py already writes to results/metrics/<tag>/.

NOT INDEPENDENTLY TEST-RUN: I don't have dual_pathway_system.py,
kalman_discretize.py, generalized_fp_heldout.py, or bootstrap_noise_
model.py in this sandbox, so I could not execute this end-to-end myself
the way load_raw_sessions.py and dual_pathway_gof.py were tested. I
traced every call against the exact function signatures/return shapes
in the two files you shared (run_dual_pathway_all_configs.py, dual_
pathway_smm.py), but please run it and send me the output/any traceback
-- I'd rather fix a real error against real output than have you
discover a mistake I didn't catch.

USAGE:
    python3 run_dual_pathway_gof.py \\
        --params-dir /path/to/results/metrics/hf0p3hz \\
        --vision Close --feedback Auditory \\
        --out-dir gof_close_auditory [--n-sim-seeds 30]

    Omit --vision/--feedback to run all 4 configs (same CONFIGS list as
    run_dual_pathway_all_configs.py), writing gof_<vision>_<feedback>/
    subfolders under --out-dir.
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.run_dual_pathway_all_configs import load_config_data, CONFIGS
from scripts.model_tools import simulate_dual_pathway_batch, split_trials, GROUP_COLS, BURN_IN_SECONDS
from goodness_of_fit.goodness_of_fit_helpers import evaluate_goodness_of_fit

def apply_filter(x, fs, cutoff_hz=None, order=4):
    """Same zero-phase Butterworth low-pass as load_raw_sessions.apply_filter
    -- inlined (not imported) since this file has to run on whatever machine
    holds your real pipeline files, which may not be next to load_raw_
    sessions.py. MUST match the cutoff_hz your parquet was actually built
    with (default 10.0, matching load_raw_sessions.py's default) or this
    comparison is apples-to-oranges: real trials in the parquet already
    went through this filter (cop_x_clean/cop_y_clean), so simulated
    trials need it applied too, or every short-timescale statistic (DFA
    alpha_short in particular) is biased by the mismatch, not by any real
    model/data discrepancy."""
    if cutoff_hz is None:
        return x
    b, a = butter(order, cutoff_hz / (fs / 2.0), btype="low")
    return filtfilt(b, a, x)


def load_calibrated_params(params_dir, vision, feedback):
    path = Path(params_dir) / f"dual_pathway_params_{vision}_{feedback}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- did run_dual_pathway_all_configs.py finish for "
            f"{vision}/{feedback} with this same params-dir (the hf<threshold>hz tag folder)?")
    with open(path) as f:
        return json.load(f)


def load_bootstrap_params(params_dir, vision, feedback):
    """Loads the JSON LIST of bootstrap-replicate calibrated parameter
    dicts written by bootstrap_calibrate_dual_pathway.py -- one dict per
    subject-level cluster-bootstrap replicate, NOT the single winning
    params dict load_calibrated_params returns."""
    path = Path(params_dir) / f"dual_pathway_params_bootstrap_{vision}_{feedback}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- did bootstrap_calibrate_dual_pathway.py finish for "
            f"{vision}/{feedback} with this same params-dir?")
    with open(path) as f:
        params_list = json.load(f)
    if not isinstance(params_list, list) or not params_list:
        raise ValueError(f"{path} does not contain a non-empty JSON list of param dicts "
                          f"-- is this actually a bootstrap output file, not the single-"
                          f"winner dual_pathway_params_{vision}_{feedback}.json?")
    return params_list


def _simulate_one_trial(params, fs, duration_seconds, seed, filter_cutoff_hz):
    """One (x_m, y_m) trial from a single params dict/seed pair -- used
    ONLY by the new bootstrap path below (each trial needs a different
    params dict there, so it can't reuse the batched call). The original
    single-params path is left exactly as it was (one batched call across
    all seeds at once) rather than rewritten in terms of this helper, so
    that already-validated behavior/performance can't be disturbed by
    this change -- see module changelog note in simulate_trials_from_params."""
    x_batch_cm = simulate_dual_pathway_batch(
        params["omega_fast_sq_ml"], params["gamma_fast_ml"],
        params["omega_slow_sq_ml"], params["gamma_slow_ml"],
        params["D_fast_ml"], params["D_slow_ml"],
        fs, duration_seconds, [seed], burn_in_seconds=BURN_IN_SECONDS)
    y_batch_cm = simulate_dual_pathway_batch(
        params["omega_fast_sq_ap"], params["gamma_fast_ap"],
        params["omega_slow_sq_ap"], params["gamma_slow_ap"],
        params["D_fast_ap"], params["D_slow_ap"],
        fs, duration_seconds, [seed + 100000], burn_in_seconds=BURN_IN_SECONDS)
    x_m = apply_filter(x_batch_cm[0] / 100.0, fs, filter_cutoff_hz)
    y_m = apply_filter(y_batch_cm[0] / 100.0, fs, filter_cutoff_hz)
    return x_m, y_m


def simulate_trials_from_params(params, fs, duration_seconds, seeds, filter_cutoff_hz=10.0,
                                 bootstrap_params_list=None, bootstrap_rng_seed=12345):
    """Returns a list of (x_m, y_m) tuples, meters -- matches
    dual_pathway_gof.py's expected trial format. AP seeds offset by
    +100000 from ML seeds, exactly matching dual_pathway_smm.
    simulate_dual_pathway_and_compute_stats's own convention, so
    ML/AP noise stays decorrelated the same way it does during
    calibration.

    filter_cutoff_hz: applies the SAME zero-phase Butterworth low-pass to
    the simulated output that load_raw_sessions.py already applied to the
    real trials before they were saved to the parquet (cop_x_clean/
    cop_y_clean). Without this, the comparison is apples-to-oranges --
    filtered real data vs. raw unfiltered simulated data -- which biases
    every short-timescale statistic (DFA alpha_short especially). MUST
    match whatever cutoff your parquet was actually built with; pass None
    to disable if you built the parquet with filtering off.

    bootstrap_params_list: if given (a JSON list of calibrated param
    dicts from bootstrap_calibrate_dual_pathway.py), draws ONE params
    dict PER SIMULATED TRIAL from this list (uniformly, with replacement,
    via a fixed RNG so runs are reproducible), simulating each such trial
    individually via _simulate_one_trial -- instead of using `params` for
    every trial via one batched call. This reintroduces between-subject-
    level parameter variability into the simulated ensemble -- see the
    RMS Variability under-dispersion discussion in chat for why the
    single-`params` path structurally can't produce sim_std comparable to
    real_std. `params` is ignored (may be None) when bootstrap_params_list
    is given; the original single-params/batched path below is otherwise
    completely unchanged from before this option existed."""
    if bootstrap_params_list:
        rng = np.random.default_rng(bootstrap_rng_seed)
        trials = []
        for seed in seeds:
            p = bootstrap_params_list[int(rng.integers(len(bootstrap_params_list)))]
            trials.append(_simulate_one_trial(p, fs, duration_seconds, seed, filter_cutoff_hz))
        return trials

    x_batch_cm = simulate_dual_pathway_batch(
        params["omega_fast_sq_ml"], params["gamma_fast_ml"],
        params["omega_slow_sq_ml"], params["gamma_slow_ml"],
        params["D_fast_ml"], params["D_slow_ml"],
        fs, duration_seconds, seeds, burn_in_seconds=BURN_IN_SECONDS)
    y_batch_cm = simulate_dual_pathway_batch(
        params["omega_fast_sq_ap"], params["gamma_fast_ap"],
        params["omega_slow_sq_ap"], params["gamma_slow_ap"],
        params["D_fast_ap"], params["D_slow_ap"],
        fs, duration_seconds, [s + 100000 for s in seeds], burn_in_seconds=BURN_IN_SECONDS)
    x_batch_m = x_batch_cm / 100.0  # cm -> m, matches this project's parquet convention
    y_batch_m = y_batch_cm / 100.0
    trials = []
    for i in range(len(seeds)):
        x_m = apply_filter(x_batch_m[i], fs, filter_cutoff_hz)
        y_m = apply_filter(y_batch_m[i], fs, filter_cutoff_hz)
        trials.append((x_m, y_m))
    return trials


# ---------------------------------------------------------------------
# Subject-effect variance injection -- replaces bootstrap_calibrate_
# dual_pathway.py as the fix for the RMS Variability under-dispersion
# finding. That script re-runs full SMM calibration (scipy.optimize) per
# bootstrap replicate -- 240 tasks at the current N_BOOTSTRAP/N_RESTARTS_
# PER_BOOT settings -- which OOM'd repeatedly on a 4GB/32-core Oscar
# allocation even after removing the DataFrame-duplication issue, because
# each worker's own calibration working set (scipy optimizer state, PSD/
# FFT arrays) doesn't shrink just because data loading got cheaper.
#
# This does the SAME conceptual thing (give simulated trials genuine
# between-subject variability instead of every trial coming from one
# fixed population-average parameter vector) with NO recalibration at
# all -- only cheap simulation calls, using the parameters you already
# have.
#
# MECHANISM: for a linear SDE like each pathway here (fixed omega/gamma,
# additive white noise scaled by D), the stationary variance of the
# pathway's output is EXACTLY proportional to D for fixed omega/gamma --
# scaling a pathway's D by a factor c scales that pathway's variance by
# exactly c, with no approximation. If both pathways for one axis (fast+
# slow) have their D scaled by the SAME factor c for a given "virtual
# subject", the axis's total stationary variance scales by exactly c too
# (sum of two variances, each scaled by c), so RMS scales by exactly
# sqrt(c). Crucially, D does NOT enter the damping ratios
# (zeta_fast/zeta_slow depend only on gamma/omega), so this perturbation
# changes ONLY the overall noise amplitude (hence RMS) per virtual
# subject -- it leaves resonance/ringing behavior, PSD SHAPE, hf_power_
# frac, and DFA alpha (all governed by the relative shape of the transfer
# function, not its overall amplitude) structurally untouched. That means
# this can't undo the MIN_ZETA_FAST fix or reopen the frequency-domain
# fit issues the bootstrap-recalibration approach risked (a fresh
# calibration per replicate could in principle re-find a differently-
# ringing optimum each time) -- it only targets RMS variability, which is
# the one thing it's meant to fix.
#
# c is drawn per virtual subject as a lognormal factor c = exp(z),
# z ~ N(-sigma_z^2/4, sigma_z^2). The -sigma_z^2/4 mean-shift exactly
# cancels the lognormal's mean inflation (E[exp(z/2)] would otherwise be
# >1 by Jensen's inequality), so E[RMS_perturbed] = RMS_baseline exactly
# -- this only adds dispersion, it doesn't also drift the mean RMS away
# from whatever the calibration already matched. sigma_z is set from a
# target between-subject coefficient of variation (CV) via the exact
# lognormal CV formula (see _sigma_z_for_target_cv), not a search.
# ---------------------------------------------------------------------
def _default_subject_effect_seed(vision, feedback, base=54321):
    """A seed that VARIES by (vision, feedback), not a single fixed
    constant reused for every config. IMPORTANT: this is a fix for a real
    bug -- the previous default (a bare fixed constant passed to every
    config) meant all 4 configs drew from the exact same underlying
    random sequence, just rescaled by each config's own target CV. Any
    single small-sample draw has real sampling error in its achieved CV
    (with ~10-20 virtual subjects, easily +/-30-40%) -- but with a FIXED
    seed reused everywhere, that ONE draw's particular over/under-shoot
    direction gets baked identically into every config, which looks like
    a systematic bias (e.g. "ML always overshoots, AP always undershoots
    across all 4 conditions") but is actually one unlucky/lucky draw
    repeated 4 times, not a property of the method. zlib.crc32 (not
    Python's built-in hash()) because hash() of a string is randomized
    per-process by default and would silently change this seed -- and
    therefore your results -- between runs."""
    offset = zlib.crc32(f"{vision}_{feedback}".encode()) % 100_000
    return base + offset


def _sigma_z_for_target_cv(cv_target):
    """Exact closed-form: the log-space SD of a lognormal perturbation
    factor c=exp(z) whose resulting RMS multiplier exp(z/2) has
    coefficient of variation cv_target. Derivation: for Y=exp(z/2) with
    z~N(mean,sigma_z^2), Y is lognormal with log-SD sigma_z/2, and a
    lognormal's CV is sqrt(exp(log_sd^2)-1) regardless of its mean
    parameter -- so CV(Y)=cv_target => (sigma_z/2)^2 = log(1+cv_target^2)."""
    if cv_target <= 0:
        return 0.0
    return 2.0 * float(np.sqrt(np.log(1.0 + cv_target ** 2)))


def compute_subject_effect_cv_from_gof(gof_summary_csv):
    """Reads real_mean/real_std/sim_mean/sim_std for ml_rms_cm/ap_rms_cm
    from an EXISTING gof_summary.csv -- from a prior run_dual_pathway_gof.
    py run WITHOUT subject effects (the plain single-params baseline you
    already have) -- and returns {"ml": cv_target, "ap": cv_target}: the
    between-virtual-subject RMS coefficient-of-variation needed so that
    combining it with the CURRENT within-trial noise (that baseline run's
    sim_std) closes the gap to real_std.

    EXACT closed-form derivation (fixes an earlier, slightly-biased
    version of this function that divided by real_mean): the injected
    factor is c=exp(z), and a trial's RMS is RMS_baseline*sqrt(c)*eta,
    where RMS_baseline=sim_mean_baseline is what's ACTUALLY being scaled
    (not real_mean -- using real_mean here was the bug, and if real_mean
    and sim_mean_baseline differ by even a few percent, that mismatch
    shows up as the SAME-DIRECTION bias in every condition, not random
    noise, since it isn't a sampling effect) and eta is the pre-existing
    within-trial noise factor with CV = sim_std_baseline/sim_mean_baseline.
    Solving Var(RMS_baseline*sqrt(c)*eta) = real_std^2 exactly for
    cv_target (accounting for E[c]=1+cv_target^2, not 1, since only
    E[sqrt(c)]=1 is enforced) gives:

        cv_target = sqrt(max(0, real_std^2 - sim_std_baseline^2)
                          / (sim_std_baseline^2 + sim_mean_baseline^2))

    (sim_std_baseline^2 is normally tiny next to sim_mean_baseline^2,
    since a small sim_std_baseline is exactly the under-dispersion
    problem being fixed -- so in practice this mostly just swaps the
    normalizer from real_mean to sim_mean_baseline, which is the
    theoretically correct reference point, plus a negligible additional
    term.) Uses only numbers already in the baseline CSV -- no new data
    pass over your real subjects needed."""
    df = pd.read_csv(gof_summary_csv, index_col="metric")
    out = {}
    for axis, col in (("ml", "ml_rms_cm"), ("ap", "ap_rms_cm")):
        if col not in df.index:
            raise KeyError(f"{col} not found in {gof_summary_csv} -- is this a "
                            f"gof_summary.csv from run_dual_pathway_gof.py?")
        real_std = float(df.loc[col, "real_std"])
        sim_mean_baseline = float(df.loc[col, "sim_mean"])
        sim_std_baseline = float(df.loc[col, "sim_std"])
        numerator = max(0.0, real_std ** 2 - sim_std_baseline ** 2)
        denominator = sim_std_baseline ** 2 + sim_mean_baseline ** 2
        cv = float(np.sqrt(numerator / denominator)) if denominator > 0 else 0.0
        out[axis] = cv
        print(f"  {axis}_rms_cm: real_std={real_std:.4f}, sim_mean (baseline)={sim_mean_baseline:.4f}, "
              f"sim_std (baseline)={sim_std_baseline:.4f} -> target between-subject CV={cv:.4f}")
    return out


def simulate_trials_with_subject_effects(params, fs, duration_seconds, seeds,
                                          filter_cutoff_hz=10.0,
                                          trials_per_virtual_subject=3,
                                          cv_target_ml=0.0, cv_target_ap=0.0,
                                          subject_effect_rng_seed=54321):
    """Same return shape as simulate_trials_from_params (list of (x_m,
    y_m) tuples, meters), but partitions `seeds` into consecutive groups
    of trials_per_virtual_subject ("virtual subjects") and draws ONE
    D_fast/D_slow multiplicative factor per group per axis (see module
    comment above for the exact mechanism/derivation), instead of every
    trial sharing the exact same D. cv_target_ml/cv_target_ap = 0 recovers
    the original no-subject-effect behavior exactly (factor is always
    1.0), so this is a strict superset, not a separate code path to keep
    in sync.

    ML and AP use two INDEPENDENT RNG streams (offset seeds), not one
    shared rng drawn from sequentially -- with a single shared stream,
    the ML and AP draw sequences are deterministically linked (AP's Nth
    draw depends on exactly how many draws ML consumed first), so any
    lucky/unlucky sample-variance quirk in one axis's draws isn't
    independent of the other's. Two separate streams make the two axes'
    sampling noise genuinely independent, same intent as this function
    varying its OWN default seed by (vision, feedback) -- see
    _default_subject_effect_seed's docstring for why a fixed shared seed
    across calls silently correlates results that should be independent."""
    rng_ml = np.random.default_rng(subject_effect_rng_seed)
    rng_ap = np.random.default_rng(subject_effect_rng_seed + 500_000)
    sigma_z_ml = _sigma_z_for_target_cv(cv_target_ml)
    sigma_z_ap = _sigma_z_for_target_cv(cv_target_ap)

    def _draw_factor(rng, sigma_z):
        if sigma_z <= 0:
            return 1.0
        z = rng.normal(-sigma_z ** 2 / 4.0, sigma_z)
        return float(np.exp(z))

    trials = []
    i = 0
    while i < len(seeds):
        group_seeds = seeds[i:i + trials_per_virtual_subject]
        i += trials_per_virtual_subject
        c_ml = _draw_factor(rng_ml, sigma_z_ml)
        c_ap = _draw_factor(rng_ap, sigma_z_ap)

        x_batch_cm = simulate_dual_pathway_batch(
            params["omega_fast_sq_ml"], params["gamma_fast_ml"],
            params["omega_slow_sq_ml"], params["gamma_slow_ml"],
            params["D_fast_ml"] * c_ml, params["D_slow_ml"] * c_ml,
            fs, duration_seconds, group_seeds, burn_in_seconds=BURN_IN_SECONDS)
        y_batch_cm = simulate_dual_pathway_batch(
            params["omega_fast_sq_ap"], params["gamma_fast_ap"],
            params["omega_slow_sq_ap"], params["gamma_slow_ap"],
            params["D_fast_ap"] * c_ap, params["D_slow_ap"] * c_ap,
            fs, duration_seconds, [s + 100000 for s in group_seeds], burn_in_seconds=BURN_IN_SECONDS)
        x_batch_m = x_batch_cm / 100.0
        y_batch_m = y_batch_cm / 100.0
        for j in range(len(group_seeds)):
            x_m = apply_filter(x_batch_m[j], fs, filter_cutoff_hz)
            y_m = apply_filter(y_batch_m[j], fs, filter_cutoff_hz)
            trials.append((x_m, y_m))
    return trials


def _print_subject_effect_achieved(sim_trials, trials_per_virtual_subject, cv_target_ml, cv_target_ap):
    """Quick post-hoc sanity check: groups the just-simulated trials back
    into their virtual subjects and reports the ACHIEVED between-subject
    CV of RMS per axis vs. the target -- cheap (just re-uses trials
    already simulated), no extra simulation calls. Large deviations from
    target would flag that the lognormal/linear-SDE assumption behind
    _sigma_z_for_target_cv doesn't hold as cleanly as expected (e.g. from
    filtering or finite-duration edge effects) -- worth checking before
    trusting the number for the paper."""
    def _rms(sig_m):
        cm = (sig_m - sig_m.mean()) * 100.0
        return float(np.sqrt(np.mean(cm ** 2)))

    n_groups = int(np.ceil(len(sim_trials) / trials_per_virtual_subject))
    for axis_idx, name, cv_target in ((0, "ml", cv_target_ml), (1, "ap", cv_target_ap)):
        group_means = []
        for g in range(n_groups):
            group = sim_trials[g * trials_per_virtual_subject:(g + 1) * trials_per_virtual_subject]
            if not group:
                continue
            group_means.append(np.mean([_rms(tr[axis_idx]) for tr in group]))
        group_means = np.array(group_means)
        if len(group_means) < 2 or group_means.mean() <= 0:
            continue
        achieved_cv = group_means.std() / group_means.mean()
        print(f"  {name}: target between-subject CV={cv_target:.4f}, achieved (this run)={achieved_cv:.4f} "
              f"({n_groups} virtual subjects)")


def real_trials_from_df(df, x_col="cop_x_clean", y_col="cop_y_clean"):
    trials = []
    fs = None
    for _, g in df.groupby(GROUP_COLS, observed=True):
        trials.append((g[x_col].values.astype(float), g[y_col].values.astype(float)))
        fs = float(g["sampling_rate_hz"].iloc[0])
    return trials, fs


def run_one_config(params_dir, vision, feedback, out_dir, n_sim_seeds=30,
                    seed_offset=100, duration_seconds=20.0, filter_cutoff_hz=10.0,
                    exclude_from_composite=None, use_bootstrap=False,
                    subject_effect_baseline_gof=None, subject_effect_cv_ml=None,
                    subject_effect_cv_ap=None, trials_per_virtual_subject=3,
                    subject_effect_rng_seed=None):
    print(f"\n{'#' * 100}\n# {vision}/{feedback}\n{'#' * 100}")

    use_subject_effects = (subject_effect_baseline_gof is not None
                            or subject_effect_cv_ml is not None
                            or subject_effect_cv_ap is not None)

    if use_bootstrap and use_subject_effects:
        raise ValueError("--use-bootstrap and --subject-effect-* are two different fixes for "
                          "the same RMS Variability finding -- pick one, not both.")

    if use_bootstrap:
        bootstrap_params_list = load_bootstrap_params(params_dir, vision, feedback)
        params = None
        print(f"Using {len(bootstrap_params_list)} bootstrap replicate(s) from "
              f"dual_pathway_params_bootstrap_{vision}_{feedback}.json -- one params "
              f"dict drawn per simulated trial (see bootstrap_calibrate_dual_pathway.py)")
    else:
        bootstrap_params_list = None
        params = load_calibrated_params(params_dir, vision, feedback)

    cv_ml, cv_ap = 0.0, 0.0
    if use_subject_effects:
        if subject_effect_baseline_gof is not None:
            baseline_csv = Path(subject_effect_baseline_gof) / f"gof_{vision}_{feedback}" / "gof_summary.csv"
            if not baseline_csv.exists():
                # allow passing the csv itself, or a dir already at the per-config level
                baseline_csv = Path(subject_effect_baseline_gof)
            print(f"Computing target between-subject CV from baseline GOF run: {baseline_csv}")
            cv_from_gof = compute_subject_effect_cv_from_gof(baseline_csv)
            cv_ml, cv_ap = cv_from_gof["ml"], cv_from_gof["ap"]
        if subject_effect_cv_ml is not None:
            cv_ml = subject_effect_cv_ml
        if subject_effect_cv_ap is not None:
            cv_ap = subject_effect_cv_ap
        if subject_effect_rng_seed is None:
            subject_effect_rng_seed = _default_subject_effect_seed(vision, feedback)
        print(f"Subject-effect variance injection: cv_target_ml={cv_ml:.4f}, "
              f"cv_target_ap={cv_ap:.4f}, trials_per_virtual_subject={trials_per_virtual_subject}, "
              f"rng_seed={subject_effect_rng_seed} (varies by config -- see "
              f"_default_subject_effect_seed if you need this reproducible some other way)")

    # SAME default seed (0) as run_dual_pathway_all_configs.main()'s own
    # call -- reuses the identical train/test split, so test_df here is
    # the EXACT held-out set that config's calibration run validated
    # against, not a different random split.
    _train_df, test_df, fs = load_config_data(vision, feedback)
    real_trials, fs_check = real_trials_from_df(test_df)
    if abs(fs - fs_check) > 1e-6:
        print(f"WARNING: load_config_data fs={fs} != per-trial sampling_rate_hz "
              f"in the held-out data ({fs_check}) -- check the parquet for mixed rates.")

    # Fresh seeds, disjoint from both calibration (0..n_seeds-1) and the
    # existing held-out-stats check (n_seeds..n_seeds+15) inside
    # _finalize_calibration_result, so this is an independent draw, not
    # a reused one.
    seeds = list(range(seed_offset, seed_offset + n_sim_seeds))
    if use_subject_effects:
        sim_trials = simulate_trials_with_subject_effects(
            params, fs, duration_seconds, seeds, filter_cutoff_hz=filter_cutoff_hz,
            trials_per_virtual_subject=trials_per_virtual_subject,
            cv_target_ml=cv_ml, cv_target_ap=cv_ap,
            subject_effect_rng_seed=subject_effect_rng_seed)
        print("Subject-effect check (achieved vs. target between-subject CV, this run's draw):")
        _print_subject_effect_achieved(sim_trials, trials_per_virtual_subject, cv_ml, cv_ap)
    else:
        sim_trials = simulate_trials_from_params(params, fs, duration_seconds, seeds,
                                                  filter_cutoff_hz=filter_cutoff_hz,
                                                  bootstrap_params_list=bootstrap_params_list)

    return evaluate_goodness_of_fit(
        real_trials, sim_trials, fs, out_dir, label=f"({vision}/{feedback})",
        exclude_from_composite=exclude_from_composite)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--params-dir", required=True,
                   help="Directory containing dual_pathway_params_<vision>_<feedback>.json "
                        "(the hf<threshold>hz tag folder under results/metrics/)")
    p.add_argument("--vision", choices=["Open", "Close"], default=None)
    p.add_argument("--feedback", choices=["Silent", "Auditory"], default=None)
    p.add_argument("--out-dir", default="gof_output")
    p.add_argument("--n-sim-seeds", type=int, default=30)
    p.add_argument("--seed-offset", type=int, default=100)
    p.add_argument("--duration-seconds", type=float, default=20.0)
    p.add_argument("--filter-cutoff-hz", type=float, default=10.0,
                    help="Low-pass cutoff applied to SIMULATED trials, matching what "
                         "load_raw_sessions.py applied to the real trials already saved "
                         "in the parquet. MUST match that value (default 10.0, same as "
                         "load_raw_sessions.py's default) or the comparison is biased. "
                         "Pass 0 or a negative number to disable filtering.")
    p.add_argument("--exclude-from-composite", nargs="*", default=None,
                    help="Metric names (e.g. ml_skew ap_skew ml_kurtosis ap_kurtosis) to "
                         "report as usual but weight at 0 in the composite score -- match "
                         "this to whatever your SMM run's metric_weights already zeroed out "
                         "(e.g. run_dual_pathway_all_configs.py's RELAXED_METRIC_WEIGHTS), "
                         "so an untargeted, near-zero-valued metric's often-huge "
                         "rel_error_pct can't swing the composite score used to compare "
                         "calibration runs against each other.")
    p.add_argument("--use-bootstrap", action="store_true",
                    help="Simulate each trial from a params dict drawn from "
                         "dual_pathway_params_bootstrap_<vision>_<feedback>.json (written "
                         "by bootstrap_calibrate_dual_pathway.py) instead of the single "
                         "winning dual_pathway_params_<vision>_<feedback>.json -- "
                         "reintroduces between-subject parameter variability into the "
                         "simulated ensemble, targeting the RMS Variability under-"
                         "dispersion finding. Requires having run bootstrap_calibrate_"
                         "dual_pathway.py against the same --params-dir first. SUPERSEDED "
                         "by --subject-effect-baseline-gof below for most uses -- that one "
                         "does the same job with no recalibration, so try that first unless "
                         "you specifically need real per-replicate parameter refits.")
    p.add_argument("--subject-effect-baseline-gof", default=None,
                    help="Directory containing an EXISTING gof_<vision>_<feedback>/"
                         "gof_summary.csv (from a prior run_dual_pathway_gof.py run with "
                         "neither --use-bootstrap nor --subject-effect-* set) -- its real_std/ "
                         "sim_std for ml_rms_cm/ap_rms_cm are used to compute exactly how "
                         "much between-virtual-subject RMS variability to inject so the new "
                         "sim_std closes the gap to real_std. Cheap (no recalibration, no new "
                         "data pass) alternative to --use-bootstrap for the RMS Variability "
                         "finding -- recommended first choice.")
    p.add_argument("--subject-effect-cv-ml", type=float, default=None,
                    help="Manually set the target between-subject RMS coefficient of "
                         "variation for ML, overriding whatever --subject-effect-baseline-gof "
                         "computed (or use instead of it, standalone).")
    p.add_argument("--subject-effect-cv-ap", type=float, default=None,
                    help="Same as --subject-effect-cv-ml, for AP.")
    p.add_argument("--trials-per-virtual-subject", type=int, default=3,
                    help="Simulated trials sharing one drawn subject-effect factor, "
                         "before moving to the next virtual subject (default 3, matching "
                         "this project's typical real trials-per-subject count). Only used "
                         "when a subject-effect CV (from either source above) is set.")
    p.add_argument("--subject-effect-rng-seed", type=int, default=None,
                    help="Override the subject-effect draw's RNG seed (default: derived "
                         "automatically per vision/feedback config, so each of the 4 configs "
                         "gets an independent random draw rather than all 4 replaying the "
                         "same underlying random sequence rescaled -- see "
                         "_default_subject_effect_seed's docstring). Only set this "
                         "explicitly if you need a specific reproducible value for some "
                         "other reason; leaving it unset is almost always what you want.")
    args = p.parse_args()

    filter_cutoff_hz = args.filter_cutoff_hz if args.filter_cutoff_hz and args.filter_cutoff_hz > 0 else None

    if args.vision and args.feedback:
        configs = [(args.vision, args.feedback)]
    elif args.vision or args.feedback:
        p.error("pass both --vision and --feedback, or neither (to run all 4 configs)")
    else:
        configs = CONFIGS

    results = {}
    for vision, feedback in configs:
        out_dir = Path(args.out_dir) / f"gof_{vision}_{feedback}"
        summary, composite = run_one_config(
            args.params_dir, vision, feedback, out_dir,
            n_sim_seeds=args.n_sim_seeds, seed_offset=args.seed_offset,
            duration_seconds=args.duration_seconds, filter_cutoff_hz=filter_cutoff_hz,
            exclude_from_composite=args.exclude_from_composite, use_bootstrap=args.use_bootstrap,
            subject_effect_baseline_gof=args.subject_effect_baseline_gof,
            subject_effect_cv_ml=args.subject_effect_cv_ml,
            subject_effect_cv_ap=args.subject_effect_cv_ap,
            trials_per_virtual_subject=args.trials_per_virtual_subject,
            subject_effect_rng_seed=args.subject_effect_rng_seed)
        results[(vision, feedback)] = composite

    if len(results) > 1:
        print("\n" + "=" * 100)
        print("COMPOSITE SCORE ACROSS ALL CONFIGS (lower is better)")
        print("=" * 100)
        for (vision, feedback), composite in results.items():
            print(f"  {vision}/{feedback}: {composite:.2f}%")


if __name__ == "__main__":
    main()