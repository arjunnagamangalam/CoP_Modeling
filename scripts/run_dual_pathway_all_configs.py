"""
Runs dual-pathway SMM calibration across all 4 (vision, feedback)
configurations, using multi-start per configuration and reusing the
existing plotting functions from run_dual_pathway_smm_model.py.

WHY THIS EXISTS, NOT JUST A LOOP AROUND main(): the plain single-run
calibrate_dual_pathway_smm() call in run_dual_pathway_smm_model.py's
main() has three problems for a "qualitatively valid across 4 configs"
check specifically:

1. Its hf_weights={'hf_power_frac_ml':3.0,'hf_power_frac_ap':3.0} is a
   DIFFERENT objective than the one all the floor-reliability testing this
   session was done against (which drops skew/kurtosis per advisor
   guidance instead). Untested combination.
2. It doesn't set MIN_ZETA_SLOW, so it uses the module default -- 0.9 as
   of this version, but that default has changed multiple times this
   session and silently inheriting it is exactly the kind of assumption
   this project's own discipline says to check, not trust.
3. It's a single deterministic run. Multi-start testing at MIN_ZETA_SLOW
   =0.9 showed a ~1-in-6 chance of landing in a still-ringing basin even
   at the best floor found so far -- a single run per config risks
   putting a ringing plot in front of your advisor by bad luck, not a
   real problem with the model.

This script: N_RESTARTS independent, randomly-initialized calibrations
PER CONFIGURATION (16 total tasks across 4 configs), all run in parallel;
picks the best NON-RINGING result per config by loss; falls back to
least-ringing (clearly flagged) if literally none of a config's restarts
came back clean; then calls your existing plot_predicted_vs_real and
plot_psd_comparison unchanged on the winning result, so the output format
matches what's already been reviewed.

USAGE:
    python run_dual_pathway_all_configs.py

Requires generalized_fp_heldout.py (for split_trials), dual_pathway_smm.py,
and run_dual_pathway_smm_model.py (for the plotting functions) all
importable from this file's directory.
"""
import sys
import os
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def _status(msg):
    if _HAS_TQDM:
        _tqdm.write(msg)
    else:
        print(msg)

sys.path.append(str(Path(__file__).resolve().parent))
import dual_pathway_smm_calibration as dps
from model_tools import GROUP_COLS
import model_tools as mt


RELAXED_METRIC_WEIGHTS = {
    'skew_ap': 0.0, 'skew_ml': 0.0,
    'kurtosis_ap': 0.0, 'kurtosis_ml': 0.0,
}

CONFIGS = [('Close', 'Silent'), ('Close', 'Auditory'), ('Open', 'Silent'), ('Open', 'Auditory')]
N_RESTARTS = 8  # at MIN_ZETA_SLOW=0.9's measured ~83% clean rate per restart,
                 # P(at least one clean in 4) ~= 1-(1-0.83)^4 ~= 99.9% per config,
                 # ~99.6% all 4 configs get a clean result. Kept at 4 rather than
                 # 6 (used during the exploratory floor sweep) specifically for
                 # turnaround time before a meeting -- 16 tasks instead of 24.

MAX_ITER = 500   # matches run_dual_pathway_smm_model.py's own default, NOT the
                  # 2000 used in the earlier exploratory sweeps (which is why
                  # those took ~18-20 min/run) -- fast enough for 16 tasks to
                  # finish in a few minutes each when parallelized.

N_SEEDS_CALIB = 20  # matches your existing main()'s default, for held-out
                     # stats as smooth as what you've already been reviewing.

RINGING_THRESHOLD = 0.15  # a restart counts as "clean" if BOTH axes' local-window
                            # R^2 fall below this -- matches the threshold used
                            # to classify the floor=0.8/0.9 multi-start results.

FLOOR_VALUE = 0.9  # explicit, not inherited from dual_pathway_smm's module default
                     # -- see the note in _run_one_restart for why that trust failed.

FAST_FLOOR_VALUE = 0.5 # MIN_ZETA_FAST floor -- fixes the under-damped fast AP
                          # pathway causing ap_rms_hf_cm / ap_dfa_alpha_short
                          # overestimation. The existing ringing check
                          # (_local_ringiness_r2) is slow-pathway-only -- it
                          # early-exits whenever zeta_slow>=1.0, which the
                          # MIN_ZETA_SLOW=0.9 floor already guarantees, so it
                          # can't see fast-pathway resonance at all. This has
                          # to be enforced directly on the parameters instead.

HF_THRESHOLD_HZ = 0.3

def load_config_data(vision: str, feedback: str, seed: int = 0):
        """Loads the cleaned parquet, filters to one (vision, feedback)
        configuration, and returns a train/test trial split plus sampling rate."""

        root_dir = Path(__file__).resolve().parent.parent
        cleaned_file = root_dir / "data" / "processed" / "cleaned_balance_all.parquet"
        df = pd.read_parquet(cleaned_file)
        fs = float(df['sampling_rate_hz'].iloc[0])
        subset = df[(df['vision'] == vision) & (df['feedback'] == feedback)]

        train_df, test_df = mt.split_trials(subset, seed=seed)
        return train_df, test_df, fs
 

def _run_one_restart(vision, feedback, restart_idx, position, train_df, test_df, fs, seed_base):
    """Top-level (picklable) worker -- one calibration at one (config,
    restart) pair. Returns the full result dict (not just summary stats)
    so main() can hand the winning restart directly to the existing
    plotting functions without recomputing anything.
    """
    
    tag = f"{vision}/{feedback} r{restart_idx}"
    _status(f"[{tag}] starting (pid={os.getpid()})")
 
    # EXPLICIT, not trusting the module default: ...
    dps.MIN_ZETA_SLOW = FLOOR_VALUE
    dps.MIN_ZETA_FAST = FAST_FLOOR_VALUE
    mt.HF_THRESHOLD_HZ = HF_THRESHOLD_HZ

    rng = np.random.default_rng(seed_base + restart_idx)
    init_params = dps.random_feasible_init_params(rng)

    result = dps.calibrate_dual_pathway_smm(
        train_df, test_df, fs, duration_seconds=20.0, n_seeds=N_SEEDS_CALIB,
        init_params=init_params, metric_weights=RELAXED_METRIC_WEIGHTS,
        max_iter=MAX_ITER, verbose=False, show_progress=True,
        tqdm_position=position, tqdm_desc=f"  {vision[:2]}/{feedback[:2]} r{restart_idx}")

    p = result['params']
    r2_ap, _ = mt._local_ringiness_r2(
        p['omega_fast_sq_ap'], p['gamma_fast_ap'], p['D_fast_ap'],
        p['omega_slow_sq_ap'], p['gamma_slow_ap'], p['D_slow_ap'], fs, 20.0)
    r2_ml, _ = mt._local_ringiness_r2(
        p['omega_fast_sq_ml'], p['gamma_fast_ml'], p['D_fast_ml'],
        p['omega_slow_sq_ml'], p['gamma_slow_ml'], p['D_slow_ml'], fs, 20.0)

    _status(f"[{tag}] done -- loss={result['loss']:.5f}  ringing(ml/ap)={r2_ml:.2f}/{r2_ap:.2f}  "
             f"zeta(ml/ap)={p['zeta_slow_ml']:.2f}/{p['zeta_slow_ap']:.2f}")

    return {
        'vision': vision, 'feedback': feedback, 'restart': restart_idx,
        'loss': result['loss'], 'ringing_r2_ml': r2_ml, 'ringing_r2_ap': r2_ap,
        'zeta_slow_ml': p['zeta_slow_ml'], 'zeta_slow_ap': p['zeta_slow_ap'],
        'result': result,
    }


def _select_winner(candidates: list) -> dict:
    """Picks the best restart for one config: lowest loss among CLEAN
    (non-ringing) candidates if any exist, else the least-ringing overall
    -- with that fallback clearly flagged, since it means every restart
    for this config rang and it needs a closer look, not a silent pick.
    """
    clean = [c for c in candidates if c['ringing_r2_ml'] < RINGING_THRESHOLD
              and c['ringing_r2_ap'] < RINGING_THRESHOLD]
    if clean:
        winner = min(clean, key=lambda c: c['loss'])
        winner['flagged'] = False
        return winner
    winner = min(candidates, key=lambda c: c['ringing_r2_ml'] + c['ringing_r2_ap'])
    winner['flagged'] = True
    return winner




def _verify_hf_threshold_is_enforced():
        """Fails loudly and cheaply if the deployed bootstrap_noise_model.py
        doesn't actually respond to HF_THRESHOLD_HZ overrides -- directly
        probes with a synthetic signal that has a component specifically
        BETWEEN 0.3 and 1.0Hz, so a real threshold change produces a
        clearly different, checkable result (not just "some number changed
        slightly") rather than trusting the module's default silently
        matches what this script intends to set.
        """
        fs_check = 200.0
        t_check = np.arange(0, 20, 1 / fs_check)
        sig_check = np.sin(2 * np.pi * 0.6 * t_check)  # single component, strictly between the two thresholds
 
        mt.HF_THRESHOLD_HZ = 1.0
        frac_at_1p0 = mt._hf_power_fraction(sig_check, fs_check)
        mt.HF_THRESHOLD_HZ = 0.3
        frac_at_0p3 = mt._hf_power_fraction(sig_check, fs_check)
        mt.HF_THRESHOLD_HZ = HF_THRESHOLD_HZ  # restore to this run's actual intended value
 
        if not (frac_at_1p0 < 0.1 and frac_at_0p3 > 0.9):
            raise RuntimeError(
                f"bootstrap_noise_model.py on this filesystem does NOT respond to "
                f"HF_THRESHOLD_HZ overrides as expected (got frac@1.0Hz={frac_at_1p0:.3f}, "
                f"frac@0.3Hz={frac_at_0p3:.3f} on a 0.6Hz test signal that should score "
                f"~0 and ~1 respectively). This is very likely a stale/unpatched copy of "
                f"bootstrap_noise_model.py -- re-sync it before running the full batch."
            )
        print(f"HF-threshold self-check passed: bootstrap_noise_model.py correctly responds "
              f"to HF_THRESHOLD_HZ overrides. This run will use HF_THRESHOLD_HZ={HF_THRESHOLD_HZ}.")
 
def _verify_floor_is_enforced():
    """Fails loudly, immediately, and cheaply if the deployed
    dual_pathway_smm.py doesn't actually enforce MIN_ZETA_SLOW -- rather
    than discovering it after a full 16-task run, the way today's version
    of this bug was found. Directly probes the constraint function with a
    parameter vector known to violate FLOOR_VALUE and confirms it's
    rejected (1e6), rather than trusting that the file on disk matches
    whatever was last reviewed in chat.
    """
    dps.MIN_ZETA_SLOW = FLOOR_VALUE

    # zeta_slow_ap = gamma_slow_ap/(2*sqrt(omega_slow_sq_ap)); build a
    # params vector where AP's zeta is deliberately just under the floor.
    omega_slow_sq_ap = 2.0
    gamma_slow_ap_violating = (FLOOR_VALUE - 0.1) * 2 * np.sqrt(omega_slow_sq_ap)
    params = np.array([1.5, 58.5, 5.0, 5.0, 5.0, 3.0,
                         omega_slow_sq_ap, 58.0, 5.0, gamma_slow_ap_violating, 5.0, 3.0])
    stats = dps.simulate_dual_pathway_and_compute_stats(params, list(range(2)), 200.0, 20.0)
    rejected = all(v == 1e6 for v in stats.values())
    if not rejected:
        raise RuntimeError(
            f"dual_pathway_smm.py on this filesystem does NOT enforce MIN_ZETA_SLOW="
            f"{FLOOR_VALUE} as expected -- a deliberately floor-violating parameter set "
            f"was accepted instead of rejected. This is very likely a stale/outdated copy "
            f"of dual_pathway_smm.py. Re-sync it before running the full batch, or every "
            f"restart below is at risk of the same silent floor failure seen in the last run."
        )
    print(f"Floor-enforcement self-check passed: MIN_ZETA_SLOW={FLOOR_VALUE} is being "
          f"correctly enforced by the deployed dual_pathway_smm.py.")

def _verify_fast_floor_is_enforced():
    """Same self-check as _verify_floor_is_enforced(), for MIN_ZETA_FAST.
    Added alongside the explicit dps.MIN_ZETA_FAST = FAST_FLOOR_VALUE line in
    _run_one_restart -- without both, MIN_ZETA_FAST would be silently
    inherited from whatever dual_pathway_smm.py's module-level default
    happens to be in each worker process, exactly the failure mode this file
    already hit (and fixed) for MIN_ZETA_SLOW.
    """
    dps.MIN_ZETA_SLOW = FLOOR_VALUE
    dps.MIN_ZETA_FAST = FAST_FLOOR_VALUE

    # Build a params vector where ONLY ap's fast-pathway zeta violates
    # FAST_FLOOR_VALUE -- ml's fast zeta and both slow zetas get generous
    # margin, so a rejection can only be attributed to the AP fast-floor
    # check, not to incidentally tripping some other constraint.
    omega_slow_sq_ml, gap_ml = 1.5, 58.5
    omega_slow_sq_ap, gap_ap = 2.0, 58.0
    omega_fast_sq_ml = omega_slow_sq_ml + gap_ml
    omega_fast_sq_ap = omega_slow_sq_ap + gap_ap

    safe_zeta_fast_ml = 1.5 * FAST_FLOOR_VALUE
    safe_zeta_slow = 1.5 * FLOOR_VALUE
    violating_zeta_fast_ap = FAST_FLOOR_VALUE - 0.1

    gamma_fast_ml_safe = safe_zeta_fast_ml * 2 * np.sqrt(omega_fast_sq_ml)
    gamma_slow_ml_safe = safe_zeta_slow * 2 * np.sqrt(omega_slow_sq_ml)
    gamma_slow_ap_safe = safe_zeta_slow * 2 * np.sqrt(omega_slow_sq_ap)
    gamma_fast_ap_violating = violating_zeta_fast_ap * 2 * np.sqrt(omega_fast_sq_ap)

    params = np.array([
        omega_slow_sq_ml, gap_ml, gamma_fast_ml_safe, gamma_slow_ml_safe, 5.0, 3.0,
        omega_slow_sq_ap, gap_ap, gamma_fast_ap_violating, gamma_slow_ap_safe, 5.0, 3.0,
    ])
    stats = dps.simulate_dual_pathway_and_compute_stats(params, list(range(2)), 200.0, 20.0)
    rejected = all(v == 1e6 for v in stats.values())
    if not rejected:
        raise RuntimeError(
            f"dual_pathway_smm.py on this filesystem does NOT enforce MIN_ZETA_FAST="
            f"{FAST_FLOOR_VALUE} as expected -- a deliberately fast-floor-violating "
            f"parameter set (everything else safely feasible) was accepted instead of "
            f"rejected. This is very likely a stale/outdated copy of dual_pathway_smm.py, "
            f"or the MIN_ZETA_FAST check isn't present in simulate_dual_pathway_and_compute_"
            f"stats on this machine. Re-sync it before running the full batch."
        )
    print(f"Fast-floor-enforcement self-check passed: MIN_ZETA_FAST={FAST_FLOOR_VALUE} is "
          f"being correctly enforced by the deployed dual_pathway_smm.py.")
    
def main():
    """Entry point: runs the three startup self-checks, loads all 4 configs'
    data, fans out N_RESTARTS x 4 calibration tasks across worker processes,
    selects each config's winning (non-ringing) restart, then saves its
    validation table, calibrated params, and plots before printing a
    cross-config summary."""
    
    _verify_floor_is_enforced()
    _verify_hf_threshold_is_enforced()
    _verify_fast_floor_is_enforced()

    threshold_tag = f"hf{HF_THRESHOLD_HZ}".replace('.', 'p') + "hz"  # e.g. "hf1p0hz" / "hf0p3hz"
    root_dir = Path(__file__).resolve().parent.parent
    fig_dir = root_dir / "final_model" / "results" / "figures" / threshold_tag
    metrics_dir = root_dir / "final_model" / "results" / "metrics" / threshold_tag
    print(f"\nOutput directory for this run (HF_THRESHOLD_HZ={HF_THRESHOLD_HZ}): {metrics_dir}\n")

    print("Loading data for all 4 configurations...")
    data_by_config = {}
    for vision, feedback in CONFIGS:
        train_df, test_df, fs = load_config_data(vision, feedback)
        data_by_config[(vision, feedback)] = (train_df, test_df, fs)
        n_train = len(train_df[GROUP_COLS].drop_duplicates())
        n_test = len(test_df[GROUP_COLS].drop_duplicates())
        print(f"  {vision}/{feedback}: {n_train} train trials, {n_test} test trials")

    tasks = [(vision, feedback, r) for (vision, feedback) in CONFIGS for r in range(N_RESTARTS)]

    try:
        n_workers = min(len(tasks), len(os.sched_getaffinity(0)))
    except AttributeError:
        n_workers = min(len(tasks), os.cpu_count() or 1)

    print(f"\nRunning {len(tasks)} (config, restart) tasks across {n_workers} worker process(es)")
    print(f"-- request -n {len(tasks)} on Oscar to run all of them simultaneously\n")

    all_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for i, (vision, feedback, r) in enumerate(tasks):
            train_df, test_df, fs = data_by_config[(vision, feedback)]
            fut = executor.submit(_run_one_restart, vision, feedback, r, i,
                                    train_df, test_df, fs, seed_base=2000)
            futures[fut] = (vision, feedback, r)
        for fut in as_completed(futures):
            all_results.append(fut.result())

    print("\n" + "=" * 100)
    print("SELECTING WINNERS AND GENERATING PLOTS")
    print("=" * 100)

    summary_rows = []
    for vision, feedback in CONFIGS:
        candidates = [r for r in all_results if r['vision'] == vision and r['feedback'] == feedback]
        winner = _select_winner(candidates)

        flag_str = "  *** ALL RESTARTS RANG -- LEAST-BAD SHOWN, NEEDS ATTENTION ***" if winner['flagged'] else ""
        print(f"\n{vision}/{feedback}: restart {winner['restart']} selected "
              f"(loss={winner['loss']:.5f}, ringing ml/ap={winner['ringing_r2_ml']:.2f}/"
              f"{winner['ringing_r2_ap']:.2f}){flag_str}")

        result = winner['result']
        train_df, test_df, fs = data_by_config[(vision, feedback)]

        csv_path = metrics_dir / f"all_dual_pathway_validation_{vision}_{feedback}.csv"
        result['validation_table'].to_csv(csv_path, index=False)
        print(f"  Saved statistics table -> {csv_path}")

        # Params were NOT being persisted before -- only the validation
        # table was saved, so any later analysis needing to re-simulate
        # (like the real-vs-predicted bar charts across all 4 configs)
        # had no way to reconstruct the winning result without a full
        # rerun. Fixed: save params directly, keyed by config.
        def _json_safe(v):
            """result['params'] mixes calibrated numeric values with a
            couple of housekeeping fields left over from an earlier
            (reverted) basinhopping refactor of calibrate_dual_pathway_smm
            -- e.g. 'optimizer': 'nelder-mead-single-start', a string, not
            a number. Forcing everything to float() crashes on that field.
            Rather than touch the calibration function itself (working
            reliably, no reason to risk it for a save-to-disk convenience
            function), just serialize each value as whatever native JSON
            type actually fits -- str/int/float/bool/None pass through
            directly, anything else (e.g. a numpy scalar type) gets a
            float() attempt, falling back to str() only if that fails too.
            """
            if isinstance(v, (int, float, str, bool)) or v is None:
                return v
            try:
                return float(v)
            except (TypeError, ValueError):
                return str(v)

        params_path = metrics_dir / f"dual_pathway_params_{vision}_{feedback}.json"
        with open(params_path, 'w') as f:
            json.dump({k: _json_safe(v) for k, v in result['params'].items()}, f, indent=2)
        print(f"  Saved calibrated parameters -> {params_path}")

        mt.plot_predicted_vs_real(test_df, result, fs, vision, feedback, fig_dir,
                                  duration_seconds=20.0, seed=winner['restart'])
        mt.plot_psd_comparison(test_df, result, fs, vision, feedback, fig_dir, duration_seconds=20.0, threshold_hz=HF_THRESHOLD_HZ)

        vt = result['validation_table'].set_index('metric')
        summary_rows.append({
            'vision': vision, 'feedback': feedback, 'restart': winner['restart'],
            'flagged': winner['flagged'], 'loss': winner['loss'],
            'ringing_r2_ml': winner['ringing_r2_ml'], 'ringing_r2_ap': winner['ringing_r2_ap'],
            'zeta_slow_ml': winner['zeta_slow_ml'], 'zeta_slow_ap': winner['zeta_slow_ap'],
            'rms_ap_err_pct': vt.loc['rms_ap_cm', 'heldout_pct_err'],
            'rms_ml_err_pct': vt.loc['rms_ml_cm', 'heldout_pct_err'],
            'p2p_ap_err_pct': vt.loc['p2p_ap_cm', 'heldout_pct_err'],
            'p2p_ml_err_pct': vt.loc['p2p_ml_cm', 'heldout_pct_err'],
            'mean_vel_ap_err_pct': vt.loc['mean_vel_ap_cms', 'heldout_pct_err'],
            'mean_vel_ml_err_pct': vt.loc['mean_vel_ml_cms', 'heldout_pct_err'],
            'hf_power_frac_ap_err_pct': vt.loc['hf_power_frac_ap', 'heldout_pct_err'],
            'hf_power_frac_ml_err_pct': vt.loc['hf_power_frac_ml', 'heldout_pct_err']
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = metrics_dir / "all_dual_pathway_all_configs_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 100)
    print("SUMMARY -- ALL 4 CONFIGURATIONS")
    print("=" * 100)
    print(summary.to_string(index=False))
    print(f"\nSaved -> {summary_path}")
    if summary['flagged'].any():
        print("\n*** At least one configuration had ZERO clean restarts out of "
              f"{N_RESTARTS} -- check the flagged row(s) above before your meeting. "
              "Consider rerunning just that config with more restarts if time allows.")
    else:
        print(f"\nAll 4 configurations found at least one non-ringing result "
              f"(threshold: both-axis ringing R^2 < {RINGING_THRESHOLD}).")


if __name__ == '__main__':
    main()