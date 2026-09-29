"""
SMM calibration for the dual-pathway (fast + slow oscillator) model --
the second-resonant-channel structure motivated by this project's
earliest finding (a two-timescale velocity ACF that no single-oscillator
model, with any noise shape, could reproduce) and by every subsequent
attempt (white noise, fractional noise, on both the simple oscillator and
the DDE) hitting the same structural high-frequency-content ceiling.

WHY SMM SPECIFICALLY, HERE: this exact model was tried before via direct
MLE (dual_pathway_mle.py) and found SEVERELY non-identifiable -- a
multi-start diagnostic found wildly different (omega_fast, gamma_fast,
omega_slow, gamma_slow, D_fast, D_slow) combinations achieving nearly
identical likelihood, even after pooling 10 trials. That failure is
SPECIFIC to estimation (trying to recover "the true" parameters from
noisy data). SMM never tries to do that -- it searches for ANY parameters
that reproduce target statistics, and is structurally indifferent to
whether multiple parameter combinations could achieve the same result.
The exact failure mode that killed this model via MLE is one SMM is
immune to by construction.

Reuses dual_pathway_system.build_continuous_system_dual_pathway and
kalman_discretize.van_loan_discretize_general -- BOTH already validated
this session (block-diagonal structure confirmed, zero-noise limit
confirmed, cross-validated bit-identical against the original scalar Van
Loan function in the single-noise-source case).

STABILITY ADVANTAGE OVER THE DDE: omega_fast_sq, gamma_fast, omega_slow_sq,
gamma_slow are all constrained positive -- a genuinely damped linear
oscillator is stable for ANY positive values, unlike the DDE's
alpha-driven marginal instability. No stability guard should be needed
here, structurally, not just empirically -- worth confirming, not just
assuming, during validation below.

PARAMETERIZATION: omega_fast_sq is parameterized as omega_slow_sq +
exp(gap), structurally guaranteeing fast > slow (same trick used in
dual_pathway_mle.py) -- purely for search efficiency here (halves the
effective search space), not for identifiability (SMM doesn't need it for
that).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

sys.path.append(str(Path(__file__).resolve().parent))
from model_tools import (
    extract_real_metrics_extended_trial_aware,
    _all_metric_names,
    _compute_extended_metric_vector,
    simulate_dual_pathway_batch,
    GROUP_COLS,
    BURN_IN_SECONDS
)

SMALL_VALUE_FLOOR = {
    'skew_ap': 0.05,
    'skew_ml': 0.05,
    'hf_power_frac_ap': 0.005,
    'hf_power_frac_ml': 0.005,
}

# Floors on the damping coefficients: below these, the corresponding
# pathway stops being meaningfully damped (see the discussion inside
# simulate_dual_pathway_and_compute_stats). Promoted to module level at
# some point after the version seen earlier in this conversation, which
# still had these as local variables inside that function.
MIN_GAMMA_SLOW = 0.05
MIN_GAMMA_FAST = 0.1

# Floor on the slow pathway's damping ratio zeta_slow = gamma_slow /
# (2*sqrt(omega_slow_sq)). Enforced both as a hard rejection inside
# simulate_dual_pathway_and_compute_stats (see below) and, at the init-point
# level, via _feasible_init_params / random_feasible_init_params, which both
# derive a starting gamma_slow that is guaranteed to clear this floor by
# construction (with some safety margin) rather than relying on the
# optimizer to discover a feasible region on its own.
MIN_ZETA_SLOW = 0.3
MIN_ZETA_FAST = 0.3


# Minimum allowed ratio Var_slow / Var_fast (stationary-variance ratio of
# the two pathways). Guards against the slow pathway being calibrated into
# irrelevance (D_slow -> 0) purely to satisfy a high-frequency-content
# target at the expense of amplitude fidelity -- see the long comment
# inside simulate_dual_pathway_and_compute_stats for the empirical history
# behind this specific value (raised from an earlier, too-permissive 0.02).
MIN_VARIANCE_RATIO = 0.3

def simulate_dual_pathway_and_compute_stats(params: np.ndarray, fixed_seeds: list, fs: float,
                                             duration_seconds: float) -> dict:
    """
    params = [omega_slow_sq_ML, gap_ML, gamma_fast_ML, gamma_slow_ML, D_fast_ML, D_slow_ML,
              omega_slow_sq_AP, gap_AP, gamma_fast_AP, gamma_slow_AP, D_fast_AP, D_slow_AP]
    -- 12 parameters. omega_fast_sq = omega_slow_sq + gap (gap > 0, structurally fast > slow).
    """
    (omega_slow_sq_ml, gap_ml, gamma_fast_ml, gamma_slow_ml, D_fast_ml, D_slow_ml,
     omega_slow_sq_ap, gap_ap, gamma_fast_ap, gamma_slow_ap, D_fast_ap, D_slow_ap) = params

    if np.any(np.array(params) <= 0):
        return {name: 1000000.0 for name in _all_metric_names()}

    # Explicit floor on damping coefficients, added after finding a run
    # converge to gamma_slow_ml=0.0000 -- confirmed directly (not assumed)
    # that this produces genuinely non-stationary behavior: simulated
    # std kept growing well past the 20s calibration window (0.47->0.75 by
    # 60s, 0.63->0.91 by 120s), the signature of a near-undamped oscillator
    # with no proper stationary variance. The variance-ratio check below
    # couldn't catch this on its own: Var_slow = D_slow/(gamma_slow*omega_sq)
    # actually GROWS toward infinity as gamma_slow->0, so a near-zero gamma
    # trivially satisfies "variance ratio is large enough" for the wrong
    # reason. Every genuinely healthy prior run had gamma_slow in 0.2-0.4
    # and gamma_fast in 0.5-13.0; these floors sit well below both ranges.
    if (gamma_slow_ml < MIN_GAMMA_SLOW or gamma_slow_ap < MIN_GAMMA_SLOW or
            gamma_fast_ml < MIN_GAMMA_FAST or gamma_fast_ap < MIN_GAMMA_FAST):
        return {name: 1000000.0 for name in _all_metric_names()}

    omega_fast_sq_ml = omega_slow_sq_ml + gap_ml
    omega_fast_sq_ap = omega_slow_sq_ap + gap_ap

    # Floor on the slow pathway's damping ratio -- see MIN_ZETA_SLOW above.
    zeta_slow_ml = gamma_slow_ml / (2 * np.sqrt(omega_slow_sq_ml))
    zeta_slow_ap = gamma_slow_ap / (2 * np.sqrt(omega_slow_sq_ap))
    if zeta_slow_ml < MIN_ZETA_SLOW or zeta_slow_ap < MIN_ZETA_SLOW:
        return {name: 1000000.0 for name in _all_metric_names()}

    zeta_fast_ml = gamma_fast_ml / (2 * np.sqrt(omega_fast_sq_ml))
    zeta_fast_ap = gamma_fast_ap / (2 * np.sqrt(omega_fast_sq_ap))
    if zeta_fast_ml < MIN_ZETA_FAST or zeta_fast_ap < MIN_ZETA_FAST:
        return {name: 1000000.0 for name in _all_metric_names()}

    # Prevent the slow pathway from being effectively disabled -- found
    # directly: a run that added hf_power_frac to the objective converged to
    # D_slow_ml=0.0001 vs D_fast_ml=5.08, and severely degraded amplitude
    # fidelity (rms_ml -37.5%, p2p_ml -33.9% held-out) while boosting the
    # high-frequency ratio only by shrinking the slow pathway's contribution,
    # not by genuinely balancing both.
    #
    # IMPORTANT CORRECTION: an earlier version of this check compared raw
    # D_slow/D_fast directly and required a minimum ratio of 0.1 -- but
    # checking against every previously-successful run's actual parameters
    # showed their raw ratios ranged from 0.005 to 0.298, so that constraint
    # would have rejected legitimately good configurations. Raw D isn't the
    # right comparison: a lightly-damped, low-frequency oscillator (the slow
    # pathway, by design) produces large stationary variance from small D,
    # since variance = D/(gamma*omega_sq) and gamma*omega_sq is small there.
    # The correct comparison is each pathway's actual stationary variance
    # contribution. The collapsed run's Var_slow/Var_fast was 0.0013; a
    # genuinely healthy prior run's was 2.16 -- a >1000x gap, giving a much
    # cleaner, mechanistically-grounded threshold than the raw D check did.
    var_fast_ml = D_fast_ml / (gamma_fast_ml * omega_fast_sq_ml)
    var_slow_ml = D_slow_ml / (gamma_slow_ml * omega_slow_sq_ml)
    var_fast_ap = D_fast_ap / (gamma_fast_ap * omega_fast_sq_ap)
    var_slow_ap = D_slow_ap / (gamma_slow_ap * omega_slow_sq_ap)
    # CORRECTION: 0.02 was found to be far too permissive in practice -- a
    # run converged to ML variance ratio = 0.0207, sitting right on top of
    # that floor, with amplitude damage (rms_ml -37.8%, mean_vel_ap +80.2%
    # held-out) nearly identical to the original unconstrained collapse. The
    # optimizer wasn't landing there by chance: starving the slow pathway
    # genuinely helps match hf_power_frac, so it rides whatever floor is
    # set. 0.02 was 108x below a genuinely healthy run's actual ratio
    # (2.16), which is far too loose to force real balance. Raised to 0.3 --
    # still meaningfully below the healthy value (leaving room for genuine
    # variation) but no longer close enough to functionally re-permit the
    # same collapse under a technicality.
    if (var_slow_ml < MIN_VARIANCE_RATIO * var_fast_ml or
            var_slow_ap < MIN_VARIANCE_RATIO * var_fast_ap):
        return {name: 1000000.0 for name in _all_metric_names()}

    x_batch = simulate_dual_pathway_batch(omega_fast_sq_ml, gamma_fast_ml, omega_slow_sq_ml, gamma_slow_ml,
                                           D_fast_ml, D_slow_ml, fs, duration_seconds, fixed_seeds,
                                           burn_in_seconds=BURN_IN_SECONDS)
    y_batch = simulate_dual_pathway_batch(omega_fast_sq_ap, gamma_fast_ap, omega_slow_sq_ap, gamma_slow_ap,
                                           D_fast_ap, D_slow_ap, fs, duration_seconds,
                                           [s + 100000 for s in fixed_seeds],
                                           burn_in_seconds=BURN_IN_SECONDS)

    names = _all_metric_names()
    if not (np.all(np.isfinite(x_batch)) and np.all(np.isfinite(y_batch))):
        return {name: 1000000.0 for name in names}

    all_metrics = []
    for i in range(len(fixed_seeds)):
        x_m = x_batch[i] / 100.0  # cm -> meters
        y_m = y_batch[i] / 100.0
        all_metrics.append(_compute_extended_metric_vector(x_m - np.mean(x_m), y_m - np.mean(y_m), fs))

    arr = np.array(all_metrics)
    return {name: float(np.mean(arr[:, i])) for i, name in enumerate(names)}


def smm_dual_pathway_objective(params: np.ndarray, target_stats: dict, fixed_seeds: list, fs: float,
                                duration_seconds: float, metric_weights: dict = None) -> float:
    sim_stats = simulate_dual_pathway_and_compute_stats(params, fixed_seeds, fs, duration_seconds)
    loss = 0.0
    for name, real_v in target_stats.items():
        sim_v = sim_stats[name]
        denom = max(abs(real_v), SMALL_VALUE_FLOOR.get(name, 1e-06))
        rel_err = (sim_v - real_v) / denom
        weight = (metric_weights or {}).get(name, 1.0)
        loss += weight * rel_err ** 2
    return loss


def _feasible_init_params(init_params: np.ndarray = None) -> np.ndarray:
    """Derives a starting point that's always feasible under whatever
    MIN_ZETA_SLOW is currently active. Shared between the single-run and
    basin-hopping calibrators so the same feasibility guarantee applies to
    both, rather than duplicating (and risking re-diverging) this logic.
    """
    if init_params is not None:
        return init_params
    zeta_margin = 1.5 * MIN_ZETA_SLOW
    zeta_fast_margin = 1.5 * MIN_ZETA_FAST
    omega_slow_sq_ml_init, omega_slow_sq_ap_init = (1.5, 2.0)
    gap_ml, gap_ap = 58.5, 58.0
    omega_fast_sq_ml_init = omega_slow_sq_ml_init + gap_ml
    omega_fast_sq_ap_init = omega_slow_sq_ap_init + gap_ap
    gamma_slow_ml_init = zeta_margin * 2 * np.sqrt(omega_slow_sq_ml_init)
    gamma_slow_ap_init = zeta_margin * 2 * np.sqrt(omega_slow_sq_ap_init)
    gamma_fast_ml_init = zeta_fast_margin * 2 * np.sqrt(omega_fast_sq_ml_init)
    gamma_fast_ap_init = zeta_fast_margin * 2 * np.sqrt(omega_fast_sq_ap_init)
    return np.array([
        omega_slow_sq_ml_init, gap_ml, gamma_fast_ml_init, gamma_slow_ml_init, 5.0, 3.0,
        omega_slow_sq_ap_init, gap_ap, gamma_fast_ap_init, gamma_slow_ap_init, 5.0, 3.0,
    ])


def _finalize_calibration_result(x_final, loss_final, success_final, extra_info: dict,
                                  target_stats: dict, test_df: pd.DataFrame, fs: float,
                                  duration_seconds: float, n_seeds: int) -> dict:
    """Builds params_dict + validation_table from a final parameter vector,
    regardless of which optimizer/search strategy produced it -- shared by
    calibrate_dual_pathway_smm (and, presumably, a basin-hopping variant
    that also produces an x_final/loss_final/success_final triple) so the
    reporting logic is written and fixed in exactly one place.
    """
    param_names = [
        'omega_slow_sq_ml', 'gap_ml', 'gamma_fast_ml', 'gamma_slow_ml', 'D_fast_ml', 'D_slow_ml',
        'omega_slow_sq_ap', 'gap_ap', 'gamma_fast_ap', 'gamma_slow_ap', 'D_fast_ap', 'D_slow_ap',
    ]
    raw_params = dict(zip(param_names, x_final))
    params_dict = dict(raw_params)
    params_dict['omega_fast_sq_ml'] = raw_params['omega_slow_sq_ml'] + raw_params['gap_ml']
    params_dict['omega_fast_sq_ap'] = raw_params['omega_slow_sq_ap'] + raw_params['gap_ap']

    zeta_slow_ml = raw_params['gamma_slow_ml'] / (2 * np.sqrt(raw_params['omega_slow_sq_ml']))
    zeta_slow_ap = raw_params['gamma_slow_ap'] / (2 * np.sqrt(raw_params['omega_slow_sq_ap']))
    params_dict['zeta_slow_ml'] = zeta_slow_ml
    params_dict['zeta_slow_ap'] = zeta_slow_ap
    params_dict['min_zeta_slow_floor'] = MIN_ZETA_SLOW

    zeta_fast_ml = raw_params['gamma_fast_ml'] / (2 * np.sqrt(params_dict['omega_fast_sq_ml']))
    zeta_fast_ap = raw_params['gamma_fast_ap'] / (2 * np.sqrt(params_dict['omega_fast_sq_ap']))
    params_dict['zeta_fast_ml'] = zeta_fast_ml
    params_dict['zeta_fast_ap'] = zeta_fast_ap
    params_dict['min_zeta_fast_floor'] = MIN_ZETA_FAST
    params_dict.update(extra_info)

    fixed_seeds = list(range(n_seeds))
    sim_stats_train_target = simulate_dual_pathway_and_compute_stats(x_final, fixed_seeds, fs, duration_seconds)
    test_stats = extract_real_metrics_extended_trial_aware(
        test_df, fs, 'cop_x_clean', 'cop_y_clean', GROUP_COLS)
    sim_stats_more_seeds = simulate_dual_pathway_and_compute_stats(
        x_final, list(range(n_seeds, n_seeds + 16)), fs, duration_seconds)

    rows = []
    for name in _all_metric_names():
        train_target = target_stats[name]
        calib_sim = sim_stats_train_target[name]
        test_real = test_stats[name]
        held_out_sim = sim_stats_more_seeds[name]
        calib_err = 100 * (calib_sim - train_target) / train_target if train_target != 0 else float('nan')
        heldout_err = 100 * (held_out_sim - test_real) / test_real if test_real != 0 else float('nan')
        rows.append({
            'metric': name,
            'train_target': round(train_target, 4),
            'calibrated_sim': round(calib_sim, 4),
            'calib_pct_err': round(calib_err, 1),
            'test_heldout': round(test_real, 4),
            'heldout_sim_fresh_seeds': round(held_out_sim, 4),
            'heldout_pct_err': round(heldout_err, 1),
        })

    validation_table = pd.DataFrame(rows)
    return {
        'params': params_dict,
        'x_raw': x_final,
        'loss': loss_final,
        'success': success_final,
        'validation_table': validation_table,
    }


def calibrate_dual_pathway_smm(train_df: pd.DataFrame, test_df: pd.DataFrame, fs: float,
                                duration_seconds: float = 20.0, n_seeds: int = 8,
                                init_params: np.ndarray = None, metric_weights: dict = None,
                                max_iter: int = 2000, verbose: bool = True,
                                show_progress: bool = True, tqdm_position: int = None,
                                tqdm_desc: str = None) -> dict:
    target_stats = extract_real_metrics_extended_trial_aware(
        train_df, fs, 'cop_x_clean', 'cop_y_clean', GROUP_COLS)

    fixed_seeds = list(range(n_seeds))
    init_params = _feasible_init_params(init_params)

    if verbose:
        print("  Calibrating dual-pathway via SMM (Nelder-Mead)...")

    pbar = tqdm(
        total=max_iter,
        desc=(tqdm_desc or "  Dual-pathway SMM calibration"),
        unit="iter",
        disable=not (_HAS_TQDM and show_progress),
        position=tqdm_position,
        leave=True,
    )
    _last_loss = {'value': None}

    def _tracked_objective(params, *args):
        loss = smm_dual_pathway_objective(params, *args)
        _last_loss['value'] = loss
        return loss

    def _callback(xk):
        pbar.update(1)
        if _last_loss['value'] is not None:
            pbar.set_postfix({'loss': f'{_last_loss["value"]:.5f}'})

    if not _HAS_TQDM and show_progress:
        print("  (tqdm not installed -- running without a progress bar)")

    result = minimize(
        _tracked_objective, init_params,
        args=(target_stats, fixed_seeds, fs, duration_seconds, metric_weights),
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 0.0001, 'fatol': 1e-06, 'adaptive': True},
        callback=_callback,
    )
    pbar.close()

    if verbose:
        print(f"  Final loss: {result.fun:.6f}  [success={result.success}]  "
              f"[{result.nit} iterations, {result.nfev} evaluations]")

    return _finalize_calibration_result(
        result.x, result.fun, result.success,
        {'n_hops': 1, 'optimizer': 'nelder-mead-single-start'},
        target_stats, test_df, fs, duration_seconds, n_seeds,
    )

def random_feasible_init_params(rng: np.random.Generator = None) -> np.ndarray:
    """Like _feasible_init_params, but randomizes the starting point within
    a feasible-margin band (1.2x-3.0x MIN_ZETA_SLOW / MIN_ZETA_FAST, and
    +/-30-40% jitter on the other parameters) instead of returning one
    fixed point every time. Used for multi-start calibration, where always
    starting from the exact same point defeats the purpose. Motivated by a
    basinhopping smoke test that got stuck reproducing the same shallow
    local basin on every restart when seeded from a single fixed
    init_params.

    gamma_fast_ml/ap are derived from zeta_fast_margin * omega_fast_sq_init
    (mirroring how gamma_slow_ml/ap is already derived from zeta_margin *
    omega_slow_sq_init), not drawn independently of omega_fast_sq -- an
    earlier version drew gamma_fast as a flat 5.0 * jitter, which is only
    feasible under MIN_ZETA_FAST by coincidence and, combined with the
    initial gap being large (omega_fast_sq >> omega_slow_sq by design),
    reliably landed below MIN_ZETA_FAST in practice: confirmed directly by
    a run where every restart's loss was bit-identical across all 500
    iterations (Nelder-Mead's initial simplex was entirely infeasible, so
    every vertex returned the same flat rejection penalty and fatol's
    zero-spread convergence check fired immediately).
    """
    rng = rng or np.random.default_rng()
    zeta_margin = MIN_ZETA_SLOW * rng.uniform(1.2, 3.0)
    zeta_fast_margin = MIN_ZETA_FAST * rng.uniform(1.2, 3.0)
    omega_slow_sq_ml_init = 1.5 * rng.uniform(0.7, 1.4)
    omega_slow_sq_ap_init = 2.0 * rng.uniform(0.7, 1.4)
    gamma_slow_ml_init = zeta_margin * 2 * np.sqrt(omega_slow_sq_ml_init)
    gamma_slow_ap_init = zeta_margin * 2 * np.sqrt(omega_slow_sq_ap_init)
    gap_ml = 58.5 * rng.uniform(0.7, 1.4)
    gap_ap = 58.0 * rng.uniform(0.7, 1.4)
    omega_fast_sq_ml_init = omega_slow_sq_ml_init + gap_ml
    omega_fast_sq_ap_init = omega_slow_sq_ap_init + gap_ap
    gamma_fast_ml = zeta_fast_margin * 2 * np.sqrt(omega_fast_sq_ml_init)
    gamma_fast_ap = zeta_fast_margin * 2 * np.sqrt(omega_fast_sq_ap_init)
    D_fast = 5.0 * rng.uniform(0.6, 1.6)
    D_slow = 3.0 * rng.uniform(0.6, 1.6)
    return np.array([
        omega_slow_sq_ml_init, gap_ml, gamma_fast_ml, gamma_slow_ml_init, D_fast, D_slow,
        omega_slow_sq_ap_init, gap_ap, gamma_fast_ap, gamma_slow_ap_init, D_fast, D_slow,
    ])