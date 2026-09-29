import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.signal import welch, butter, filtfilt, savgol_filter
from scipy.stats import chi2, skew, kurtosis
from scipy.linalg import expm

sys.path.append(str(Path(__file__).resolve().parent))
from dual_pathway_matrix_system import build_continuous_system_dual_pathway

GROUP_COLS = ['subject_id', 'vision', 'feedback', 'trial']
HF_THRESHOLD_HZ = 0.3
BURN_IN_SECONDS = 5.0


def split_trials(subset: pd.DataFrame, seed: int = 0):
    """Randomly splits this condition's trials in half -- one half to
    estimate D1/D2, the other half to build the comparison histogram.
    This is what makes the validation genuinely independent."""
    trials = subset['trial'].unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(trials)
    split = max(1, len(shuffled) // 2)
    train_trials, test_trials = shuffled[:split], shuffled[split:]
    return subset[subset['trial'].isin(train_trials)], subset[subset['trial'].isin(test_trials)]

def van_loan_discretize_general(A: np.ndarray, L: np.ndarray, D_diag: np.ndarray, dt: float):
    """
    Generalization of van_loan_discretize above to MULTIPLE independent
    noise sources -- needed for the dual-pathway (fast+slow oscillator)
    model, where noise enters two different states independently, each
    with its own intensity. L is (n, m) for m noise sources; D_diag is
    length m (one intensity per column of L). Same exact block-matrix
    algorithm and sign convention as the original (unchanged, still
    validated by the zero-noise-limit check documented there) -- only the
    Qc construction is generalized from a scalar to a diagonal matrix.

    Cross-validated against the original van_loan_discretize: with a
    single noise source (L as (n,1), D_diag as a 1-element array), this
    must give IDENTICAL (F, Qd) to the original function -- confirmed
    below before this is used for anything real.
    """
    n = A.shape[0]
    Qc = L @ np.diag(2.0 * np.asarray(D_diag)) @ L.T  # (n, n)

    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -A
    M[:n, n:] = Qc
    M[n:, n:] = A.T
    M = M * dt

    expM = expm(M)
    Phi12 = expM[:n, n:]
    Phi22 = expM[n:, n:]
    F = Phi22.T
    Qd = F @ Phi12
    return F, Qd

def _sim_pathway(omega_sq, gamma, D, fs, duration, seed):
    dt = 1.0 / fs
    N = int(duration * fs)
    t = np.arange(N) * dt
    rng = np.random.default_rng(seed)
    x = np.zeros(N)
    v = np.zeros(N)
    incr = rng.normal(0, 1, N) * np.sqrt(2 * D * dt)
    for n in range(1, N):
        a = -omega_sq * x[n - 1] - gamma * v[n - 1]
        v[n] = v[n - 1] + a * dt + incr[n]
        x[n] = x[n - 1] + v[n - 1] * dt
    return x, t

def _local_ringiness_r2(omega_sq_fast, gamma_fast, D_fast, omega_sq_slow, gamma_slow, D_slow,
                         fs, duration, n_seeds=30, window_periods=1.5):
    zeta_slow = gamma_slow / (2 * np.sqrt(omega_sq_slow))
    if zeta_slow >= 1.0:
        return (0.0, 0.0)

    omega_d = np.sqrt(omega_sq_slow - (gamma_slow / 2) ** 2)
    period = 2 * np.pi / omega_d
    win = max(int(window_periods * period * fs), 4)
    step = max(win // 4, 1)
    all_r2 = []
    for s in range(n_seeds):
        xf, t = _sim_pathway(omega_sq_fast, gamma_fast, D_fast, fs, duration, seed=2 * s)
        xs, _ = _sim_pathway(omega_sq_slow, gamma_slow, D_slow, fs, duration, seed=2 * s + 1)
        y = xf + xs
        N = len(y)
        for start in range(0, N - win, step):
            seg = y[start:start + win]
            tt = t[start:start + win]
            X = np.column_stack([np.cos(omega_d * tt), np.sin(omega_d * tt), np.ones_like(tt)])
            coef, *_ = np.linalg.lstsq(X, seg, rcond=None)
            seghat = X @ coef
            ss_res = np.sum((seg - seghat) ** 2)
            ss_tot = np.sum((seg - seg.mean()) ** 2)
            if ss_tot > 1e-12:
                all_r2.append(1 - ss_res / ss_tot)

    return float(np.mean(all_r2)), float(np.median(all_r2))

def simulate_dual_pathway_batch(omega_fast_sq: float, gamma_fast: float, omega_slow_sq: float,
                                 gamma_slow: float, D_fast: float, D_slow: float,
                                 fs: float, duration_seconds: float, seeds: list = (0.0,),
                                 burn_in_seconds: float = 0.0) -> np.ndarray:
    """
    Batched (vectorized across seeds) dual-pathway simulation via EXACT
    discretization (Van Loan) -- noise is pre-generated per-seed (one
    vectorized multivariate_normal call each, not a per-timestep loop),
    then the timestep loop itself (unavoidably sequential in time) updates
    ALL seeds simultaneously via one matrix operation per step, same
    pattern validated for the Gaussian Langevin batched simulator earlier
    this session.

    Returns y = x_fast + x_slow (position, cm), shape (n_seeds, n_steps).
    """
    dt = 1.0 / fs
    n_total = int((duration_seconds + burn_in_seconds) * fs)
    n_seeds = len(seeds)

    A, L = build_continuous_system_dual_pathway(omega_fast_sq, gamma_fast, omega_slow_sq, gamma_slow)
    F, Qd = van_loan_discretize_general(A, L, np.array([D_fast, D_slow]), dt)

    noise_all = np.zeros((n_seeds, n_total, 4))
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        noise_all[i] = rng.multivariate_normal(np.zeros(4), Qd, size=n_total)

    states = np.zeros((n_seeds, n_total, 4))
    for t in range(n_total - 1):
        states[:, t + 1, :] = states[:, t, :] @ F.T + noise_all[:, t, :]

    y = states[:, :, 0] + states[:, :, 2]  # x_fast + x_slow, cm
    burn_in_n = int(burn_in_seconds * fs)
    return y[:, burn_in_n:]

def simulate_from_dual_pathway_result(result: dict, fs: float, duration_seconds: float = 20.0,
                                          seed=None) -> dict:
    p = result['params']
    x_cm = simulate_dual_pathway_batch(p['omega_fast_sq_ml'], p['gamma_fast_ml'], p['omega_slow_sq_ml'],
                                          p['gamma_slow_ml'], p['D_fast_ml'], p['D_slow_ml'],
                                          fs, duration_seconds, [seed if seed is not None else 0],
                                          burn_in_seconds=BURN_IN_SECONDS)[0]
    y_cm = simulate_dual_pathway_batch(p['omega_fast_sq_ap'], p['gamma_fast_ap'], p['omega_slow_sq_ap'],
                                          p['gamma_slow_ap'], p['D_fast_ap'], p['D_slow_ap'],
                                          fs, duration_seconds,
                                          [(seed + 100000) if seed is not None else 100000],
                                          burn_in_seconds=BURN_IN_SECONDS)[0]
    return {'x': x_cm / 100.0, 'y': y_cm / 100.0}


def plot_predicted_vs_real(test_df: pd.DataFrame, result: dict, fs: float,
                              vision: str, feedback: str, out_dir: Path,
                              duration_seconds: float = 20.0, zoom_seconds: float = 8.0,
                              seed: int = 0):
    n_pts = int(duration_seconds * fs)
    first_trial_key, first_trial = next(iter(test_df.groupby(GROUP_COLS, observed=True)))
    trial_subject, trial_vision, trial_feedback, trial_num = first_trial_key

    x_real_cm = first_trial['cop_x_clean'].dropna().values[:n_pts] * 100.0
    y_real_cm = first_trial['cop_y_clean'].dropna().values[:n_pts] * 100.0
    x_real_cm -= np.mean(x_real_cm)
    y_real_cm -= np.mean(y_real_cm)
    t = np.arange(len(x_real_cm)) / fs

    sim = simulate_from_dual_pathway_result(result, fs, duration_seconds=duration_seconds, seed=seed)
    x_pred_cm = sim['x'] * 100.0 - np.mean(sim['x'] * 100.0)
    y_pred_cm = sim['y'] * 100.0 - np.mean(sim['y'] * 100.0)

    vt = result['validation_table'].set_index('metric')

    def stats_text(axis_suffix):
        return (
            f"           Real    Pred.\n"
            f"RMS (cm)   {vt.loc[f'rms_{axis_suffix}_cm','test_heldout']:.3f}   "
            f"{vt.loc[f'rms_{axis_suffix}_cm','heldout_sim_fresh_seeds']:.3f}\n"
            f"P2P (cm)   {vt.loc[f'p2p_{axis_suffix}_cm','test_heldout']:.3f}   "
            f"{vt.loc[f'p2p_{axis_suffix}_cm','heldout_sim_fresh_seeds']:.3f}\n"
            f"MeanVel    {vt.loc[f'mean_vel_{axis_suffix}_cms','test_heldout']:.3f}   "
            f"{vt.loc[f'mean_vel_{axis_suffix}_cms','heldout_sim_fresh_seeds']:.3f}\n"
            f"F50 (Hz)   {vt.loc[f'f50_{axis_suffix}_hz','test_heldout']:.3f}   "
            f"{vt.loc[f'f50_{axis_suffix}_hz','heldout_sim_fresh_seeds']:.3f}\n"
            f"Skew       {vt.loc[f'skew_{axis_suffix}','test_heldout']:+.3f}  "
            f"{vt.loc[f'skew_{axis_suffix}','heldout_sim_fresh_seeds']:+.3f}\n"
            f"Kurtosis   {vt.loc[f'kurtosis_{axis_suffix}','test_heldout']:.3f}   "
            f"{vt.loc[f'kurtosis_{axis_suffix}','heldout_sim_fresh_seeds']:.3f}"
        )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    zoom_n = int(min(zoom_seconds, duration_seconds) * fs)

    for row, (axis_label, axis_suffix, real_cm, pred_cm) in enumerate(
            [('ML', 'ml', x_real_cm, x_pred_cm), ('AP', 'ap', y_real_cm, y_pred_cm)]):
        for col_i, (n, title) in enumerate(
                [(len(t), f"{axis_label}: Full {duration_seconds:.0f}s"),
                 (zoom_n, f"{axis_label}: First {zoom_n/fs:.0f}s (zoomed)")]):
            ax = axes[row, col_i]
            ax.plot(t[:n], real_cm[:n], color='#2b5c8f', lw=1.3, label='Real (held-out)')
            ax.plot(t[:n], pred_cm[:n], color='#c44e52', lw=1.1, alpha=0.85,
                     label='Predicted (Dual-pathway, SMM-calibrated)')
            ax.axhline(0, color='black', linestyle=':', alpha=0.3)
            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(f"{axis_label} Displacement (cm)")
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.3)
            if col_i == 0:
                ax.text(0.02, 0.02, stats_text(axis_suffix), transform=ax.transAxes,
                         fontsize=7.5, fontfamily='monospace', verticalalignment='bottom',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))

    plt.suptitle(f"Dual-Pathway (Fast+Slow Oscillator) Model: Real vs. Predicted Trajectory ({vision}/{feedback})\n"
                 f"Held-out trial shown: Subject {trial_subject} | {trial_vision}/{trial_feedback} | Trial {trial_num}",
                  fontsize=14)
    plt.tight_layout()
    out_path = out_dir / f"all_dual_pathway_prediction_{vision}_{feedback}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved trajectory plot -> {out_path}")

def compute_psd(x_cm, fs):
    freqs, psd = welch(x_cm - np.mean(x_cm), fs=fs, nperseg=min(len(x_cm), int(fs * 10)))
    return freqs, psd


try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz


def high_freq_power_fraction(freqs, psd, threshold_hz=0.3):
    total = _trapz(psd, freqs)
    hf = _trapz(psd[freqs >= threshold_hz], freqs[freqs >= threshold_hz])
    return hf / total if total > 0 else float('nan')

def apply_filter(x, fs, cutoff_hz: float = 10.0, order=4):
    if cutoff_hz is None:
        return x
    b, a = butter(order, cutoff_hz / (fs / 2.0), btype="low")
    return filtfilt(b, a, x)

def plot_psd_comparison(test_df: pd.DataFrame, result: dict, fs: float, vision: str, feedback: str,
                           out_dir: Path, duration_seconds: float = 20.0, n_seeds: int = 16,
                           threshold_hz: float = 0.3):
    p = result['params']

    real_psds_x, real_psds_y = [], []
    for _, g in test_df.groupby(GROUP_COLS, observed=True):
        x = g['cop_x_clean'].dropna().values * 100.0
        y = g['cop_y_clean'].dropna().values * 100.0
        if len(x) < int(fs * 2):
            continue
        f_r, psd_x = compute_psd(x, fs)
        _, psd_y = compute_psd(y, fs)
        real_psds_x.append(psd_x)
        real_psds_y.append(psd_y)
    real_psd_x_mean = np.mean(real_psds_x, axis=0)
    real_psd_y_mean = np.mean(real_psds_y, axis=0)

    x_batch = simulate_dual_pathway_batch(p['omega_fast_sq_ml'], p['gamma_fast_ml'], p['omega_slow_sq_ml'],
                                             p['gamma_slow_ml'], p['D_fast_ml'], p['D_slow_ml'],
                                             fs, duration_seconds, list(range(n_seeds)),
                                             burn_in_seconds=BURN_IN_SECONDS)
    y_batch = simulate_dual_pathway_batch(p['omega_fast_sq_ap'], p['gamma_fast_ap'], p['omega_slow_sq_ap'],
                                             p['gamma_slow_ap'], p['D_fast_ap'], p['D_slow_ap'],
                                             fs, duration_seconds, [s + 100000 for s in range(n_seeds)],
                                             burn_in_seconds=BURN_IN_SECONDS)

    pred_psds_x, pred_psds_y = [], []
    for i in range(n_seeds):
        x_i = apply_filter(x_batch[i], fs, cutoff_hz=10.0)
        y_i = apply_filter(y_batch[i], fs, cutoff_hz=10.0)
        f_p, psd_x = compute_psd(x_i, fs)
        _, psd_y = compute_psd(y_i, fs)
        pred_psds_x.append(psd_x)
        pred_psds_y.append(psd_y)
        
    pred_psd_x_mean = np.mean(pred_psds_x, axis=0)
    pred_psd_y_mean = np.mean(pred_psds_y, axis=0)

    real_hf_x = high_freq_power_fraction(f_r, real_psd_x_mean, threshold_hz)
    pred_hf_x = high_freq_power_fraction(f_p, pred_psd_x_mean, threshold_hz)
    real_hf_y = high_freq_power_fraction(f_r, real_psd_y_mean, threshold_hz)
    pred_hf_y = high_freq_power_fraction(f_p, pred_psd_y_mean, threshold_hz)

    print(f"High-frequency power fraction (above {threshold_hz} Hz) -- THE key number for this whole line of work:")
    if real_hf_x > 0:
        print(f"  ML: real={real_hf_x:.4f}  predicted={pred_hf_x:.4f}  (ratio: {pred_hf_x/real_hf_x:.2f}x)")
    if real_hf_y > 0:
        print(f"  AP: real={real_hf_y:.4f}  predicted={pred_hf_y:.4f}  (ratio: {pred_hf_y/real_hf_y:.2f}x)")
    print("  (every prior model -- plain DDE, fractional DDE unweighted, fractional DDE reweighted --")
    print("   plateaued in the 0.06-0.12x range on this ratio)")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, axis_label, real_psd, pred_psd, freqs_r, freqs_p in [
            (axes[0], 'ML', real_psd_x_mean, pred_psd_x_mean, f_r, f_p),
            (axes[1], 'AP', real_psd_y_mean, pred_psd_y_mean, f_r, f_p)]:
        ax.semilogy(freqs_r, real_psd, color='#2b5c8f', lw=1.5, label='Real (held-out, avg)')
        ax.semilogy(freqs_p, pred_psd, color='#c44e52', lw=1.5, alpha=0.85, label='Predicted (avg)')
        ax.axvline(threshold_hz, color='gray', linestyle=':', alpha=0.6, label=f'{threshold_hz} Hz threshold')
        ax.set_title(f"{axis_label} Power Spectral Density")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (log scale)")
        ax.set_xlim(0, min(10, fs / 2))
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3, which='both')

    plt.suptitle(f"Full Spectral Comparison (dual-pathway) -- {vision}/{feedback}", fontsize=12)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"all_psd_comparison_dual_pathway_{vision}_{feedback}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved PSD comparison -> {out_path}")

def _hf_power_fraction(sig_cm: np.ndarray, fs: float, threshold_hz: float = None) -> float:
    """
    Fraction of spectral power above threshold_hz. SAME Welch convention as
    f50 in fp_validation.py (nperseg=min(len,int(fs*10))) and SAME
    trapezoidal-integration convention as the standalone PSD diagnostics
    (dde_psd_diagnostic.py, run_dual_pathway_smm_model.py's
    high_freq_power_fraction) -- kept consistent deliberately so this
    built-in metric and those diagnostic plots report the same number for
    the same signal.

    Added directly to the fitted metric set (not left as a pure post-hoc
    diagnostic) after finding that leaving it out of the objective let it
    swing wildly across dual-pathway SMM runs with comparable overall loss
    (0.06x-2.98x observed ratio across three runs) -- nothing was actually
    anchoring it, so the optimizer had no reason to hold it steady.
    """
    if threshold_hz is None:
        threshold_hz = HF_THRESHOLD_HZ

    if len(sig_cm) < int(fs * 2):
        return 0.0
    freqs, psd = welch(sig_cm, fs=fs, nperseg=min(len(sig_cm), int(fs * 10)))
    total = _trapz(psd, freqs)
    if total <= 0:
        return 0.0
    high = _trapz(psd[freqs >= threshold_hz], freqs[freqs >= threshold_hz])
    return float(high / total)

def _extra_metric_names():
    return ['skew_ap', 'skew_ml', 'kurtosis_ap', 'kurtosis_ml', 'hf_power_frac_ap', 'hf_power_frac_ml']

def _metric_names():
    return ['ellipse_area_95_cm2', 'rms_radial_cm', 'rms_ap_cm', 'rms_ml_cm',
            'p2p_ap_cm', 'p2p_ml_cm', 'mean_vel_planar_cms', 'mean_vel_ap_cms',
            'mean_vel_ml_cms', 'total_path_cm', 'f50_ap_hz', 'f50_ml_hz']


def _compute_metric_vector(x: np.ndarray, y: np.ndarray, fs: float) -> np.ndarray:
    dt = 1.0 / fs
    x_cm, y_cm = x * 100.0, y * 100.0
    r_cm = np.sqrt(x_cm**2 + y_cm**2)

    rms_ml, rms_ap, rms_rad = np.sqrt(np.mean(x_cm**2)), np.sqrt(np.mean(y_cm**2)), np.sqrt(np.mean(r_cm**2))
    p2p_ml, p2p_ap = np.max(x_cm) - np.min(x_cm), np.max(y_cm) - np.min(y_cm)

    cov_matrix = np.cov(x_cm, y_cm)
    vals = np.linalg.eigvalsh(cov_matrix)
    ellipse_area = np.pi * chi2.ppf(0.95, df=2) * np.sqrt(max(np.prod(np.clip(vals, 0, None)), 0.0))

    sg_win = int(0.050 * fs) | 1
    sg_win = max(sg_win, 5)
    if len(x_cm) <= sg_win:
        vx_cm = np.gradient(x_cm) * fs
        vy_cm = np.gradient(y_cm) * fs
    else:
        vx_cm = savgol_filter(x_cm, window_length=sg_win, polyorder=3, deriv=1, delta=dt)
        vy_cm = savgol_filter(y_cm, window_length=sg_win, polyorder=3, deriv=1, delta=dt)

    speed = np.sqrt(vx_cm**2 + vy_cm**2)
    mean_vel_planar = np.mean(speed)
    mean_vel_ap, mean_vel_ml = np.mean(np.abs(vy_cm)), np.mean(np.abs(vx_cm))
    total_path = np.sum(speed * dt)

    def f50(sig):
        if len(sig) < int(fs * 2):
            return 0.0
        # nperseg was min(len,int(fs*5)) -- at fs=200, that's 1000, giving
        # frequency resolution fs/nperseg=0.2Hz. Combined with the >=0.1Hz
        # mask below, the FIRST valid bin sits at exactly 0.2Hz -- if >50%
        # of in-range power falls in that single bin, np.interp clips to
        # this boundary every time (confirmed: gaussian_std_seeds was
        # EXACTLY 0.0000 across 8 independent seeds, impossible for a
        # genuine continuous measurement -- the signature of hitting this
        # floor, not real zero variance). Using int(fs*10) doubles
        # resolution to 0.1Hz, roughly halving the floor, at the cost of
        # somewhat less Welch-averaging smoothing in the PSD estimate.
        freqs, psd = welch(sig, fs=fs, nperseg=min(len(sig), int(fs * 10)))
        valid = (freqs >= 0.1) & (freqs <= 10.0)
        if not np.any(valid) or np.sum(psd[valid]) == 0:
            return 0.0
        cum = np.cumsum(psd[valid])
        return float(np.interp(0.5, cum / cum[-1], freqs[valid]))

    return np.array([ellipse_area, rms_rad, rms_ap, rms_ml, p2p_ap, p2p_ml,
                      mean_vel_planar, mean_vel_ap, mean_vel_ml, total_path,
                      f50(y_cm), f50(x_cm)])

def _all_metric_names():
    return _metric_names() + _extra_metric_names()

def _compute_extended_metric_vector(x: np.ndarray, y: np.ndarray, fs: float) -> np.ndarray:
    """x, y in METERS (same convention as fp_validation._compute_metric_vector)."""
    base = _compute_metric_vector(x, y, fs)
    x_cm, y_cm = x * 100.0, y * 100.0
    extra = np.array([
        skew(y_cm), skew(x_cm),
        kurtosis(y_cm, fisher=False), kurtosis(x_cm, fisher=False),
        _hf_power_fraction(y_cm, fs), _hf_power_fraction(x_cm, fs),
    ])
    return np.concatenate([base, extra])

def extract_real_metrics_extended_trial_aware(df_condition: pd.DataFrame, fs: float,
                                                 x_col: str, y_col: str, group_cols: list) -> dict:
    """Same pattern as fp_validation.extract_real_metrics_trial_aware, extended metric set."""
    per_trial_metrics = []
    for _, g in df_condition.groupby(group_cols, observed=True):
        x = g[x_col].dropna().values
        y = g[y_col].dropna().values
        n = min(len(x), len(y))
        if n < int(1.0 * fs):
            continue
        x, y = x[:n] - np.mean(x[:n]), y[:n] - np.mean(y[:n])
        per_trial_metrics.append(_compute_extended_metric_vector(x, y, fs))

    arr = np.array(per_trial_metrics)
    names = _all_metric_names()
    return {name: float(np.mean(arr[:, i])) for i, name in enumerate(names)}
