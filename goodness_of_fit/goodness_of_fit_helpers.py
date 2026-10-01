"""
All helpers for goodness_of_fit_dual_pathway.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew as _skew, kurtosis as _kurtosis, ks_2samp
from scipy.signal import welch, butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOW_HIGH_RMS_CUTOFF_HZ = 0.3

def _band_limited_rms(cm, fs, cutoff_hz=LOW_HIGH_RMS_CUTOFF_HZ, order=4):
    """Returns (rms_low, rms_high): RMS (cm) of the low-pass and
    high-pass zero-phase Butterworth components of cm, split at
    cutoff_hz. Not a strict variance decomposition -- real (non-ideal)
    filters overlap somewhat at the cutoff, so rms_low and rms_high
    don't sum exactly back to the broadband RMS -- but this is the
    standard way band-limited RMS is reported in posturography, and
    matches how hf_power_frac is already defined here."""
    if len(cm) < 4 * (order + 1):
        return float("nan"), float("nan")
    nyq = fs / 2.0
    b_lo, a_lo = butter(order, cutoff_hz / nyq, btype="low")
    b_hi, a_hi = butter(order, cutoff_hz / nyq, btype="high")
    low = filtfilt(b_lo, a_lo, cm)
    high = filtfilt(b_hi, a_hi, cm)
    return float(np.sqrt(np.mean(low ** 2))), float(np.sqrt(np.mean(high ** 2)))

def _median_power_freq(sig_cm, fs):
    """F50: the frequency below which half of a signal's Welch PSD power
    falls (median power frequency), via cumulative-sum interpolation."""

    if len(sig_cm) < int(fs * 2):
        return np.nan
    freqs, psd = welch(sig_cm, fs=fs, nperseg=min(len(sig_cm), int(fs * 10)))
    cum = np.cumsum(psd)
    if cum[-1] <= 0:
        return np.nan
    half = cum[-1] / 2.0
    return float(freqs[np.searchsorted(cum, half)])

def dfa_curve(x, fs, order=1, scale_min=None, scale_max_frac=0.25, n_scales=20):
    """Detrended Fluctuation Analysis: integrates x into a cumulative
    profile, then for each window scale n in a geometric grid, detrends
    (polynomial order `order`) each non-overlapping window and returns the
    RMS residual F(n). Returns (scales, F) for log-log slope fitting."""

    x = np.asarray(x, dtype=float)
    N = len(x)
    if scale_min is None:
        scale_min = 4 * (order + 1)
    scale_max = max(scale_min + 1, int(N * scale_max_frac))
    if scale_max <= scale_min:
        return np.array([]), np.array([])
    scales = np.unique(np.geomspace(scale_min, scale_max, n_scales).astype(int))
    profile = np.cumsum(x - x.mean())
    F = np.empty(len(scales))
    for i, n in enumerate(scales):
        n_windows = N // n
        if n_windows < 1:
            F[i] = np.nan
            continue
        local_vars = []
        for w in range(n_windows):
            seg = profile[w * n:(w + 1) * n]
            idx = np.arange(n)
            coeffs = np.polyfit(idx, seg, order)
            trend = np.polyval(coeffs, idx)
            local_vars.append(np.mean((seg - trend) ** 2))
        F[i] = np.sqrt(np.mean(local_vars))
    return scales, F


def dfa_two_regime(x, fs, min_segment_points=3, **kwargs):
    """Fits a two-regime (short-/long-timescale) breakpoint to a DFA curve:
    searches candidate breakpoints for the split that minimizes total
    squared residual of two separate log-log line fits. Falls back to a
    single overall slope (alpha_short == alpha_long) if there aren't enough
    points for two segments. Returns (alpha_short, alpha_long, breakpoint,
    scales, F)."""

    scales, F = dfa_curve(x, fs, **kwargs)
    mask = np.isfinite(F) & (F > 0)
    scales, F = scales[mask], F[mask]
    if len(scales) < 2 * min_segment_points:
        if len(scales) >= 2:
            log_s, log_f = np.log10(scales), np.log10(F)
            alpha = np.polyfit(log_s, log_f, 1)[0]
            return alpha, alpha, None, scales, F
        return np.nan, np.nan, None, scales, F

    log_s, log_f = np.log10(scales), np.log10(F)
    n = len(scales)
    best = None
    for i in range(min_segment_points, n - min_segment_points + 1):
        a1 = np.polyfit(log_s[:i], log_f[:i], 1)
        a2 = np.polyfit(log_s[i - 1:], log_f[i - 1:], 1)
        sse = (np.sum((np.polyval(a1, log_s[:i]) - log_f[:i]) ** 2) +
               np.sum((np.polyval(a2, log_s[i - 1:]) - log_f[i - 1:]) ** 2))
        if best is None or sse < best[0]:
            best = (sse, a1[0], a2[0], scales[i - 1])
    _, alpha_short, alpha_long, breakpoint = best
    return alpha_short, alpha_long, breakpoint, scales, F


def trial_metrics(x_m, y_m, fs):
    """x_m, y_m: 1D arrays, meters. Returns a flat dict with ml_/ap_
    prefixes (x=ML, y=AP, matching this project's established convention
    -- see load_raw_sessions.py's AXIS LABELING note if that needs
    revisiting)."""
    out = {}
    for axis, sig_m in (("ml", x_m), ("ap", y_m)):
        cm = (sig_m - sig_m.mean()) * 100.0
        dt = 1.0 / fs
        vel = np.diff(cm) / dt
        alpha_short, alpha_long, _, _, _ = dfa_two_regime(cm, fs)
        rms_lf, rms_hf = _band_limited_rms(cm, fs)
        out[f"{axis}_rms_cm"] = float(np.sqrt(np.mean(cm ** 2)))
        out[f"{axis}_rms_lf_cm"] = rms_lf
        out[f"{axis}_rms_hf_cm"] = rms_hf
        out[f"{axis}_p2p_cm"] = float(cm.max() - cm.min())
        out[f"{axis}_meanvel_cms"] = float(np.mean(np.abs(vel)))
        out[f"{axis}_f50_hz"] = _median_power_freq(cm, fs)
        out[f"{axis}_skew"] = float(_skew(cm))
        out[f"{axis}_kurtosis"] = float(_kurtosis(cm, fisher=False))
        out[f"{axis}_dfa_alpha_short"] = float(alpha_short)
        out[f"{axis}_dfa_alpha_long"] = float(alpha_long)
    return out

def metrics_table(trials, fs):
    """trials: list of (x_m, y_m) tuples. Returns a DataFrame, one row
    per trial."""
    rows = [trial_metrics(x, y, fs) for x, y in trials]
    return pd.DataFrame(rows)


def plot_metric_comparison(summary, out_path, label=""):
    """Bar chart of real vs. simulated mean +/- std for every metric in
    `summary` (goodness_of_fit_report's output); saves to out_path."""

    metrics = summary.index.tolist()
    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 0.5), 5))
    xpos = np.arange(len(metrics))
    ax.bar(xpos - 0.18, summary["real_mean"], width=0.36, yerr=summary["real_std"],
           label="Real", capsize=3)
    ax.bar(xpos + 0.18, summary["sim_mean"], width=0.36, yerr=summary["sim_std"],
           label="Simulated", capsize=3)
    ax.set_xticks(xpos)
    ax.set_xticklabels(metrics, rotation=60, ha="right", fontsize=8)
    ax.set_title(f"Metric comparison (mean +/- std across trials) {label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    import matplotlib.pyplot as _plt
    _plt.close(fig)


def plot_metric_distributions(real_df, sim_df, out_path, label=""):
    """Grid of per-metric histograms (real vs. simulated, one subplot per
    shared column of real_df/sim_df); saves to out_path."""

    cols = [c for c in real_df.columns if real_df[c].notna().sum() > 1]
    ncols = 4
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for i, col in enumerate(cols):
        ax = axes[i]
        ax.hist(real_df[col].dropna(), bins=15, alpha=0.5, density=True, label="Real")
        ax.hist(sim_df[col].dropna(), bins=15, alpha=0.5, density=True, label="Simulated")
        ax.set_title(col, fontsize=9)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(len(cols), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Per-trial metric distributions, real vs simulated {label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_psd_comparison(real_trials, sim_trials, fs, out_path, label=""):
    """Mean PSD (ML and AP), real vs. simulated, with a geometric (log-space)
    mean +/- 1 std band across trials -- see the inline comment below for
    why geometric rather than arithmetic banding is used. Saves to out_path."""

    def psd_stack(trials, axis_idx):
        psds, freqs = [], None
        for tr in trials:
            sig = (tr[axis_idx] - tr[axis_idx].mean()) * 100.0
            f, p = welch(sig, fs=fs, nperseg=min(len(sig), int(fs * 10)))
            freqs = f
            psds.append(p)
        return freqs, np.array(psds)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, axis_idx, name in ((axes[0], 0, "ML"), (axes[1], 1, "AP")):
        f_r, p_r = psd_stack(real_trials, axis_idx)
        f_s, p_s = psd_stack(sim_trials, axis_idx)
        for f, p, color, lbl in ((f_r, p_r, "tab:blue", "Real"), (f_s, p_s, "tab:red", "Simulated")):
            # PSD across trials is heavy-tailed and strictly positive, so an
            # arithmetic mean +/- std band routinely goes negative (a trial
            # can sit near-zero at a given frequency bin) and gets clipped to
            # the 1e-12 floor -- on a log axis that reads as a plunge to the
            # bottom of the chart, an artifact of the band, not the data.
            # A geometric (log-space) mean +/- std avoids this and is the
            # standard way to band spectral estimates.
            log_p = np.log10(np.clip(p, 1e-12, None))
            mean_log, std_log = log_p.mean(axis=0), log_p.std(axis=0)
            mean_p = 10 ** mean_log
            ax.plot(f, mean_p, color=color, label=lbl)
            ax.fill_between(f, 10 ** (mean_log - std_log), 10 ** (mean_log + std_log),
                             color=color, alpha=0.2)
        ax.set_yscale("log")
        ax.set_xlim(0, 10)
        ax.set_title(f"{name} PSD (mean +/- 1 std across trials)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (log scale)")
        ax.legend()
    fig.suptitle(f"PSD comparison {label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_dfa_curves(real_trials, sim_trials, fs, out_path, label=""):
    """Mean DFA fluctuation curve (ML and AP), real vs. simulated, each
    trial's curve interpolated onto a common scale grid before averaging;
    saves to out_path."""

    def avg_curve(trials, axis_idx):
        curves = []
        common_scales = None
        for tr in trials:
            sig = (tr[axis_idx] - tr[axis_idx].mean()) * 100.0
            scales, F = dfa_curve(sig, fs)
            if len(scales) == 0:
                continue
            curves.append((scales, F))
        if not curves:
            return None, None
        ref_scales = curves[0][0]
        aligned = [np.interp(ref_scales, s, f) for s, f in curves]
        return ref_scales, np.mean(aligned, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, axis_idx, name in ((axes[0], 0, "ML"), (axes[1], 1, "AP")):
        s_r, f_r = avg_curve(real_trials, axis_idx)
        s_s, f_s = avg_curve(sim_trials, axis_idx)
        if s_r is not None:
            ax.loglog(s_r / fs, f_r, "o-", color="tab:blue", label="Real")
        if s_s is not None:
            ax.loglog(s_s / fs, f_s, "o-", color="tab:red", label="Simulated")
        ax.set_title(f"{name} DFA: F(n) vs window size")
        ax.set_xlabel("window size (s)")
        ax.set_ylabel("F(n) (cm)")
        ax.legend()
    fig.suptitle(f"DFA fluctuation curves (avg across trials) {label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_example_trial(real_trial, sim_trial, fs, out_path, label=""):
    """Overlays one real and one simulated example trial (ML and AP) --
    purely qualitative illustration, not a GOF metric itself. Saves to
    out_path."""
    
    t = np.arange(len(real_trial[0])) / fs
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, axis_idx, name in ((axes[0], 0, "ML"), (axes[1], 1, "AP")):
        real_cm = (real_trial[axis_idx] - real_trial[axis_idx].mean()) * 100.0
        sim_cm = (sim_trial[axis_idx] - sim_trial[axis_idx].mean()) * 100.0
        t_sim = np.arange(len(sim_cm)) / fs
        ax.plot(t, real_cm, color="tab:blue", label="Real (example)")
        ax.plot(t_sim, sim_cm, color="tab:red", label="Simulated (example)", alpha=0.8)
        ax.set_title(f"{name} example trajectory (qualitative only)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("cm")
        ax.legend()
    fig.suptitle(f"Example single-trial overlay -- illustrative, NOT the GOF score {label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

def goodness_of_fit_report(real_df, sim_df, weights=None, exclude_from_composite=None):
    """Returns (summary_df, composite_score). summary_df has one row per
    metric: real_mean, real_std, sim_mean, sim_std, rel_error_pct,
    ks_stat, ks_pvalue, composite_weight. composite_score is the
    (optionally weighted) mean of rel_error_pct across metrics -- lower
    is better, 0 = perfect ensemble-mean match (says nothing about
    distribution SHAPE on its own -- check ks_pvalue for that: a low
    p-value means the real and simulated distributions for that metric
    are distinguishable even if their means happen to be close).

    exclude_from_composite: metric names (e.g. "ap_skew") to still
    report in the table -- full rel_error_pct, KS test, everything -- but
    weight at 0.0 when computing composite_score. Use this for metrics
    your calibration objective doesn't target (e.g. skew/kurtosis, when
    the SMM run used weights like RELAXED_METRIC_WEIGHTS to zero them
    out). An untargeted metric whose real/sim means both sit near zero
    (skew especially) can show rel_error_pct in the hundreds of percent
    from a trivial absolute difference -- that's not a sign the fit got
    worse, it's a small-denominator artifact, and left in an unweighted
    composite it can swamp real signal from the metrics that were
    actually optimized for. This never drops or hides the metric --
    only removes its vote in the single composite number."""

    rows = []
    for col in real_df.columns:
        real_vals = real_df[col].dropna().values
        sim_vals = sim_df[col].dropna().values
        if len(real_vals) < 2 or len(sim_vals) < 2:
            continue
        real_mean, sim_mean = real_vals.mean(), sim_vals.mean()
        rel_err = abs(sim_mean - real_mean) / (abs(real_mean) + 1e-9) * 100.0
        ks_stat, ks_p = ks_2samp(real_vals, sim_vals)
        rows.append({
            "metric": col, "real_mean": real_mean, "real_std": real_vals.std(),
            "sim_mean": sim_mean, "sim_std": sim_vals.std(),
            "rel_error_pct": rel_err, "ks_stat": ks_stat, "ks_pvalue": ks_p,
        })
    summary = pd.DataFrame(rows).set_index("metric")
    if weights:
        w = pd.Series(weights).reindex(summary.index).fillna(1.0)
    else:
        w = pd.Series(1.0, index=summary.index)
    if exclude_from_composite:
        present = [m for m in exclude_from_composite if m in summary.index]
        missing = [m for m in exclude_from_composite if m not in summary.index]
        if missing:
            print(f"WARNING: --exclude-from-composite named metric(s) not present in "
                  f"this summary, ignoring: {missing}")
        w = w.copy()
        w.loc[present] = 0.0
    summary["composite_weight"] = w
    composite = float((summary["rel_error_pct"] * w).sum() / w.sum())
    return summary, composite

def evaluate_goodness_of_fit(real_trials, sim_trials, fs, out_dir, label="", weights=None,
                              exclude_from_composite=None):
    """real_trials, sim_trials: lists of (x_m, y_m) tuples, meters, same
    fs. Runs the full report + all plots, saves to out_dir, returns
    (summary_df, composite_score). exclude_from_composite: see
    goodness_of_fit_report's docstring -- metrics still fully reported,
    just weighted 0 in the single composite number."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Real trials: {len(real_trials)}, simulated trials: {len(sim_trials)}, fs={fs}Hz")

    real_df = metrics_table(real_trials, fs)
    sim_df = metrics_table(sim_trials, fs)

    summary, composite = goodness_of_fit_report(
        real_df, sim_df, weights=weights, exclude_from_composite=exclude_from_composite)
    print("\n" + "=" * 100)
    print(f"GOODNESS-OF-FIT SUMMARY {label}")
    print("=" * 100)
    print(summary.round(4).to_string())
    if exclude_from_composite:
        excluded_present = [m for m in exclude_from_composite if m in summary.index]
        if excluded_present:
            print(f"\nExcluded from composite score (still fully reported above, "
                  f"composite_weight=0): {excluded_present}")
    composite_note = ("unweighted mean across metrics" if not (weights or exclude_from_composite)
                       else "see composite_weight column for per-metric weights used")
    print(f"\nComposite relative-error score (lower is better, {composite_note}): {composite:.2f}%")
    n_ks_significant = (summary["ks_pvalue"] < 0.05).sum()
    print(f"Metrics where real vs simulated DISTRIBUTIONS differ significantly "
          f"(KS test, p<0.05): {n_ks_significant} of {len(summary)}")
    if n_ks_significant:
        print("  " + ", ".join(summary[summary['ks_pvalue'] < 0.05].index.tolist()))

    summary.to_csv(out_dir / "gof_summary.csv")
    print(f"\nsaved -> {out_dir / 'gof_summary.csv'}")

    plot_metric_comparison(summary, out_dir / "metric_comparison.png", label=label)
    plot_metric_distributions(real_df, sim_df, out_dir / "metric_distributions.png", label=label)
    # plot_psd_comparison(real_trials, sim_trials, fs, out_dir / "psd_comparison.png", label=label)
    plot_dfa_curves(real_trials, sim_trials, fs, out_dir / "dfa_curves.png", label=label)
    # plot_example_trial(real_trials[0], sim_trials[0], fs, out_dir / "example_trial.png", label=label)
    print(f"saved plots -> {out_dir}/*.png")

    return summary, composite