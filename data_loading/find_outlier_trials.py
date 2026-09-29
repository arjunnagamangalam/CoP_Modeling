"""
Follow-up to verify_cleaned_parquet.py: that script's section 5 found
rms_ap_cm/range_ap_cm maxes far above the rest of the distribution
(75th percentile ~0.27cm but max 5.78cm for rms_ap_cm), and section 6
showed the ML axis got the expected Close>Open sway pattern but AP
reversed it. A handful of outlier trials skewing a ~115-trial mean is
the most likely explanation for a mixed result like that (real axis
mislabeling would flip BOTH axes, not just one).

This script: (1) lists the worst-offending trials by AP sway so we can
see which subjects/conditions they land in, (2) recomputes the Close-
vs-Open comparison with the median instead of the mean (robust to a
handful of outliers) to see if the expected pattern comes back, and
(3) saves trajectory plots of the worst few trials as PNGs so we can
actually look at what's happening in them -- send those back along with
the printed output.

USAGE:
    python3 find_outlier_trials.py /path/to/cleaned_balance_all.parquet [n_plots]
"""
import sys

import numpy as np
import pandas as pd

GROUP_COLS = ["subject_id", "vision", "feedback", "trial"]


def per_trial_stats(g):
    x_cm = (g["cop_x_clean"].values - g["cop_x_clean"].values.mean()) * 100.0
    y_cm = (g["cop_y_clean"].values - g["cop_y_clean"].values.mean()) * 100.0
    return pd.Series({
        "rms_ml_cm": np.sqrt(np.mean(x_cm ** 2)),
        "rms_ap_cm": np.sqrt(np.mean(y_cm ** 2)),
        "range_ml_cm": x_cm.max() - x_cm.min(),
        "range_ap_cm": y_cm.max() - y_cm.min(),
    })


def main(path, n_plots=5):
    print(f"Loading {path} ...")
    df = pd.read_parquet(path)

    stats = df.groupby(GROUP_COLS, observed=True, group_keys=False).apply(per_trial_stats)
    stats = stats.reset_index()

    print("\n" + "=" * 70)
    print(f"TOP {min(20, len(stats))} TRIALS BY rms_ap_cm")
    print("=" * 70)
    worst = stats.sort_values("rms_ap_cm", ascending=False).head(20)
    print(worst.to_string(index=False))

    print("\n" + "=" * 70)
    print("MEAN vs MEDIAN Close-vs-Open comparison (median = robust to outliers)")
    print("=" * 70)
    by_vision_mean = stats.groupby(stats.merge(
        df[GROUP_COLS].drop_duplicates(), on=GROUP_COLS)["vision"]
    )[["rms_ml_cm", "rms_ap_cm"]].mean()
    joined = stats.merge(df[GROUP_COLS].drop_duplicates(), on=GROUP_COLS)
    by_vision_median = joined.groupby("vision", observed=True)[["rms_ml_cm", "rms_ap_cm"]].median()
    print("MEAN:")
    print(joined.groupby("vision", observed=True)[["rms_ml_cm", "rms_ap_cm"]].mean().round(3))
    print("\nMEDIAN:")
    print(by_vision_median.round(3))
    if "Close" in by_vision_median.index and "Open" in by_vision_median.index:
        closed_bigger = (by_vision_median.loc["Close"] > by_vision_median.loc["Open"]).all()
        print(f"\nMedian-based Close > Open on both axes: {closed_bigger}")

    print("\n" + "=" * 70)
    print(f"PLOTTING the {n_plots} worst trials by rms_ap_cm")
    print("=" * 70)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for i, row in worst.head(n_plots).iterrows():
        mask = ((df.subject_id == row.subject_id) & (df.vision == row.vision) &
                 (df.feedback == row.feedback) & (df.trial == row.trial))
        g = df[mask]
        fs = g["sampling_rate_hz"].iloc[0]
        t = np.arange(len(g)) / fs
        x_cm = g["cop_x_clean"].values * 100.0
        y_cm = g["cop_y_clean"].values * 100.0

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(t, x_cm)
        axes[0].set_title("cop_x (ML) vs time")
        axes[0].set_xlabel("s"); axes[0].set_ylabel("cm")
        axes[1].plot(t, y_cm)
        axes[1].set_title("cop_y (AP) vs time")
        axes[1].set_xlabel("s"); axes[1].set_ylabel("cm")
        axes[2].plot(x_cm, y_cm, lw=0.5)
        axes[2].set_title("x-y trajectory")
        axes[2].set_xlabel("ML cm"); axes[2].set_ylabel("AP cm")
        fname = f"outlier_{row.subject_id}_{row.vision}_{row.feedback}_{row.trial}.png"
        fig.suptitle(f"{row.subject_id} {row.vision} {row.feedback} trial={row.trial} "
                      f"(rms_ap={row.rms_ap_cm:.2f}cm, range_ap={row.range_ap_cm:.2f}cm)")
        fig.tight_layout()
        fig.savefig(fname, dpi=100)
        plt.close(fig)
        print(f"  saved {fname}")

    print("\nDone. Send the saved PNGs back so we can look at what's happening "
          "in these trials.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    n_plots = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    main(sys.argv[1], n_plots)