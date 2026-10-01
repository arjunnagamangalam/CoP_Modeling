"""Plots real-vs-simulated RMS split into low-/high-frequency bands (ML
and AP), across all 4 vision/feedback conditions, reading the rms_lf_cm/
rms_hf_cm figures straight out of each condition's gof_summary.csv
(written by goodness_of_fit_dual_pathway.py). Companion to
plot_rms_variability.py -- this one checks fit broken out by frequency
band rather than overall dispersion.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIGS = [("Close", "Silent"), ("Close", "Auditory"), ("Open", "Silent"), ("Open", "Auditory")]
BAND_COLS = {
    ("ML", "low"): "ml_rms_lf_cm", ("ML", "high"): "ml_rms_hf_cm",
    ("AP", "low"): "ap_rms_lf_cm", ("AP", "high"): "ap_rms_hf_cm",
}


def _load_one_csv(csv_path):
    """Reads one condition's gof_summary.csv and pulls out real/sim
    mean + std + rel_error_pct for each (axis, band) in BAND_COLS,
    returning {(axis, band): {...}}."""

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found -- did run_dual_pathway_gof.py finish for this "
            f"condition, and does --gof-dir (or --csv) point at the right place?")
    df = pd.read_csv(csv_path, index_col="metric")
    out = {}
    for (axis, band), col in BAND_COLS.items():
        if col not in df.index:
            raise KeyError(
                f"{col} not found in {csv_path} -- was this GOF run generated with a "
                f"version of dual_pathway_gof.py that computes the rms_lf_cm/rms_hf_cm "
                f"metrics (the low/high-frequency RMS split added for this analysis)?")
        out[(axis, band)] = {
            "real_mean": float(df.loc[col, "real_mean"]), "real_std": float(df.loc[col, "real_std"]),
            "sim_mean": float(df.loc[col, "sim_mean"]), "sim_std": float(df.loc[col, "sim_std"]),
            "rel_error_pct": float(df.loc[col, "rel_error_pct"]),
        }
    return out


def load_rms_lowhigh(gof_dir=None, explicit_csvs=None, configs=CONFIGS):
    """Returns an ordered dict {"Vision/Feedback": {(axis, band): {...}}}.
    Exactly one of gof_dir / explicit_csvs should be given -- explicit_csvs
    is {"Vision/Feedback": csv_path}."""
    data = {}
    if explicit_csvs:
        for label, csv_path in explicit_csvs.items():
            data[label] = _load_one_csv(csv_path)
    else:
        for vision, feedback in configs:
            label = f"{vision}/{feedback}"
            csv_path = Path(gof_dir) / f"gof_{vision}_{feedback}" / "gof_summary.csv"
            data[label] = _load_one_csv(csv_path)
    return data


def plot_rms_lowhigh(data, out_path, cutoff_hz=0.3):
    """2x2 grid of grouped bar charts (rows = ML/AP, columns = low-/high-
    frequency band), real vs. simulated mean +/- std per condition in
    `data`; saves to out_path. cutoff_hz is used only for axis/title
    labeling (see main's --cutoff-hz help for why)."""

    labels = list(data.keys())
    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for row, axis in enumerate(("ML", "AP")):
        for col, (band, band_label) in enumerate(
                (("low", f"<{cutoff_hz}Hz"), ("high", f">{cutoff_hz}Hz"))):
            ax = axes[row, col]
            entries = [data[lbl][(axis, band)] for lbl in labels]
            real_means = [e["real_mean"] for e in entries]
            real_stds = [e["real_std"] for e in entries]
            sim_means = [e["sim_mean"] for e in entries]
            sim_stds = [e["sim_std"] for e in entries]

            ax.bar(x - width / 2, real_means, width, yerr=real_stds, capsize=4,
                   color="#2b5c8f", alpha=0.85, label="Real")
            ax.bar(x + width / 2, sim_means, width, yerr=sim_stds, capsize=4,
                   color="#c44e52", alpha=0.85, label="Simulated")

            # rel_error_pct annotated per bar pair -- this is the same number
            # already reported in the GOF summary table, just placed on the
            # chart so a reader doesn't have to cross-reference the CSV.
            # for xi, e in zip(x, entries):
            #     y_top = max(e["real_mean"] + e["real_std"], e["sim_mean"] + e["sim_std"])
            #     ax.text(xi, y_top * 1.03, f"{e['rel_error_pct']:.0f}%", ha="center",
            #             fontsize=8, color="dimgray")

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_ylabel(f"RMS {axis} (cm)")
            ax.set_title(f"{axis}: {band_label}")
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(f"RMS by Frequency Band ({cutoff_hz}Hz cutoff), Real vs. Simulated -- ML/AP", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def print_summary(data, cutoff_hz=0.3):
    """Prints a plain-text table of real_mean/sim_mean/rel_error_pct per
    condition, axis, and band -- the console counterpart of
    plot_rms_lowhigh's figure."""

    print(f"\nRMS by frequency band (cutoff={cutoff_hz}Hz), real vs. simulated:")
    print(f"  {'condition':16s} {'axis':4s} {'band':5s} {'real_mean':>10s} "
          f"{'sim_mean':>10s} {'rel_err%':>9s}")
    for lbl in data:
        for axis in ("ML", "AP"):
            for band, band_label in (("low", "low"), ("high", "high")):
                e = data[lbl][(axis, band)]
                print(f"  {lbl:16s} {axis:4s} {band_label:5s} {e['real_mean']:10.4f} "
                      f"{e['sim_mean']:10.4f} {e['rel_error_pct']:8.2f}%")


def _parse_csv_arg(values):
    """--csv Vision/Feedback=path, repeated -- returns {"Vision/Feedback": path}."""

    out = {}
    for v in values:
        if "=" not in v:
            raise argparse.ArgumentTypeError(
                f"--csv expects Vision/Feedback=path/to/gof_summary.csv, got: {v}")
        label, path = v.split("=", 1)
        out[label] = path
    return out


def main():
    """CLI entry point: resolves which gof_summary.csv files to read
    (either --gof-dir's standard per-condition layout, or explicit
    --csv Vision/Feedback=path overrides), then prints the low/high-band
    summary table and saves the comparison plot to --out."""
    
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gof-dir", default=None,
                    help="Directory containing gof_<Vision>_<Feedback>/gof_summary.csv "
                         "for all 4 conditions (the --out-dir you passed to "
                         "run_dual_pathway_gof.py). Default 'gof_output' if neither "
                         "--gof-dir nor --csv is given.")
    p.add_argument("--csv", action="append", default=None,
                    help="Vision/Feedback=path/to/gof_summary.csv -- repeat 4 times "
                         "to point at CSVs outside the standard --gof-dir layout. "
                         "Overrides --gof-dir if given.")
    p.add_argument("--out", default="rms_lowhigh_freq.png")
    p.add_argument("--cutoff-hz", type=float, default=0.3,
                    help="Only used for axis labels/titles -- MUST match "
                         "LOW_HIGH_RMS_CUTOFF_HZ in dual_pathway_gof.py, which is what "
                         "actually determined the filter used to compute the "
                         "rms_lf_cm/rms_hf_cm values in your gof_summary.csv files. "
                         "Changing this flag does not re-filter anything.")
    args = p.parse_args()

    explicit_csvs = _parse_csv_arg(args.csv) if args.csv else None
    gof_dir = args.gof_dir if (args.gof_dir or not explicit_csvs) else None
    if not explicit_csvs and not gof_dir:
        gof_dir = "gof_output"

    data = load_rms_lowhigh(gof_dir=gof_dir, explicit_csvs=explicit_csvs)
    print_summary(data, cutoff_hz=args.cutoff_hz)
    plot_rms_lowhigh(data, args.out, cutoff_hz=args.cutoff_hz)


if __name__ == "__main__":
    main()
