
"""
Plots the "RMS Variability" metric your advisor asked for -- across-trial
SD of RMS, real vs. simulated -- for both axes (ML, AP), across all 4
conditions, matching the definition already used in your GOF write-up
(Table 3: "Across-trial SD of RMS, real vs. simulated").

DOESN'T RECOMPUTE ANYTHING: real_std and sim_std for ml_rms_cm/ap_rms_cm
are already columns in the gof_summary.csv files run_dual_pathway_gof.py
(via dual_pathway_gof.py's evaluate_goodness_of_fit) saves per condition
-- this script just reads those four CSVs and plots them, so the numbers
here are guaranteed consistent with whatever GOF run produced those
files, not a separate/divergent calculation.

WHY SD, NOT VARIANCE OR CV: matches the "Across-trial SD of RMS, real vs.
simulated" framing you settled on earlier for this section, directly in
RMS's own units (cm) -- not normalized, so the bars are directly
comparable to the RMS values themselves.

USAGE:
    python3 plot_rms_variability.py \\
        --gof-dir gof_output \\
        --out rms_variability.png

Expects --gof-dir to contain the standard layout run_dual_pathway_gof.py
writes: gof_<Vision>_<Feedback>/gof_summary.csv, one folder per
condition. If your CSVs live somewhere else (different --out-dir per
config, or renamed), pass --csv four times instead, each as
VISION/FEEDBACK=path/to/gof_summary.csv, e.g.:

    python3 plot_rms_variability.py \\
        --csv Close/Silent=runs/cs/gof_summary.csv \\
        --csv Close/Auditory=runs/ca/gof_summary.csv \\
        --csv Open/Silent=runs/os/gof_summary.csv \\
        --csv Open/Auditory=runs/oa/gof_summary.csv \\
        --out rms_variability.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONFIGS = [("Close", "Silent"), ("Close", "Auditory"), ("Open", "Silent"), ("Open", "Auditory")]
RMS_COLS = {"ML": "ml_rms_cm", "AP": "ap_rms_cm"}

def _load_one_csv(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found -- did run_dual_pathway_gof.py finish for this "
            f"condition, and does --gof-dir (or --csv) point at the right place?")
    df = pd.read_csv(csv_path, index_col="metric")
    out = {}
    for axis, col in RMS_COLS.items():
        if col not in df.index:
            raise KeyError(
                f"{col} not found in {csv_path} -- was this GOF run generated with a "
                f"version of dual_pathway_gof.py that computes ml_rms_cm/ap_rms_cm? "
                f"(it always should, but checking beats a silent KeyError deeper in)")
        out[axis] = {
            "real_std": float(df.loc[col, "real_std"]), "sim_std": float(df.loc[col, "sim_std"]),
            "real_mean": float(df.loc[col, "real_mean"]), "sim_mean": float(df.loc[col, "sim_mean"]),
        }
    return out


def load_rms_variability(gof_dir=None, explicit_csvs=None, configs=CONFIGS):
    """Returns an ordered dict {"Vision/Feedback": {"ML": {...}, "AP": {...}}}.
    Exactly one of gof_dir / explicit_csvs should be given -- explicit_csvs
    is {"Vision/Feedback": csv_path}, already parsed from --csv args."""
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


def plot_rms_variability(data, out_path):
    labels = list(data.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    width = 0.35
    x = np.arange(len(labels))

    for ax, axis in zip(axes, ("ML", "AP")):
        real_vals = [data[lbl][axis]["real_std"] for lbl in labels]
        sim_vals = [data[lbl][axis]["sim_std"] for lbl in labels]
        ax.bar(x - width / 2, real_vals, width, label="Real", color="#2b5c8f")
        ax.bar(x + width / 2, sim_vals, width, label="Simulated", color="#c44e52")

        # Annotate the real/sim ratio above each pair -- under-dispersion
        # (sim_std < real_std) is expected here (simulated trials are all
        # drawn from one calibrated population-average parameter set,
        # real trials span multiple subjects), so the interesting number
        # is BY HOW MUCH, not just the direction.
        for xi, r, s in zip(x, real_vals, sim_vals):
            ratio = r / s if s > 0 else np.nan
            ax.text(xi, max(r, s) * 1.03, f"{ratio:.1f}x", ha="center", fontsize=9, color="dimgray")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(f"{axis} RMS across-trial SD (cm)")
        ax.set_title(f"{axis} axis")
        ax.legend()
        ax.set_ylim(0, max(real_vals + sim_vals) * 1.2)

    fig.suptitle("RMS Variability: across-trial SD of RMS, real vs. simulated")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def print_summary(data):
    print("\nRMS Variability (across-trial SD, real vs. simulated):")
    print(f"  {'condition':16s} {'axis':4s} {'real_std':>10s} {'sim_std':>10s} {'ratio':>8s}")
    for lbl in data:
        for axis in ("ML", "AP"):
            d = data[lbl][axis]
            ratio = d["real_std"] / d["sim_std"] if d["sim_std"] > 0 else float("nan")
            print(f"  {lbl:16s} {axis:4s} {d['real_std']:10.4f} {d['sim_std']:10.4f} {ratio:7.2f}x")


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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gof-dir", default=None,
                    help="Directory containing gof_<Vision>_<Feedback>/gof_summary.csv "
                         "for all 4 conditions (the --out-dir you passed to "
                         "run_dual_pathway_gof.py). Default 'gof_output' if neither "
                         "--gof-dir nor --csv is given.")
    p.add_argument("--csv", action="append", default=None,
                    help="Vision/Feedback=path/to/gof_summary.csv -- repeat 4 times "
                         "(one per condition) to point at CSVs that aren't in the "
                         "standard --gof-dir layout. Overrides --gof-dir if given.")
    p.add_argument("--out", default="rms_variability.png")
    args = p.parse_args()

    explicit_csvs = _parse_csv_arg(args.csv) if args.csv else None
    gof_dir = args.gof_dir if (args.gof_dir or not explicit_csvs) else None
    if not explicit_csvs and not gof_dir:
        gof_dir = "gof_output"

    data = load_rms_variability(gof_dir=gof_dir, explicit_csvs=explicit_csvs)
    print_summary(data)
    plot_rms_variability(data, args.out)


if __name__ == "__main__":
    main()