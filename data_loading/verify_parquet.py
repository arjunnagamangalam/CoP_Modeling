"""
Sanity-checks a freshly built cleaned_balance_all.parquet -- run this on
Oscar against the file you just produced. It can't prove the CoP values
are "true" in an absolute sense (no independent ground truth exists),
but it checks everything that WOULD be wrong if something upstream had
gone wrong: schema, units, missing data, and the two physiological
patterns this kind of postural-sway data is expected to show. Share the
printed output back and we can look at anything that flags together.

USAGE:
    python3 verify_cleaned_parquet.py /path/to/cleaned_balance_all.parquet
"""
import sys

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = {
    "subject_id": "object", "vision": "object", "feedback": "object",
    "trial": "object", "cop_x_clean": "float64", "cop_y_clean": "float64",
    "sampling_rate_hz": "float64",
}
GROUP_COLS = ["subject_id", "vision", "feedback", "trial"]


def _section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def check_schema(df):
    _section("1. SCHEMA")
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(EXPECTED_COLUMNS)
    if missing:
        print(f"  MISSING columns: {missing}")
    if extra:
        print(f"  extra columns present (not necessarily a problem): {extra}")
    for col, expected_dtype in EXPECTED_COLUMNS.items():
        if col not in df.columns:
            continue
        actual = str(df[col].dtype)
        ok = actual == expected_dtype or (expected_dtype == "object" and actual in ("object", "string", "str"))
        print(f"  {col}: {actual} {'OK' if ok else f'expected {expected_dtype} -- CHECK THIS'}")
    print(f"  total rows: {len(df):,}")


def check_missing_and_finite(df):
    _section("2. MISSING / NON-FINITE VALUES")
    for col in ["cop_x_clean", "cop_y_clean", "sampling_rate_hz"]:
        n_nan = df[col].isna().sum()
        n_inf = np.isinf(df[col].values).sum()
        flag = "  <-- CHECK THIS" if (n_nan or n_inf) else ""
        print(f"  {col}: {n_nan} NaN, {n_inf} inf{flag}")


def check_trial_structure(df):
    _section("3. TRIAL / SAMPLE-COUNT STRUCTURE")
    inventory = df[GROUP_COLS].drop_duplicates()
    print(f"  {inventory['subject_id'].nunique()} subjects, {len(inventory)} trials total")
    print(inventory.groupby(["vision", "feedback"], observed=True).size().rename("n_trials"))

    lens = df.groupby(GROUP_COLS, observed=True).size()
    print(f"\n  samples per trial: min={lens.min()}, max={lens.max()}, "
          f"median={lens.median():.0f}")
    ragged = lens[lens != lens.median()]
    if len(ragged):
        print(f"  {len(ragged)} trial(s) have a sample count different from the median "
              f"(could be legit if you have varying-duration trials, or could be a parsing "
              f"issue -- worth spot-checking a couple):")
        print(ragged.head(10))
    else:
        print("  all trials have the same sample count -- OK")

    fs_vals = df["sampling_rate_hz"].unique()
    print(f"\n  sampling_rate_hz values present: {fs_vals}")


def check_baseline(df):
    _section("4. BASELINE (first sample should be exactly 0 per trial)")
    firsts = df.groupby(GROUP_COLS, observed=True).first()
    bad_x = (firsts["cop_x_clean"].abs() > 1e-12).sum()
    bad_y = (firsts["cop_y_clean"].abs() > 1e-12).sum()
    if bad_x or bad_y:
        print(f"  {bad_x} trial(s) with nonzero first cop_x_clean, "
              f"{bad_y} with nonzero first cop_y_clean -- CHECK THIS "
              f"(the loader is supposed to zero every trial's own first sample).")
    else:
        print(f"  every trial's first sample is exactly 0 on both axes -- OK, "
              f"matches the loader's baseline-subtraction step.")


def check_units_and_magnitude(df):
    _section("5. UNITS / MAGNITUDE (in meters; converted to cm below for readability)")
    print("  If this were still in millimeters, these numbers would be ~1000x too big.")
    print("  If it were accidentally double-converted, they'd be ~1000x too small.")
    print("  Typical quiet-standing CoP sway in the literature: RMS roughly 0.3-1.5 cm, "
          "path range (max-min) roughly 1-6 cm per axis. Values wildly outside that "
          "(sub-millimeter, or tens of cm) are worth investigating before trusting the data.\n")

    def per_trial_stats(g):
        x_cm = (g["cop_x_clean"].values - g["cop_x_clean"].values.mean()) * 100.0
        y_cm = (g["cop_y_clean"].values - g["cop_y_clean"].values.mean()) * 100.0
        return pd.Series({
            "rms_ml_cm": np.sqrt(np.mean(x_cm ** 2)),
            "rms_ap_cm": np.sqrt(np.mean(y_cm ** 2)),
            "range_ml_cm": x_cm.max() - x_cm.min(),
            "range_ap_cm": y_cm.max() - y_cm.min(),
        })

    stats = df.groupby(GROUP_COLS, observed=True, group_keys=False).apply(per_trial_stats)
    print(stats.describe().loc[["min", "25%", "50%", "75%", "max"]].round(3))

    suspicious = stats[(stats["rms_ml_cm"] < 0.01) | (stats["rms_ml_cm"] > 10) |
                        (stats["rms_ap_cm"] < 0.01) | (stats["rms_ap_cm"] > 10)]
    if len(suspicious):
        print(f"\n  {len(suspicious)} trial(s) with RMS sway outside a plausible 0.01-10cm "
              f"range -- worth pulling up individually and looking at the raw trajectory:")
        print(suspicious.head(10))
    else:
        print("\n  no trials with wildly implausible RMS sway -- OK")

    return stats


def check_physiological_pattern(stats, df):
    _section("6. DIRECTIONAL SANITY CHECK (not a hard pass/fail, just a known tendency)")
    print("  Well-established finding in postural-control studies: removing vision "
          "(eyes CLOSED) increases sway versus eyes OPEN, because the subject loses a "
          "stabilizing sensory input. If your data shows the OPPOSITE on average, that's "
          "not necessarily wrong, but it's surprising enough to double check axis/condition "
          "labeling before trusting downstream results.\n")
    joined = stats.reset_index().merge(df[GROUP_COLS].drop_duplicates(), on=GROUP_COLS)
    by_vision = joined.groupby("vision", observed=True)[["rms_ml_cm", "rms_ap_cm"]].mean()
    print(by_vision.round(3))
    if "Close" in by_vision.index and "Open" in by_vision.index:
        closed_bigger = (by_vision.loc["Close"] > by_vision.loc["Open"]).all()
        print(f"\n  Close > Open on both axes: {closed_bigger} "
              f"{'(matches the expected pattern)' if closed_bigger else '(CHECK -- unexpected)'}")

    by_feedback = joined.groupby("feedback", observed=True)[["rms_ml_cm", "rms_ap_cm"]].mean()
    print(f"\n  by feedback condition (auditory feedback commonly reduces sway "
          f"somewhat vs no/silent feedback, but this is a weaker/less universal "
          f"effect than the vision one above -- informational only):")
    print(by_feedback.round(3))


def main(path):
    print(f"Loading {path} ...")
    df = pd.read_parquet(path)

    check_schema(df)
    check_missing_and_finite(df)
    check_trial_structure(df)
    check_baseline(df)
    stats = check_units_and_magnitude(df)
    check_physiological_pattern(stats, df)

    _section("DONE")
    print("Share this output and we can look at anything flagged above together.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])