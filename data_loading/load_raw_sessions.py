"""
Builds cleaned_balance_all.parquet from raw per-trial force-plate CSVs
(motion-capture-suite ASCII export: a "Devices" block of two force plates'
channels, including a vendor-computed CoP in mm, followed by a
"Trajectories" block that this loader ignores), after the original cleaned
parquet and loading script were both lost.

Per file: parses subject/vision/feedback/trial from the filename
({subject}_{vision}_{feedback}_{trial}.csv; vision in Open/Close, feedback
in Silent/Auditory -- Tactile/Combined files are skipped, and calibration
files like "S01 Cal 01.csv" are detected via CALIBRATION_RE and excluded).
Picks whichever force plate actually carries the subject's weight (by mean
|Fz|); if both plates show a real load (observed for some subjects, not
all), falls back to the dominant plate rather than dropping the trial --
flagged and counted in load_all_sessions' printed report rather than done
silently. Converts CoP (Cx/Cy, mm) to meters, applies a low-pass filter,
then subtracts each trial's own first sample so every cleaned trial starts
at exactly 0 (confirmed against the old cached parquet's own convention).

Two unconfirmed defaults, called out here rather than asserted as fact:
the 10 Hz 4th-order zero-phase Butterworth filter cutoff (the original
pipeline's actual cutoff is unknown -- pass a different value via the CLI
or filter_cutoff_hz= and compare if downstream PSD/DFA results look off),
and the Cx->ML/Cy->AP axis mapping (conventional default, not confirmed
against lab mounting notes -- doesn't affect calibration correctness,
only which axis label ends up in reports/plots).

USAGE:
    python3 load_raw_sessions.py /path/to/raw/session/files [output.parquet] [cutoff_hz]

    cutoff_hz defaults to 10 (Hz). Pass "none" to disable filtering and
    get the raw (baseline-subtracted only) signal.
"""

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


FILENAME_RE = re.compile(
    r"^(?P<subject>S\d+)_(?P<vision>Open|Close)_(?P<feedback>Silent|Auditory)_(?P<trial>\d+)\.csv$",
    re.IGNORECASE
)

# Calibration trials use a different naming convention entirely (example
# given: "S01 Cal 01") and must be excluded, not loaded as a real trial --
# matched and filtered out BEFORE FILENAME_RE ever sees the file. Accepts
# space or underscore as separator and "Cal"/"Calibration", since only one
# example filename was confirmed -- see CALIBRATION TRIALS in this
# module's docstring if a file isn't being caught.
CALIBRATION_RE = re.compile(
    r"^(?P<subject>S\d+)[ _]Cal(?:ibration)?[ _](?P<trial>\d+)\.csv$",
    re.IGNORECASE,
)


def is_calibration_file(path: Path):
    """True if `path`'s filename matches the calibration-trial naming
    convention (CALIBRATION_RE), e.g. "S01 Cal 01.csv" -- such files are
    excluded from the cleaned dataset entirely, before FILENAME_RE ever
    runs on them."""

    return bool(CALIBRATION_RE.match(path.name))

EXPECTED_DURATION_SECONDS = 20.0
EXPECTED_FS = 200.0
# how far a plate's own no-load Cx/Cy is allowed to sit from a "real"
# reading before we call it inactive -- Fz is the primary active/inactive
# signal (near-exactly 0.0 N for an unloaded plate, confirmed in the
# sample file), this is just a defensive secondary check.
MIN_ACTIVE_MEAN_ABS_FZ = 5.0  # newtons -- comfortably below a real ~590N
                                # body-weight load, comfortably above the
                                # exact-0.0N seen for an unloaded plate.


def parse_filename(path: Path):
    """Parses a real (non-calibration) trial filename against
    FILENAME_RE and returns (subject_id, vision, feedback, trial) --
    vision/feedback are capitalized to match this project's Open/Close
    and Silent/Auditory conventions. Raises ValueError (not silently
    skipped) if the filename doesn't match, naming the exact expected
    pattern so a genuine naming-convention mismatch is easy to diagnose."""

    m = FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(
            f"Filename doesn't match the expected "
            f"'{{subject}}_{{vision}}_{{feedback}}_{{trial}}.csv' pattern: {path.name}\n"
            f"  If your files use a different convention, edit FILENAME_RE at the top "
            f"of this script rather than renaming hundreds of files.")
    g = m.groupdict()
    return g["subject"], g["vision"].capitalize(), g["feedback"].capitalize(), g["trial"]


def _read_devices_block(path: Path):
    """Returns (fs_hz, data_array) for just the Devices (force-plate)
    block, stopping before the Trajectories block. data_array has one row
    per sample, columns [Frame, SubFrame, Fx2,Fy2,Fz2,Mx2,My2,Mz2,Cx2,Cy2,Cz2,
    Fx1,Fy1,Fz1,Mx1,My1,Mz1,Cx1,Cy1,Cz1] matching the file's own column order
    (device group #2 first, then #1 -- this is the file's order, not a
    judgment about which one is "real").
    """
    with open(path, encoding="utf-8-sig") as f:
        lines = f.readlines()

    if not lines or lines[0].strip() != "Devices":
        raise ValueError(f"{path.name}: expected 'Devices' as the first line, "
                          f"got {lines[0].strip() if lines else '(empty file)'}")
    fs_hz = float(lines[1].strip())

    try:
        traj_idx = next(i for i, l in enumerate(lines) if l.strip() == "Trajectories")
    except StopIteration:
        raise ValueError(f"{path.name}: no 'Trajectories' marker found -- "
                          f"can't safely bound the CoP data block. Inspect this file by hand.")

    # data rows: after the 5-line header block (Devices / rate / group
    # names / column names / units), up to (not including) the blank
    # separator line before "Trajectories".
    data_lines = [l for l in lines[5:traj_idx] if l.strip()]
    rows = [l.strip().split(",") for l in data_lines]
    try:
        arr = np.array(rows, dtype=float)
    except ValueError as e:
        raise ValueError(f"{path.name}: non-numeric value in the Devices data block "
                          f"(corrupted file?): {e}")
    return fs_hz, arr


def load_single_trial_csv(path: Path, filter_cutoff_hz=10.0):
    """Returns dict with cop_x_m, cop_y_m (1D arrays, meters, mean NOT
    removed -- downstream code in this project demeans per trial itself),
    fs_hz, and dominant_plate_fallback (bool, see TWO-PLATE FALLBACK in
    this module's docstring), or raises ValueError with a specific reason
    if the file doesn't check out.

    filter_cutoff_hz: passed straight to apply_filter() -- see LOW-PASS
    FILTER in this module's docstring. Default 10Hz is an unconfirmed
    guess, not your original pipeline's actual value. Pass None to skip
    filtering entirely."""
    fs_hz, arr = _read_devices_block(path)

    # device #2 columns: Fz=idx4, Cx=idx8, Cy=idx9
    # device #1 columns: Fz=idx13, Cx=idx17, Cy=idx18
    fz2, cx2, cy2 = arr[:, 4], arr[:, 8], arr[:, 9]
    fz1, cx1, cy1 = arr[:, 13], arr[:, 17], arr[:, 18]

    mean_fz1, mean_fz2 = np.abs(fz1).mean(), np.abs(fz2).mean()
    active2 = mean_fz2 > MIN_ACTIVE_MEAN_ABS_FZ
    active1 = mean_fz1 > MIN_ACTIVE_MEAN_ABS_FZ

    dominant_plate_fallback = False
    if active1 and active2:
        # TWO-PLATE FALLBACK (your call, "no preference" -- this is my best
        # judgment, not a confirmed fact -- see this module's docstring):
        # take whichever plate carries the larger load as the real one,
        # same rule as the clean single-plate case. Flagged via the
        # returned dominant_plate_fallback so load_all_sessions can count
        # and report how many trials relied on this.
        if mean_fz1 >= mean_fz2:
            cx, cy = cx1, cy1
        else:
            cx, cy = cx2, cy2
        dominant_plate_fallback = True
    elif active1:
        cx, cy = cx1, cy1
    elif active2:
        cx, cy = cx2, cy2
    else:
        raise ValueError(f"{path.name}: neither force plate shows a real load "
                          f"(mean|Fz1|={mean_fz1:.2f}N, mean|Fz2|={mean_fz2:.2f}N) "
                          f"-- likely an empty/failed trial.")

    n = len(cx)
    expected_n = int(round(EXPECTED_DURATION_SECONDS * fs_hz))
    if n != expected_n:
        warnings.warn(f"{path.name}: {n} samples at {fs_hz}Hz = {n/fs_hz:.2f}s, "
                       f"expected {EXPECTED_DURATION_SECONDS}s ({expected_n} samples). Keeping as-is.")

    x_m = cx / 1000.0   # mm -> m
    y_m = cy / 1000.0

    # low-pass filter BEFORE baseline subtraction -- see LOW-PASS FILTER
    # and the ordering note under BASELINE SUBTRACTION in this module's
    # docstring for why this order matters.
    x_m = apply_filter(x_m, fs_hz, filter_cutoff_hz)
    y_m = apply_filter(y_m, fs_hz, filter_cutoff_hz)

    # per-trial baseline subtraction: shift so the trial's OWN first sample
    # is exactly 0 -- confirmed with you (see BASELINE SUBTRACTION in this
    # module's docstring), not an assumption.
    x_m = x_m - x_m[0]
    y_m = y_m - y_m[0]

    return {
        "cop_x_m": x_m,
        "cop_y_m": y_m,
        "fs_hz": fs_hz,
        "dominant_plate_fallback": dominant_plate_fallback,
    }


def apply_filter(x, fs, cutoff_hz=None, order=4):
    """Zero-phase (filtfilt) 4th-order Butterworth low-pass, applied only
    if cutoff_hz is not None -- see LOW-PASS FILTER in this module's
    docstring for the current default (10Hz) and why it's an unconfirmed
    guess, not a verified match to whatever the original pipeline used.
    """
    if cutoff_hz is None:
        return x
    b, a = butter(order, cutoff_hz / (fs / 2.0), btype="low")
    return filtfilt(b, a, x)


def load_all_sessions(raw_dir, out_path=None, filter_cutoff_hz=10.0):
    """filter_cutoff_hz: passed to load_single_trial_csv -> apply_filter
    for every trial (see LOW-PASS FILTER in this module's docstring).
    Default 10Hz. Pass None to disable filtering entirely."""
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No .csv files found under {raw_dir} (searched recursively)")

    records = []
    skipped = []
    excluded_calibration = []
    dominant_plate_fallback_files = []
    for path in files:
        if is_calibration_file(path):
            excluded_calibration.append(path.name)
            continue
        try:
            subject_id, vision, feedback, trial = parse_filename(path)
        except ValueError as e:
            skipped.append((path.name, str(e)))
            continue
        try:
            trial_data = load_single_trial_csv(path, filter_cutoff_hz=filter_cutoff_hz)
        except ValueError as e:
            skipped.append((path.name, str(e)))
            continue

        if trial_data["dominant_plate_fallback"]:
            dominant_plate_fallback_files.append(path.name)

        records.append(pd.DataFrame({
            "subject_id": subject_id,
            "vision": vision,
            "feedback": feedback,
            "trial": trial,
            "cop_x_clean": trial_data["cop_x_m"],
            "cop_y_clean": trial_data["cop_y_m"],
            "sampling_rate_hz": trial_data["fs_hz"],
        }))

    if not records:
        raise RuntimeError(f"No files loaded successfully out of {len(files)} found. "
                            f"First few failures:\n" + "\n".join(f"  {n}: {r}" for n, r in skipped[:5]))

    df = pd.concat(records, ignore_index=True)

    if filter_cutoff_hz is None:
        print("Filtering: OFF (raw signal, baseline-subtracted only)")
    else:
        print(f"Filtering: {filter_cutoff_hz}Hz, 4th-order Butterworth, zero-phase "
              f"-- UNCONFIRMED default, not verified against your original pipeline's "
              f"actual cutoff (see LOW-PASS FILTER in this script's docstring)")
    print(f"Loaded {len(records)} of {len(files)} files found under {raw_dir}")
    if dominant_plate_fallback_files:
        print(f"\n{len(dominant_plate_fallback_files)} file(s) had BOTH plates showing a real load -- "
              f"used the dominant (larger mean|Fz|) plate and ignored the other (see TWO-PLATE "
              f"FALLBACK in this script's docstring; this is my best-guess handling, not confirmed "
              f"with you -- spot-check a few of these trials before trusting them):")
        for name in dominant_plate_fallback_files[:10]:
            print(f"  {name}")
        if len(dominant_plate_fallback_files) > 10:
            print(f"  ... and {len(dominant_plate_fallback_files) - 10} more")
    print(f"\nExcluded {len(excluded_calibration)} calibration trial(s) "
          f"(matched CALIBRATION_RE -- e.g. \"S01 Cal 01\")")
    if excluded_calibration:
        for name in excluded_calibration:
            print(f"  {name}")
    if skipped:
        print(f"\n{len(skipped)} file(s) skipped:")
        for name, reason in skipped:
            print(f"  {name}: {reason}")

    inventory = df[["subject_id", "vision", "feedback", "trial"]].drop_duplicates()
    print(f"\n{inventory['subject_id'].nunique()} subjects, "
          f"{len(inventory)} trials total across "
          f"{inventory.groupby(['vision', 'feedback'], observed=True).ngroups} conditions:")
    print(inventory.groupby(["vision", "feedback"], observed=True).size().rename("n_trials"))
    print("\nTrials per subject per condition (flag anything ragged):")
    counts = inventory.groupby(["subject_id", "vision", "feedback"], observed=True).size()
    print(counts.groupby(["vision", "feedback"], observed=True).describe()[["min", "max", "mean"]])

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"\nsaved -> {out_path}  ({len(df):,} rows)")

    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    raw_dir = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "data/processed/cleaned_balance_all.parquet"
    if len(sys.argv) > 3:
        cutoff_arg = sys.argv[3]
        cutoff_hz = None if cutoff_arg.strip().lower() == "none" else float(cutoff_arg)
    else:
        cutoff_hz = 10.0  # unconfirmed default -- see LOW-PASS FILTER in this module's docstring
    load_all_sessions(raw_dir, out_path, filter_cutoff_hz=cutoff_hz)