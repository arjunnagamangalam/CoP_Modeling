"""
Rebuilds cleaned_balance_all.parquet from the raw per-trial force-plate
CSVs, after the original cleaned parquet (and the original loading script)
were both lost.

RAW FILE FORMAT (confirmed directly against a real sample file, not
guessed -- see the numeric verification below and in this project's chat
history):

  - Motion-capture-suite ("Cortex"-style) ASCII export, BOM-prefixed
    ("Devices" as the very first line).
  - Line 2: the analog sample rate in Hz (200 in the sample checked).
  - Two stacked data blocks in one file: a "Devices" block (force-plate
    channels) first, then a "Trajectories" block (motion-capture marker
    positions) after a blank line. Only the Devices block is CoP data --
    this loader stops there and never reads past the "Trajectories" line.
  - The Devices block has TWO force plates side by side, each with 9
    columns: Fx,Fy,Fz (N), Mx,My,Mz (N.mm), Cx,Cy,Cz (mm). Cx,Cy,Cz is a
    VENDOR-COMPUTED CoP, not a raw channel -- verified numerically against
    the standard formula:
        Cx = -My/Fz + 200.0   (residual std ~0.0003 mm across 4000 samples)
        Cy =  Mx/Fz + 300.0   (residual std ~0.0003 mm across 4000 samples)
    i.e. Cx/Cy equal the textbook CoP formula plus a fixed calibration
    offset (almost certainly the plate's known physical position). Since
    every consumer of this data mean-centers each trial before using it,
    that constant offset has zero effect on anything downstream -- using
    Cx/Cy directly is equivalent to recomputing -My/Fz and Mx/Fz by hand,
    and simpler/less error-prone (no risk of getting a plate-thickness
    z-offset correction term wrong from memory).
  - In quiet single-leg-plate-loaded trials only ONE of the two plates
    actually carries the subject's weight -- the other reports Fz=0 and a
    constant "no-load default" CoP (a clearly different, much larger-
    magnitude offset than the active plate's -- e.g. (-1040.2, 296) vs the
    active plate's (200, 300) pattern above). Which plate (#1 or #2) is
    active was NOT confirmed to be consistent across all files, so this
    loader auto-detects the active plate per file via mean |Fz|, rather
    than hardcoding a device index.

TWO-PLATE FALLBACK (your call was "no preference" -- this is MY judgment
call, not a confirmed fact, flag it if trajectories still look off): a
real production run turned up many Silent/Auditory files where BOTH
plates read a real load -- e.g. every S02 trial has plate1~667N (body
weight) AND plate2~51N; S03 ~739N/~54N; S16 ~614N/~84N. Each subject's
secondary-plate magnitude is its OWN very consistent value (roughly
7-14% of the main plate) across dozens of that subject's trials, and
plenty of OTHER trials (e.g. the original S01 sample) have a clean
single active plate. That pattern -- consistent per-subject, but not
present in every trial or every subject -- reads much more like an
imperfectly zeroed/tared secondary plate than a genuine deliberate
two-plate stance (a real two-plate protocol would show up in every
trial, every subject). So: this loader now treats "both plates active"
the same as the clean single-plate case -- picks whichever plate has
the larger mean|Fz| (the one actually carrying body weight) and ignores
the other, rather than raising and dropping the trial. Every file this
fallback fired on is now reported by load_all_sessions (both a count and
the filenames, capped at 10 printed) specifically so you can spot-check
a few before trusting them -- if it turns out these really were a
genuine two-plate stance needing real force/moment combination across
both plates, tell me and I'll redo this properly instead of guessing.
  - Exactly one of the sample checked: 4000 samples in the Devices block
    (Frame 1..2000, SubFrame 0/1 each) = 20s at 200Hz, matching this
    project's established trial length/rate. Enforced here as a check,
    not an assumption -- a file that doesn't match is flagged, not
    silently truncated/padded.

FILENAME CONVENTION (confirmed against the one sample provided --
S01_Close_Auditory_31.csv): "{subject}_{vision}_{feedback}_{trial}.csv"
where vision in {Open, Close} and feedback in {Silent, Auditory}. This
loader parses subject/vision/feedback/trial from the FILENAME only (not
from directory structure), so it works whether your raw files sit in one
flat folder or nested per-subject subfolders -- it searches recursively.

A real run also turned up feedback = "Tactile" and "Combined" files
(e.g. S02_Close_Tactile_52.csv, S03_Open_Combined_03.csv) alongside
Silent/Auditory. You confirmed you only want Silent/Auditory -- so
FILENAME_RE deliberately does NOT match Tactile/Combined, and those
files fall into the normal "skipped" list (reported, not silently
dropped) rather than being loaded.

CALIBRATION TRIALS (confirmed with you): each session includes a
calibration trial that must be EXCLUDED from the cleaned dataset -- you
gave "S01 Cal 01" as an example filename. This is a different naming
convention from the real trials above (space-separated, "Cal" instead of
a vision/feedback pair), so CALIBRATION_RE below matches it separately
and BEFORE the normal filename parser runs, and such files are reported
as explicitly excluded (not lumped in with genuinely malformed/unexpected
filenames). It accepts both "Cal" and "Calibration", and either a space
or underscore as the separator, since I only have the one example and
don't know which your files actually use -- if a session's calibration
file isn't being caught (check the "excluded as calibration trials"
count printed against how many sessions you have), tell me its exact
filename and I'll tighten/adjust the pattern.

AXIS LABELING (Cx -> cop_x/"ML", Cy -> cop_y/"AP"): NOT independently
confirmed against lab setup notes as of this writing -- this is the
conventional default (x=ML, y=AP, matching Cx/Cy's column order in the
file) but you should double check it against your force-plate mounting
notes before trusting any ML-vs-AP-specific claim from the calibrated
model. It does NOT affect calibration correctness either way (both axes
are fit independently and symmetrically) -- getting it backwards would
only mean the "ML" and "AP" labels in reports/plots are swapped for both
real and simulated data alike, consistently, everywhere.

CLEANING: this loader does four things beyond parsing: active-plate
selection, mm -> m unit conversion, a LOW-PASS FILTER, and a per-trial
BASELINE SUBTRACTION -- plus a sample-count sanity check.

LOW-PASS FILTER (changed after real calibration output looked wrong --
see below; cutoff itself is still an unconfirmed guess): originally you
said no filter was used, and the loader shipped with none. After a real
run through the (separate, older) dual-pathway model, the held-out real
trajectories looked visibly noisier than before the parquet was lost,
and the model's predicted-vs-real PSD comparison got noticeably worse.
You then confirmed you DID likely use a Butterworth low-pass originally,
but don't remember the exact cutoff. So: this loader now applies one by
default -- 10Hz, 4th-order, zero-phase (filtfilt), via apply_filter()
below -- since that's a commonly-cited value in this literature (range
usually reported ~5-12.5Hz) and a reasonable starting point, NOT a
confirmed match to whatever the original pipeline actually used. If
downstream results (PSD shape, DFA, etc.) still look off, the cutoff is
the first thing to revisit -- pass a different value via the CLI's 3rd
argument or the filter_cutoff_hz= parameter and compare. Pass
filter_cutoff_hz=None (or CLI value "none") to disable filtering
entirely and get the raw signal back.

BASELINE SUBTRACTION (confirmed with you, not guessed): every trial's
cop_x_clean/cop_y_clean is shifted so its OWN first sample is exactly 0 --
i.e. cop_x_clean = cx_filt - cx_filt[0], cop_y_clean = cy_filt - cy_filt[0].
This was first noticed as a pattern in the still-cached old
cleaned_balance_all.parquet (every trial checked had cop_x_clean ==
cop_y_clean == 0.0 exactly at samples 0 and 1) and then confirmed
directly with you before being wired in here, since it couldn't be
verified numerically against the one raw sample file available (its raw
trial number, 31, isn't among the trial labels present in the old
cached data for that subject/condition -- see TRIAL LABELING below).
IMPORTANT ordering: this subtraction happens AFTER the low-pass filter
above, not before -- filtering a signal that's already been zeroed at
one sample can nudge that sample slightly off zero again (filtfilt's
edge-padding isn't perfectly DC-preserving right at the boundary), which
would silently break the "starts at exactly 0" property the old cached
data actually has. Filtering first, then zeroing the filtered signal's
own first sample, guarantees the property holds regardless of the
filter's edge behavior.

TRIAL LABELING (confirmed with you): the 'trial' column is the raw
filename's own trial number (e.g. "31"), used as-is -- NOT re-indexed
per subject/condition. Also confirmed: no trial-selection/QC step is
applied here -- every raw file found is loaded, however many trials that
gives per subject/condition (the old cached parquet had exactly 8 per
subject/condition, i.e. trial labels 0-7, which is a real mismatch
against raw trial numbers like "31" -- you confirmed this loader should
NOT try to replicate that particular subsetting).

The output columns are still named cop_x_clean/cop_y_clean for schema
compatibility with the rest of this project's code.

USAGE:
    python3 load_raw_sessions.py /path/to/raw/session/files [output.parquet] [cutoff_hz]

    cutoff_hz defaults to 10 (Hz) if omitted. Pass "none" to disable
    filtering and get the raw (baseline-subtracted only) signal.
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
    re.IGNORECASE,
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