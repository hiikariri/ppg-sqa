"""
PPG record loaders for dataset_primer_1 (Hi-Me! 2.0).

Each subject has a ``*_ppg.csv`` with columns ``Time (s), PPG_Red, PPG_IR``
sampled at 125 Hz for 60 s (MAX30105 reflectance sensor), plus a
``*_metadata.json`` ground truth and a folder-level ``.rr_excluded.json``
carrying the dataset's own signal-quality verdicts.

The PPG is recorded in inverted polarity, so :func:`load_primer_ppg` negates it
on load to put systolic peaks upward (the orientation the SQIs expect).

Mirrors the :class:`ecg_datasets.ECGRecord` interface so the PPG SQA GUI can
reuse the same load / discover / navigate flow.
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
PRIMER_DIR = os.path.join(_ROOT, "dataset_primer_1")

PPG_SUFFIX = "_ppg.csv"
# Display name -> CSV column. IR is the default analysis channel (deeper
# penetration, stronger pulsatile component than Red on the MAX30105).
PPG_CHANNELS = {"IR": "PPG_IR", "Red": "PPG_Red"}
LABELS_FILE = ".rr_excluded.json"


@dataclass
class PPGRecord:
    name: str
    ppg: np.ndarray              # single-channel PPG, sign-inverted to systolic-up
    fs: float
    time: np.ndarray            # per-sample time axis (s)
    ref_hr: Optional[float]     # ground-truth heart rate (bpm), if available
    channel: str = ""           # which channel the PPG came from

    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0]) if self.time.size else 0.0


def short_label(name: str) -> str:
    """Trim ``data_`` prefix and ``_YYYYMMDD_HHMMSS`` timestamp from a name."""
    s = re.sub(r"_\d{8}_\d{6}$", "", name)
    return s[len("data_"):] if s.startswith("data_") else s


def _sampling_rate(time: np.ndarray) -> float:
    """fs from the total span, robust to per-sample timestamp rounding."""
    if time.size < 2:
        return float("nan")
    span = float(time[-1] - time[0])
    return (time.size - 1) / span if span > 0 else float("nan")


def _ref_hr(meta_path: str) -> Optional[float]:
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        return float(meta["ground_truth"]["heart_rate_bpm"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def load_primer_ppg(path: str, *, channel: str = "PPG_IR") -> PPGRecord:
    """Load one primer ``*_ppg.csv``. ``path`` may include or omit the suffix."""
    base = path[:-len(PPG_SUFFIX)] if path.endswith(PPG_SUFFIX) else path
    csv_path = base + PPG_SUFFIX
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    col = channel.strip()
    if col not in df.columns:                       # fall back to IR / last col
        col = "PPG_IR" if "PPG_IR" in df.columns else df.columns[-1]
    time = df["Time (s)"].to_numpy(dtype=float)
    ppg = df[col].to_numpy(dtype=float)
    # The sensor records PPG in inverted polarity, so negate it on load: this
    # puts the systolic upstroke as an upward peak — the orientation the Elgendi
    # SQIs in ppg_sqa.py assume (skewness sign, upward peak detectors). Without
    # it, otherwise-clean records (e.g. data_006_Abel) get a negative SSQI and
    # are wrongly flagged unfit.
    ppg = -ppg
    fs = _sampling_rate(time)
    name = os.path.basename(base)
    ref_hr = _ref_hr(base + "_metadata.json")
    return PPGRecord(name=name, ppg=ppg, fs=fs, time=time, ref_hr=ref_hr,
                     channel=col)


def discover_ppg(folder: str):
    """Return ``[(base_path, name), ...]`` for every ``*_ppg.csv`` in folder."""
    recs = []
    for f in sorted(glob.glob(os.path.join(folder, f"*{PPG_SUFFIX}"))):
        base = f[:-len(PPG_SUFFIX)]
        recs.append((base, os.path.basename(base)))
    return recs


def load_sqa_labels(folder: str) -> Dict[str, bool]:
    """Read ``.rr_excluded.json`` -> ``{record_base_name: is_bad}``.

    A record is "bad" if it is listed in ``sqa_bad``; an entry in ``overrides``
    takes precedence (its boolean *is* the bad/good flag the user pinned).
    Returns an empty dict if the file is missing or malformed.
    """
    path = os.path.join(folder, LABELS_FILE)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (ValueError, OSError):
        return {}
    bad = set(data.get("sqa_bad", []))
    overrides = data.get("overrides", {})
    labels = {name: True for name in bad}
    for name, is_bad in overrides.items():
        labels[name] = bool(is_bad)
    return labels
