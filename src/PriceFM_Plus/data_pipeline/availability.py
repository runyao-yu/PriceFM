"""Series availability rule, applied after processing.

A processed series is marked NOT AVAILABLE, and all its values are set to
blank (real missing; flags 0) in every fill variant, when over the archive
window either
  * its missing rate exceeds MAX_MISSING (0.5), e.g. a feature that only
    starts late in the archive (PL offshore wind from 2026-07), or
  * at least DEGENERATE_SHARE (95 %) of its observed values are exactly 0 or
    1 (placeholders: always 0, a mix of 0 and blank, or a constant 1).
Values ENTSO-E may still return for such a series are discarded.

Decisions are made on the copy variant before blanking and stored in
config/series_availability.json (with reason, missing rate, first/last valid
timestamp). A series that is already blank keeps its stored reason.

Usage:  python availability.py [REGION ...]      (default: all active regions)
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from .core import ALL_FEATURES, CONFIG, PROCESSED, atomic_json, load_config

MAX_MISSING = 0.5
DEGENERATE_SHARE = 0.95
DECISIONS = CONFIG / "series_availability.json"
VARIANT_DIRS = ("per_region_feature", "per_region_feature_interp", "per_region_feature_nofill")


def _load_decisions():
    return json.loads(DECISIONS.read_text()) if DECISIONS.exists() else {}


def valid_spans(ok: np.ndarray, times: pd.Series, min_gap_days: int = 7):
    """Spans of observed data, merging gaps shorter than `min_gap_days`."""
    idx = np.flatnonzero(ok)
    if not len(idx):
        return []
    breaks = np.flatnonzero(np.diff(idx) > 96 * min_gap_days)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return [[f"{times.iloc[a]:%Y-%m-%d}", f"{times.iloc[b]:%Y-%m-%d}"] for a, b in zip(starts, ends)]


def describe_spans(spans, times) -> str:
    first, last = f"{times.iloc[0]:%Y-%m-%d}", f"{times.iloc[-1]:%Y-%m-%d}"
    if len(spans) == 1:
        a, b = spans[0]
        if a > first and b >= last:
            return f"only available from {a}"
        if a <= first and b < last:
            return f"not available after {b}"
        return f"only available {a} to {b}"
    return "available only " + ", ".join(f"{a} to {b}" for a, b in spans[:3]) + (" ..." if len(spans) > 3 else "")


def decide(frame: pd.DataFrame) -> dict:
    values = pd.to_numeric(frame["value"], errors="coerce").to_numpy(float)
    ok = np.isfinite(values)
    times = pd.to_datetime(frame["timestamp_utc"], utc=True)
    info = {"missing_rate": float(1 - ok.mean()), "start": str(times.iloc[0]),
            "end": str(times.iloc[-1])}
    if not ok.any():
        return {**info, "available": False, "reason": "never published"}
    first, last = np.flatnonzero(ok)[[0, -1]]
    info.update(first_valid=str(times.iloc[first]), last_valid=str(times.iloc[last]),
                share_0_or_1=float(np.isin(values[ok], [0.0, 1.0]).mean()))
    if info["share_0_or_1"] >= DEGENERATE_SHARE:
        constant = np.unique(values[ok])
        label = ("always 0" if set(constant) == {0.0} else
                 "always 1" if set(constant) == {1.0} else "only 0/1 placeholder values")
        return {**info, "available": False, "reason": label}
    info["valid_spans"] = valid_spans(ok, times)
    if info["missing_rate"] > MAX_MISSING:
        return {**info, "available": False,
                "reason": f"missing rate > {MAX_MISSING:.0%}; {describe_spans(info['valid_spans'], times)}"}
    return {**info, "available": True, "reason": ""}


def apply(regions, features=ALL_FEATURES):
    decisions = _load_decisions()
    changed = []
    for region in regions:
        for feature in features:
            key = f"{region}/{feature}"
            path = PROCESSED / VARIANT_DIRS[0] / f"{region}_{feature}.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            decision = decide(frame)
            prior = decisions.get(key)
            if decision["reason"] == "never published" and prior and not prior["available"] \
                    and prior["reason"] != "never published":
                decision = prior          # already blanked by this rule: keep the reason
            decisions[key] = decision
            if decision["available"] or decision["reason"] == "never published":
                continue
            for name in VARIANT_DIRS:
                target = PROCESSED / name / f"{region}_{feature}.csv"
                if not target.exists():
                    continue
                data = pd.read_csv(target)
                if data["value"].notna().any():
                    data["value"] = np.nan
                    data["source_observed"] = 0
                    data["hourly_expanded"] = 0
                    tmp = target.with_suffix(".csv.tmp")
                    data.to_csv(tmp, index=False, na_rep="")
                    tmp.replace(target)
                    changed.append(f"{name}/{region}_{feature}")
    atomic_json(DECISIONS, decisions)
    return changed


if __name__ == "__main__":
    regions = sys.argv[1:] or list(load_config()[0])
    for item in apply(regions):
        print("blanked", item)
