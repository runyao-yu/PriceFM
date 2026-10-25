"""Verify cached request coverage and processed quarter-hour output."""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd

from .core import RAW, sha256
from .download import load_index, plan_missing
from .process import OUTPUT


def check_raw(regions, features, start, end, *, verify_hashes=False):
    index = load_index()
    missing = plan_missing(regions, features, start, end, index)
    if missing:
        sample = ", ".join(
            f"{item.region}/{item.family}/{item.neighbor or 'area'} "
            f"{item.start.date()}–{item.end.date()}" for item in missing[:5])
        raise RuntimeError(f"Raw request intervals remain missing ({len(missing)}): {sample}")
    if verify_hashes:
        for row in index:
            if row["region"] not in regions:
                continue
            path = RAW / row["path"]
            with gzip.open(path, "rb") as stream:
                if sha256(stream.read()) != row["sha256_uncompressed"]:
                    raise ValueError(f"Raw checksum mismatch: {path}")
    return {"cached_requests": len(index), "missing_intervals": 0}


def check_processed(regions, features, start, end):
    expected = pd.date_range(start, end, freq="15min", inclusive="left", tz="UTC")
    result = []
    for region in regions:
        for feature in features:
            path = OUTPUT / f"{region}_{feature}.csv"
            frame = pd.read_csv(path)
            if list(frame.columns) != ["timestamp_utc", "value",
                                       "source_observed", "hourly_expanded"]:
                raise ValueError(f"Unexpected processed schema: {path}")
            time = pd.DatetimeIndex(pd.to_datetime(frame["timestamp_utc"], utc=True))
            if time.has_duplicates or not time.is_monotonic_increasing:
                raise ValueError(f"Bad processed timestamps: {path}")
            inside = (time >= start) & (time < end)
            if not time[inside].equals(expected):
                raise ValueError(f"Requested quarter-hour grid incomplete: {path}")
            value = pd.to_numeric(frame.loc[inside, "value"],
                                  errors="coerce").to_numpy()
            observed = frame.loc[inside, "source_observed"].to_numpy(bool)
            expanded = frame.loc[inside, "hourly_expanded"].to_numpy(bool)
            if np.any(observed & expanded) and "PowerFlow" not in feature:
                raise ValueError(f"Source/expanded flags overlap: {path}")
            if np.any(expanded & ~np.isfinite(value)) or np.isinf(value).any():
                raise ValueError(f"Invalid processed values: {path}")
            result.append({"region": region, "feature": feature,
                           "requested_rows": len(expected),
                           "valid_rows": int(np.isfinite(value).sum())})
    return result
