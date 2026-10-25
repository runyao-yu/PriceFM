"""Four-case data with onshore/offshore wind kept in separate channels."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .alignment import FEATURES, FEATURE_COUNT


PROCESSED = Path.cwd() / "Data" / "Processed"
# Hourly-to-quarter-hour fill variants written by DATASETS/scripts/process.py.
VARIANT_SOURCES = {"copy": PROCESSED / "per_region_feature",
                   "interp": PROCESSED / "per_region_feature_interp",
                   "none": PROCESSED / "per_region_feature_nofill"}
SOURCE = Path(os.environ.get("PRICEFM_CLEAN_SOURCE",
                             str(VARIANT_SOURCES["copy"])))
VARIANT = "copy"


def source_path(region, name):
    path = SOURCE / f"{region}_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required clean series absent: {path}. "
                                "Run DATASETS/prepare_data.ipynb first.")
    return path
SPLITS = {
    "train": ("2023-01-01", "2025-01-01"),
    "val": ("2025-01-01", "2025-06-01"),
    # Held out; never used for training or checkpoint selection.
    # test_hourly: hourly SDAC; test_15min: after the 2025-10-01 15-min SDAC.
    "test_hourly": ("2025-06-01", "2025-10-01"),
    "test_15min": ("2025-10-01", "2026-10-01"),
}
TARGET_DATES = pd.date_range("2023-01-01", "2026-09-30", freq="D", tz="UTC")
# Named split schemes: (splits, first target date, last target date).
SPLIT_SCHEMES = {
    "default": (dict(SPLITS), "2023-01-01", "2026-09-30"),
    # Single-region benchmark: train on everything before the last six months
    # of 2025, validate on 2025-07..12, test on all of 2026 (to 2026-10-01).
    "single2026": ({"train": ("2022-01-08", "2025-07-01"),
                    "val": ("2025-07-01", "2026-01-01"),
                    "test": ("2026-01-01", "2026-10-02")},
                   "2022-01-08", "2026-10-01"),
    "fold1": ({"train": ("2022-01-01", "2025-10-01"),
               "val": ("2025-10-01", "2026-01-01"),
               "test": ("2026-01-01", "2026-04-01")},
              "2022-01-01", "2026-03-31"),
    "fold2": ({"train": ("2022-01-01", "2026-01-01"),
               "val": ("2026-01-01", "2026-04-01"),
               "test": ("2026-04-01", "2026-07-01")},
              "2022-01-01", "2026-06-30"),
    "fold3": ({"train": ("2022-01-01", "2026-04-01"),
               "val": ("2026-04-01", "2026-07-01"),
               "test": ("2026-07-01", "2026-10-01")},
              "2022-01-01", "2026-09-30"),
    "fold4": ({"train": ("2022-01-01", "2026-07-01"),
               "val": ("2026-07-01", "2026-10-01"),
               "test": ("2026-10-01", "2027-01-01")},
              "2022-01-01", "2026-09-30"),
}
SPLIT_SCHEME = "default"
INPUT_NAMES = FEATURES + ("DayAheadPrice",)
FORECAST_CHANNELS = (("DayAheadLoad", 0), ("DayAheadSolar", 1),
                     ("DayAheadWindOnshore", 2),
                     ("DayAheadWindOffshore", 3))
SCHEDULE_CHANNELS = (("DayAheadPowerFlowImport", 4),
                     ("DayAheadPowerFlowExport", 5))
CASE5_CHANNELS = FORECAST_CHANNELS + SCHEDULE_CHANNELS
CACHE_ROOT = Path.cwd() / "Data" / ".cache"
CACHE = CACHE_ROOT
SCALER_MODE = "regional_robust"
GLOBAL_SCALERS = {}


def _refresh_cache():
    global CACHE
    CACHE = CACHE_ROOT
    if VARIANT != "copy":
        CACHE = CACHE / VARIANT
    if SPLIT_SCHEME != "default":
        CACHE = CACHE / f"split_{SPLIT_SCHEME}"
    if SCALER_MODE != "regional_robust":
        CACHE = CACHE / SCALER_MODE


def set_variant(variant):
    """Point the loader (and its cache) at one fill variant of the data."""
    global SOURCE, VARIANT
    SOURCE = VARIANT_SOURCES[variant]
    VARIANT = variant
    _refresh_cache()


def set_splits(scheme):
    """Select a named split scheme (changes target dates, scalers and cache)."""
    global SPLITS, TARGET_DATES, SPLIT_SCHEME
    splits, first, last = SPLIT_SCHEMES[scheme]
    SPLITS = dict(splits)
    TARGET_DATES = pd.date_range(first, last, freq="D", tz="UTC")
    SPLIT_SCHEME = scheme
    _refresh_cache()


def set_scaler_mode(mode, regions=None, forecast_names=(), allow_empty=False):
    """Fit one train-only scaler per raw series across all selected regions."""
    global SCALER_MODE, GLOBAL_SCALERS
    if mode not in ("regional_robust", "global_standard",
                    "global_robust", "global_mad"):
        raise ValueError(mode)
    SCALER_MODE = mode
    GLOBAL_SCALERS = {}
    _refresh_cache()
    scaler_names = INPUT_NAMES + tuple(forecast_names)
    if mode != "regional_robust":
        if not regions:
            raise ValueError("Global scaling requires the region list")
        train_start, train_end = SPLITS["train"]
        train_masks = {}
        for region in regions:
            reference = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(
                source_path(region, "DayAheadPrice"),
                usecols=["timestamp_utc"])["timestamp_utc"], utc=True))
            train_masks[region] = (
                (reference >= pd.Timestamp(train_start, tz="UTC")) &
                (reference < pd.Timestamp(train_end, tz="UTC")))
    if mode == "global_standard":
        totals = {name: (0, 0.0, 0.0) for name in scaler_names}
        for region in regions:
            train = train_masks[region]
            for name in scaler_names:
                values = pd.to_numeric(pd.read_csv(
                    source_path(region, name), usecols=["value"])["value"],
                    errors="coerce").to_numpy(np.float32)
                if len(values) != len(train):
                    raise ValueError(f"Timestamp grid length differs: {region}_{name}")
                observed = values[train & np.isfinite(values)].astype("float64")
                if not len(observed):
                    continue
                n, center, m2 = totals[name]
                n_new = len(observed)
                center_new = float(observed.mean())
                m2_new = float(np.square(observed - center_new).sum())
                combined = n + n_new
                delta = center_new - center
                totals[name] = (
                    combined,
                    center + delta * n_new / combined,
                    m2 + m2_new + delta * delta * n * n_new / combined,
                )
        for name, (n, center, m2) in totals.items():
            if not n:
                raise ValueError(f"No observed training values for {name}")
            scale = float(np.sqrt(m2 / n))  # StandardScaler population std
            GLOBAL_SCALERS[name] = {
                "center": center, "scale": scale if scale > 0 else 1.0,
                "global_observed_train_points": n,
            }
    elif mode in ("global_robust", "global_mad"):
        for name in scaler_names:
            chunks = []
            for region in regions:
                values = pd.to_numeric(pd.read_csv(
                    source_path(region, name), usecols=["value"])["value"],
                    errors="coerce").to_numpy(np.float32)
                train = train_masks[region]
                if len(values) != len(train):
                    raise ValueError(f"Timestamp grid length differs: {region}_{name}")
                chunks.append(values[train & np.isfinite(values)])
            observed = np.concatenate(chunks).astype("float64")
            if not len(observed):
                if not allow_empty:
                    raise ValueError(f"No observed training values for {name}")
                GLOBAL_SCALERS[name] = {
                    "center": 0.0, "scale": 1.0,
                    "global_observed_train_points": 0,
                    "fallback": "unavailable_in_selected_regions",
                }
                continue
            center = float(np.median(observed))
            if mode == "global_robust":
                width = float(np.quantile(observed, .75) - np.quantile(observed, .25))
                primary = "IQR"
            else:
                width = float(1.4826 * np.median(np.abs(observed - center)))
                primary = "1.4826*MAD"
            fallback = None
            if width <= 0:
                width = float(np.quantile(observed, .95) - np.quantile(observed, .05))
                fallback = "p95-p05"
            if width <= 0:
                width = float(np.std(observed))
                fallback = "population_std"
            if width <= 0:
                width = 1.0
                fallback = "constant_series_1"
            GLOBAL_SCALERS[name] = {
                "center": center, "scale": width,
                "global_observed_train_points": int(len(observed)),
                "primary_scale": primary, "fallback": fallback,
            }


def split_indices(target_mask):
    """Split day indices; days without any observed target price are dropped."""
    usable = target_mask.any(axis=1)
    return {split: np.flatnonzero((TARGET_DATES >= pd.Timestamp(start, tz="UTC")) &
                                  (TARGET_DATES < pd.Timestamp(end, tz="UTC")) &
                                  usable)
            for split, (start, end) in SPLITS.items()}


def _build_region_uncached(region, regions):
    if region not in regions:
        raise ValueError(region)
    timestamps, raw = None, {}
    for name in INPUT_NAMES:
        path = source_path(region, name)
        columns = ["timestamp_utc", "value"] + (
            ["source_observed"] if name == "DayAheadPrice" else [])
        frame = pd.read_csv(path, usecols=columns)
        if set(frame.columns) != set(columns):
            raise ValueError(f"Unexpected columns: {path}")
        if name == "DayAheadPrice":
            # Natively reported price slots: identical in every fill variant,
            # so they form the common evaluation set.
            native_price = frame["source_observed"].to_numpy(bool)
        index = pd.DatetimeIndex(pd.to_datetime(frame["timestamp_utc"], utc=True))
        if timestamps is None:
            timestamps = index
            if index.has_duplicates or not index.is_monotonic_increasing:
                raise ValueError(f"Bad time grid: {path}")
            if not np.all((index[1:] - index[:-1]) == pd.Timedelta(minutes=15)):
                raise ValueError(f"Non-quarter-hour grid: {path}")
        elif not timestamps.equals(index):
            raise ValueError(f"Timestamp mismatch: {path}")
        raw[name] = pd.to_numeric(frame["value"], errors="coerce").to_numpy(np.float32)

    # Pad the grid with blank days so every target date has its seven-day
    # history; series that end early simply contribute no observations.
    first = min(timestamps[0], TARGET_DATES[0] - pd.Timedelta(days=7))
    last = max(timestamps[-1], TARGET_DATES[-1] + pd.Timedelta(hours=23, minutes=45))
    grid = pd.date_range(first, last, freq="15min", tz="UTC")
    if not grid.equals(timestamps):
        position = grid.get_indexer(timestamps)
        if (position < 0).any():
            raise ValueError(f"Source grid off the quarter-hour axis: {region}")
        for name in INPUT_NAMES:
            padded = np.full(len(grid), np.nan, np.float32)
            padded[position] = raw[name]
            raw[name] = padded
        padded = np.zeros(len(grid), bool)
        padded[position] = native_price
        native_price = padded
        timestamps = grid
    if timestamps[0].hour or timestamps[0].minute or len(timestamps) % 96:
        raise ValueError(f"UTC days not aligned: {region}")
    days = pd.date_range(timestamps[0], periods=len(timestamps) // 96,
                         freq="D", tz="UTC")
    if not timestamps[::96].equals(days):
        raise ValueError(f"Daily timestamps not aligned: {region}")

    train_start, train_end = SPLITS["train"]
    train_time = ((timestamps >= pd.Timestamp(train_start, tz="UTC")) &
                  (timestamps < pd.Timestamp(train_end, tz="UTC")))
    normalized, scalers = {}, {}
    for name in INPUT_NAMES:
        observed = raw[name][train_time & np.isfinite(raw[name])].astype("float64")
        if SCALER_MODE != "regional_robust":
            fitted = GLOBAL_SCALERS[name]
            center, scale = fitted["center"], fitted["scale"]
        else:
            center = float(np.median(observed)) if len(observed) else 0.0
            width = float(np.quantile(observed, .75) - np.quantile(observed, .25)) if len(observed) else 1.0
            scale = width if width > 0 else 1.0
        normalized[name] = ((raw[name] - center) / scale).astype("float32")
        scalers[name] = {"center": center, "scale": scale,
                          "observed_train_points": int(len(observed))}
        if SCALER_MODE != "regional_robust":
            scalers[name]["global_observed_train_points"] = fitted["global_observed_train_points"]
            if fitted.get("fallback"):
                scalers[name]["fallback"] = fitted["fallback"]
        if not len(observed):
            # No training observations means no scaler; a series that starts
            # later (e.g. new offshore wind) would enter unscaled. Mask it.
            normalized[name][:] = np.nan
            scalers[name]["masked_no_train_observations"] = True

    actual = np.stack([normalized[name].reshape(-1, 96) for name in FEATURES], axis=-1)
    price = normalized["DayAheadPrice"].reshape(-1, 96)
    target_index = (TARGET_DATES - days[0]).days.to_numpy()
    if target_index.min() < 7 or target_index.max() >= len(days):
        raise ValueError(f"Insufficient date coverage: {region}")
    count = len(target_index)
    # Target day t=d+1: actuals are d-6,...,d-1 fully, then d 00:00-12:00.
    lag_full = np.stack([actual[target_index - offset]
                         for offset in range(7, 1, -1)], axis=1)
    lag = np.concatenate((lag_full.reshape(count, 576, FEATURE_COUNT),
                          actual[target_index - 1, :48, :]), axis=1)
    price_history = np.stack([price[target_index - offset]
                              for offset in range(7, 0, -1)], axis=1).reshape(count, 672, 1)
    lead = actual[target_index]
    target = price[target_index].copy()
    target_mask = np.isfinite(target)
    eval_mask = native_price.reshape(-1, 96)[target_index] & target_mask
    if lag.shape != (count, 624, FEATURE_COUNT):
        raise AssertionError("Lag length misaligned")
    if not np.array_equal(lag[:, :96], actual[target_index - 7], equal_nan=True):
        raise AssertionError("Lag start misaligned")
    if not np.array_equal(lag[:, 576:], actual[target_index - 1, :48], equal_nan=True):
        raise AssertionError("Issue-day actuals misaligned")
    if not np.array_equal(price_history[:, -96:, 0],
                          price[target_index - 1], equal_nan=True):
        raise AssertionError("Most recent historical price misaligned")
    if not np.array_equal(lead, actual[target_index], equal_nan=True):
        raise AssertionError("Oracle lead misaligned")
    region_code = np.broadcast_to(
        np.eye(len(regions), dtype="float32")[regions.index(region)],
        (count, len(regions))).copy()
    data = {
        "lag_values": lag,
        "lag_mask": np.isfinite(lag),
        "lead_values": lead,
        "lead_mask": np.isfinite(lead),
        "price_values": price_history,
        "price_mask": np.isfinite(price_history),
        "region_code": region_code,
        "target": target,
        "target_mask": target_mask,
        "eval_mask": eval_mask,
    }
    indices = split_indices(target_mask)
    audit = {
        "region": region,
        "source_first_timestamp": str(timestamps[0]),
        "source_last_timestamp": str(timestamps[-1]),
        "lag_actual_offsets_from_target": [7, 6, 5, 4, 3, 2, 1],
        "lag_issue_day_slots": 48,
        "historical_price_offsets_from_target": [7, 6, 5, 4, 3, 2, 1],
        "lead_offset_from_target": 0,
        "split_days": {name: len(ids) for name, ids in indices.items()},
        "observed_target_slots": {name: int(target_mask[ids].sum())
                                  for name, ids in indices.items()},
        "native_target_slots": {name: int(eval_mask[ids].sum())
                                for name, ids in indices.items()},
        "variant": VARIANT,
        "scaler_mode": SCALER_MODE,
    }
    return data, indices, scalers, audit


def build_region(region, regions):
    """Cache the expensive CSV alignment; refresh if any source file changes."""
    if region not in regions:
        raise ValueError(region)
    if SCALER_MODE != "regional_robust":
        # Pooled statistics depend on every region's source file. Avoid a
        # per-region cache whose key cannot track the other 39 regions.
        return _build_region_uncached(region, regions)
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{region}_base.npz"
    meta_path = CACHE / f"{region}_base.json"
    source_state = {}
    for name in INPUT_NAMES:
        source = source_path(region, name)
        stat = source.stat()
        source_state[name] = {"path": str(source.resolve()), "size": stat.st_size,
                              "mtime_ns": stat.st_mtime_ns}
    if path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        if metadata.get("version") == 4 and metadata.get("source_state") == source_state:
            with np.load(path) as bundle:
                data = {key: bundle[key] for key in
                        ("lag_values", "lead_values", "price_values", "target",
                         "eval_mask")}
            data["lag_mask"] = np.isfinite(data["lag_values"])
            data["lead_mask"] = np.isfinite(data["lead_values"])
            data["price_mask"] = np.isfinite(data["price_values"])
            data["target_mask"] = np.isfinite(data["target"])
            data["target_dates"] = TARGET_DATES
            data["region_code"] = np.broadcast_to(
                np.eye(len(regions), dtype="float32")[regions.index(region)],
                (len(TARGET_DATES), len(regions))).copy()
            indices = split_indices(data["target_mask"])
            return data, indices, metadata["scalers"], metadata["audit"]
    data, indices, scalers, audit = _build_region_uncached(region, regions)
    temporary = CACHE / f"{region}_base.tmp"
    with temporary.open("wb") as handle:
        np.savez(handle, **{key: data[key] for key in
                             ("lag_values", "lead_values", "price_values", "target",
                              "eval_mask")})
    temporary.replace(path)
    meta_temporary = CACHE / f"{region}_base.json.tmp"
    meta_temporary.write_text(json.dumps({
        "version": 4, "source_state": source_state,
        "scalers": scalers, "audit": audit}, indent=2))
    meta_temporary.replace(meta_path)
    return data, indices, scalers, audit


def build_case_region(region, case, regions):
    """Apply the lead-source definition without altering lag or price history."""
    if case not in ("case1", "case2", "case3", "case4", "case5"):
        raise ValueError(case)
    lead_channels = (CASE5_CHANNELS if case == "case5" else FORECAST_CHANNELS)
    if case in ("case2", "case5") and SCALER_MODE != "regional_robust":
        missing = [name for name, _ in lead_channels
                   if name not in GLOBAL_SCALERS]
        if missing:
            raise ValueError(f"Forecast scalers were not fitted: {missing}")
    data, indices, scalers, audit = build_region(region, regions)
    data = dict(data)
    scalers = dict(scalers)
    audit = dict(audit)
    lead = np.full_like(data["lead_values"], np.nan)
    if case in ("case2", "case5"):
        reference = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(
            source_path(region, "DayAheadPrice"),
            usecols=["timestamp_utc"])["timestamp_utc"], utc=True))
        days = pd.date_range(reference[0], periods=len(reference) // 96,
                             freq="D", tz="UTC")
        if not reference[::96].equals(days):
            raise ValueError(f"Forecast daily alignment: {region}")
        target_index = (TARGET_DATES - days[0]).days.to_numpy()
        start, end = SPLITS["train"]
        train_time = ((reference >= pd.Timestamp(start, tz="UTC")) &
                      (reference < pd.Timestamp(end, tz="UTC")))
        for name, channel in lead_channels:
            frame = pd.read_csv(source_path(region, name),
                                usecols=["timestamp_utc", "value"])
            if set(frame.columns) != {"timestamp_utc", "value"}:
                raise ValueError(f"Unexpected forecast columns: {region}_{name}")
            timestamps = pd.DatetimeIndex(pd.to_datetime(
                frame["timestamp_utc"], utc=True))
            if not timestamps.equals(reference):
                raise ValueError(f"Forecast timestamp mismatch: {region}_{name}")
            raw = pd.to_numeric(frame["value"], errors="coerce").to_numpy(np.float32)
            observed = raw[train_time & np.isfinite(raw)].astype("float64")
            if SCALER_MODE == "regional_robust":
                center = float(np.median(observed)) if len(observed) else 0.0
                width = (float(np.quantile(observed, .75) - np.quantile(observed, .25))
                         if len(observed) else 1.0)
                scale = width if width > 0 else 1.0
            else:
                fitted = GLOBAL_SCALERS[name]
                center, scale = fitted["center"], fitted["scale"]
            lead[:, :, channel] = ((raw - center) / scale).astype("float32").reshape(
                -1, 96)[target_index]
            scalers[name] = {"center": center, "scale": scale,
                             "observed_train_points": int(len(observed))}
            if SCALER_MODE != "regional_robust":
                scalers[name]["global_observed_train_points"] = fitted["global_observed_train_points"]
                if fitted.get("fallback"):
                    scalers[name]["fallback"] = fitted["fallback"]
    elif case == "case3":
        lead[:, :, :4] = data["lead_values"][:, :, :4]
    elif case == "case4":
        lead = data["lead_values"]
    data["lead_values"] = lead
    data["lead_mask"] = np.isfinite(lead)
    audit["case"] = case
    audit["lead_source"] = {
        "case1": "none",
        "case2": "target-day day-ahead load/solar/wind-onshore/wind-offshore",
        "case3": "target-day actual load/solar/wind-onshore/wind-offshore",
        "case4": "target-day actual load/solar/wind-onshore/wind-offshore/import/export",
        "case5": "target-day day-ahead load/solar/wind forecasts and A01 commercial exchange schedules for import/export",
    }[case]
    audit["lead_observed_fraction"] = {
        split: data["lead_mask"][ids].mean(axis=(0, 1)).astype(float).tolist()
        for split, ids in indices.items()}
    return data, indices, scalers, audit


def select(data, ids):
    return {name: values[ids] for name, values in data.items()}
