"""Build small, browser-readable forecast shards from the released NPZ archives.

Usage: python Project_page/build_forecast_data.py --release-root PATH
The source archives and full-precision Hugging Face release are never modified.
Each month is independent, so a later test fold only adds new month files.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

CASES = ("Realistic", "TSO", "Oracle")
MODELS = (
    "PriceFM_Plus", "TabPFN_TS_3", "TiRex_2", "TimesFM_3", "Chronos_2",
    "MLP", "LSTM", "Transformer", "Persistence_1", "Persistence_2", "Persistence_3",
)
TRUTH_SCALE = 10  # Observed prices retain 0.1 EUR/MWh resolution.
FORECAST_SCALE = 0.4  # Viewer forecasts use 2.5 EUR/MWh steps; full precision remains on Hugging Face.
MISSING = -32768


def source_path(root: Path, model: str, case: str, fold: int) -> Path:
    f = f"fold{fold}.npz"
    base = root / "Forecasts"
    if model == "PriceFM_Plus":
        return base / model / case / "Joint" / "L2" / f
    if model in ("MLP", "LSTM", "Transformer"):
        return base / model / case / "Joint" / "Include_2022" / f
    if model.startswith("Persistence_"):
        return base / "Persistence" / model / f
    return base / model / case / f


def quantize(a: np.ndarray, dtype: str, scale: float) -> np.ndarray:
    finite = np.isfinite(a)
    bounds = np.iinfo(dtype)
    missing = bounds.min
    if finite.any() and (np.rint(a[finite] * scale).min() <= missing
                         or np.rint(a[finite] * scale).max() > bounds.max):
        raise ValueError(f"Price outside {dtype} range at scale {scale}")
    result = np.full(a.shape, missing, dtype=dtype)
    result[finite] = np.rint(a[finite] * scale).astype(dtype)
    return result


def time_deltas(a: np.ndarray) -> np.ndarray:
    """Difference consecutive quarter hours; preserve missing and gap restarts."""
    result = a.copy()
    missing = np.iinfo(a.dtype).min
    current, previous = a[:, :, 1:], a[:, :, :-1]
    valid = (current != missing) & (previous != missing)
    diff = current.astype("<i8") - previous.astype("<i8")
    if valid.any() and (diff[valid].min() < np.iinfo(a.dtype).min + 1
                        or diff[valid].max() > np.iinfo(a.dtype).max):
        raise ValueError("Time delta outside selected integer range")
    result[:, :, 1:][valid] = diff[valid].astype(a.dtype)
    return result


def read_truth(root: Path, regions: list[str], dates: list[str]) -> np.ndarray:
    day_lookup = {day: i for i, day in enumerate(dates)}
    result = np.full((len(dates), len(regions), 96), np.nan, dtype=np.float32)
    for region_i, region in enumerate(regions):
        path = root / "Data" / "Processed" / "per_region_feature_interp" / f"{region}_DayAheadPrice.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                stamp = row["timestamp_utc"]
                day_i = day_lookup.get(stamp[:10])
                if day_i is None or not row["value"]:
                    continue
                hour, minute = int(stamp[11:13]), int(stamp[14:16])
                result[day_i, region_i, 4 * hour + minute // 15] = float(row["value"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    root, output = args.release_root, args.output
    output.mkdir(parents=True, exist_ok=True)
    reference = source_path(root, "PriceFM_Plus", "Realistic", 1)
    with np.load(reference) as z:
        regions = z["regions"].tolist()
    if len(regions) != 40:
        raise ValueError(f"Expected 40 regions, found {len(regions)}")

    manifest = {
        "schema": 2, "quantization": "truth 0.1 EUR/MWh; forecasts 2.5 EUR/MWh",
        "truth_scale": TRUTH_SCALE, "forecast_scale": FORECAST_SCALE, "missing": MISSING,
        "slots_per_day": 96, "quantiles": [0.1, 0.5, 0.9],
        "regions": regions, "models": list(MODELS), "cases": list(CASES), "shards": {},
    }
    # All feature cases share held-out dates and the same observed prices.
    test_days = []
    for fold in (1, 2, 3, 4):
        path = source_path(root, "PriceFM_Plus", "Realistic", fold)
        if path.exists():
            with np.load(path) as z:
                test_days.extend(str(d) for d in z["dates"].astype("datetime64[D]"))
    test_days = sorted(set(test_days))
    truth_all = read_truth(root, regions, test_days)
    truth_lookup = {day: i for i, day in enumerate(test_days)}
    total = 0
    for case in CASES:
        # The test folds currently cover Jan–Sep 2026. A future fold 4 is discovered automatically.
        with np.load(source_path(root, "PriceFM_Plus", case, 1)) as first:
            all_days = [str(d) for d in first["dates"].astype("datetime64[D]")]
        folds = [1]
        for fold in (2, 3, 4):
            path = source_path(root, "PriceFM_Plus", case, fold)
            if path.exists():
                with np.load(path) as z:
                    all_days.extend(str(d) for d in z["dates"].astype("datetime64[D]"))
                folds.append(fold)
        all_days = sorted(set(all_days))
        by_month = {}
        for day in all_days:
            by_month.setdefault(day[:7], []).append(day)
        arrays = {}
        for model in MODELS:
            for fold in folds:
                path = source_path(root, model, case, fold)
                if not path.exists():
                    raise FileNotFoundError(path)
                with np.load(path) as z:
                    arrays[(model, fold)] = (
                        z["dates"].astype("datetime64[D]").astype(str),
                        z["region_index"], z["day_index"], z["slot_utc"],
                        z["prediction_EUR_per_MWh"],
                    )
        for month, days in by_month.items():
            lookup = {day: i for i, day in enumerate(days)}
            truth = truth_all[[truth_lookup[d] for d in days]]
            forecast = np.full((len(days), 40, 96, len(MODELS), 3), np.nan, dtype=np.float32)
            for mi, model in enumerate(MODELS):
                for fold in folds:
                    zdates, ri, di, slot, values = arrays[(model, fold)]
                    which = np.array([lookup.get(zdates[d], -1) for d in di], dtype=np.int16)
                    keep = which >= 0
                    forecast[which[keep], ri[keep], slot[keep], mi] = values[keep]
            maximum = max(float(np.nanmax(np.abs(truth))) * TRUTH_SCALE,
                          float(np.nanmax(np.abs(forecast))) * FORECAST_SCALE)
            dtype = "<i2" if maximum < np.iinfo(np.int16).max else "<i4"
            missing = int(np.iinfo(dtype).min)
            raw = (time_deltas(quantize(truth, dtype, TRUTH_SCALE)).tobytes(order="C")
                   + time_deltas(quantize(forecast, dtype, FORECAST_SCALE)).tobytes(order="C"))
            name = f"{case.lower()}-{month}.bin.gz"
            target = output / name
            with target.open("wb") as stream:
                with gzip.GzipFile(fileobj=stream, mode="wb", compresslevel=9, mtime=0) as gz:
                    gz.write(raw)
            size = target.stat().st_size
            if size >= 75_000_000:
                raise ValueError(f"Shard exceeds 75 MB: {target} ({size})")
            total += size
            manifest["shards"][f"{case}/{month}"] = {
                "file": f"data/{name}", "dates": days,
                "dtype": "int16" if dtype == "<i2" else "int32", "missing": missing,
                "bytes": size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
            print(f"{name}: {size / 1_000_000:.2f} MB; {len(days)} days")
    (output / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    print(f"Total compressed: {total / 1_000_000:.2f} MB; largest shard: "
          f"{max(v['bytes'] for v in manifest['shards'].values()) / 1_000_000:.2f} MB")


if __name__ == "__main__":
    main()
