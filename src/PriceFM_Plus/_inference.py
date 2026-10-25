"""Inference from saved joint weights and pinned zero-shot foundation checkpoints."""

from __future__ import annotations

import gzip
from datetime import date, timedelta
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

from ._api import CASES

INPUT_KEYS = ("lag_values", "lag_mask", "lead_values", "lead_mask",
              "price_values", "price_mask", "region_code")
FOLD_DATES = {1: ("2026-01-01", "2026-03-31"),
              2: ("2026-04-01", "2026-06-30"),
              3: ("2026-07-01", "2026-09-30")}
FM_KEYS = {"TabPFN_TS_3": "tabpfn_ts_3", "TiRex_2": "tirex_2",
           "TimesFM_3": "timesfm_3", "Chronos_2": "chronos_2"}


@lru_cache(maxsize=1)
def _saved_inference_metadata():
    """Small, fixed preprocessing settings bundled with the package."""
    source = files("PriceFM_Plus").joinpath("inference_metadata.json.gz")
    with source.open("rb") as stream:
        import json
        return json.loads(gzip.decompress(stream.read()))


def _save(path: Path, dates, regions, region_index, day_index, slots, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    utc_dates = pd.DatetimeIndex(dates)
    if utc_dates.tz is not None:
        utc_dates = utc_dates.tz_convert("UTC").tz_localize(None)
    np.savez_compressed(path, dates=utc_dates.to_numpy(dtype="datetime64[D]"),
                        regions=np.asarray(regions),
                        region_index=np.concatenate(region_index).astype(np.uint8),
                        day_index=np.concatenate(day_index).astype(np.uint8),
                        slot_utc=np.concatenate(slots).astype(np.uint8),
                        prediction_EUR_per_MWh=np.concatenate(values).astype(np.float32))
    return path


def _weight_folder(model: str, case: str, fold: int) -> Path:
    section = "Joint/L2" if model == "PriceFM_Plus" else "Joint/Include_2022"
    return Path.cwd() / "Weights" / model / case / section / f"fold{fold}"


def reproduce_trained(model: str, *, regions, cases, folds, batch_days: int):
    import tensorflow as tf
    from tqdm.auto import tqdm
    from . import data as D
    from .PriceFM_Plus import build_model
    from .region_connectivity import load_connectivity
    from .simple_baselines import build_simple_model

    if batch_days < 1:
        raise ValueError("batch_days must be positive")
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    model_regions, adjacency = load_connectivity()
    outputs = []
    for case in cases:
        for fold in folds:
            if fold not in FOLD_DATES:
                raise ValueError("Only folds 1–3 have held-out forecasts")
            folder = _weight_folder(model, case, fold)
            metadata = _saved_inference_metadata()[model][case][str(fold)]
            protocol = metadata["protocol"]
            saved = metadata["scalers"]
            if protocol["model_regions"] != list(model_regions):
                raise ValueError("Weight topology differs from the downloaded processed data")
            selected = tuple(regions or protocol["regions"])
            if any(region not in protocol["regions"] for region in selected):
                raise ValueError(f"Some selected regions lack saved {model} weights")
            start, end = FOLD_DATES[fold]
            D.set_variant(protocol["variant"])
            D.SPLITS = dict(protocol["split"])
            D.SPLITS["inference"] = (start, (date.fromisoformat(end) + timedelta(days=1)).isoformat())
            D.TARGET_DATES = pd.date_range(start, end, freq="D", tz="UTC")
            D.SPLIT_SCHEME = f"infer_{start}_{end}"
            D.SCALER_MODE = protocol["scaler_mode"]
            D.GLOBAL_SCALERS = (dict(saved[protocol["regions"][0]])
                                if D.SCALER_MODE != "regional_robust" else {})
            D._refresh_cache()
            if model == "PriceFM_Plus":
                network = build_model(model_regions, loops=int(protocol["loops"]),
                                      hidden_width=int(protocol["hidden_width"]),
                                      extra_depth=bool(protocol["extra_depth"]),
                                      temporal_mix=bool(protocol["temporal_mix"]),
                                      recent_price_context=bool(protocol["recent_price_context"]),
                                      recent_price_sequence=bool(protocol.get("recent_price_sequence", False)),
                                      branch_dense_layers=int(protocol["branch_dense_layers"]),
                                      head_dense_layers=int(protocol["head_dense_layers"]))
            else:
                family = {"MLP": "SimpleMLP", "LSTM": "SimpleLSTM",
                          "Transformer": "SimpleTransformer"}[model]
                network = build_simple_model(family,
                                             lstm_architecture="patch8" if model == "LSTM" else "original")
            region_index, day_index, slots, values = [], [], [], []
            for index, region in enumerate(tqdm(selected, desc=f"{model} {case} fold{fold}")):
                data, _, scalers, _ = D.build_case_region(region, protocol["case"], model_regions)
                for name, reference in saved[region].items():
                    current = scalers[name]
                    if not (np.isclose(current["center"], reference["center"], rtol=1e-6, atol=1e-5)
                            and np.isclose(current["scale"], reference["scale"], rtol=1e-6, atol=1e-5)):
                        raise ValueError(f"Saved scaler mismatch for {region} {name}")
                data["region_code"] = np.broadcast_to(adjacency[model_regions.index(region)],
                                                      (len(D.TARGET_DATES), len(model_regions))).copy()
                weight_path = folder / f"best_{region}.weights.h5"
                network.load_weights(weight_path)
                predictions = []
                for first in range(0, len(D.TARGET_DATES), batch_days):
                    last = min(first + batch_days, len(D.TARGET_DATES))
                    inputs = {key: tf.convert_to_tensor(data[key][first:last]) for key in INPUT_KEYS}
                    predictions.append(network(inputs, training=False).numpy())
                raw = np.concatenate(predictions)
                price = saved[region]["DayAheadPrice"]
                raw = raw * price["scale"] + price["center"]
                mask = data["eval_mask"] & np.isfinite(raw).all(axis=-1)
                days, positions = np.nonzero(mask)
                region_index.append(np.full(len(days), index))
                day_index.append(days)
                slots.append(positions)
                values.append(raw[days, positions])
            out = Path.cwd() / "Forecasts" / model / case / "Joint" / (
                "L2" if model == "PriceFM_Plus" else "Include_2022") / f"fold{fold}.npz"
            outputs.append(_save(out, D.TARGET_DATES.to_numpy(), selected,
                                 region_index, day_index, slots, values))
            tf.keras.backend.clear_session()
    return outputs


def reproduce_foundation(model: str, *, regions, cases, folds):
    import torch
    from tqdm.auto import tqdm
    from . import foundation_models as FM
    from . import zs_data as D

    key = FM_KEYS[model]
    spec = FM.MODEL_SPECS[key]
    checkpoint = FM.download_checkpoint(spec, Path.cwd() / "Weights" / "FoundationModels")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = FM.ADAPTERS[key](checkpoint, device, FM.DEFAULT_BATCH[key])
    selected = tuple(regions or D.active_regions())
    outputs = []
    for case in cases:
        for fold in folds:
            if fold not in FOLD_DATES:
                raise ValueError("Only folds 1–3 have held-out forecasts")
            start, end = FOLD_DATES[fold]
            dates = pd.date_range(start, end, freq="D", tz="UTC")
            region_index, day_index, slots, values = [], [], [], []
            for index, region in enumerate(tqdm(selected, desc=f"{model} {case} fold{fold}")):
                windows = D.build_windows(region, start, end, context_days=7,
                                          feature_case=CASES.index(case) + 1)
                predictions = np.full((len(dates), 96, 3), np.nan, np.float32)
                batch = FM.DEFAULT_BATCH[key]
                for first in range(0, len(dates), batch):
                    sl = slice(first, first + batch)
                    predictions[sl] = adapter.predict(
                        windows.target_past[sl], windows.cov_past[sl],
                        windows.cov_future[sl], windows.has_future,
                        windows.cov_names, windows.context_start[sl])
                mask = windows.native_price_mask & np.isfinite(predictions).all(axis=-1)
                days, positions = np.nonzero(mask)
                region_index.append(np.full(len(days), index))
                day_index.append(days)
                slots.append(positions)
                values.append(predictions[days, positions])
            out = Path.cwd() / "Forecasts" / model / case / f"fold{fold}.npz"
            outputs.append(_save(out, dates.to_numpy(), selected,
                                 region_index, day_index, slots, values))
    return outputs
