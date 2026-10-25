"""Download, scoring and public entry points. All paths are under the current directory."""

from __future__ import annotations

import csv
import getpass
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

TRAINED = ("MLP", "LSTM", "Transformer", "PriceFM_Plus")
FOUNDATION = ("TabPFN_TS_3", "TiRex_2", "TimesFM_3", "Chronos_2")
CASES = ("Realistic", "TSO", "Oracle")
DEFAULT_REPO_ID = "RunyaoYu/European_Foundation_Model_DAM"
ALIASES = {name.lower().replace("-", "_"): name for name in (*TRAINED, *FOUNDATION)}
ALIASES.update({"pricefm+": "PriceFM_Plus", "tabpfn_ts3": "TabPFN_TS_3",
                "tirex2": "TiRex_2", "timesfm3": "TimesFM_3", "chronos2": "Chronos_2"})


def _model(name: str) -> str:
    try:
        return ALIASES[name.lower().replace("-", "_")]
    except KeyError as exc:
        raise ValueError(f"Unknown model {name!r}. Choose from {(*TRAINED, *FOUNDATION)}") from exc


def _repository(repo_id: str | None) -> str:
    return repo_id or os.environ.get("PRICEFM_PLUS_HF_REPO") or DEFAULT_REPO_ID


def _extract(archive: Path, destination: Path) -> Path:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            raise ValueError(f"ZIP CRC mismatch: {bad}")
        for member in z.infolist():
            path = (destination / member.filename).resolve()
            if not path.is_relative_to(destination) or member.is_dir():
                if member.is_dir() and path.is_relative_to(destination):
                    continue
                raise ValueError(f"Unsafe archive member: {member.filename}")
            if path.exists() and path.stat().st_size == member.file_size:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as source, path.open("wb") as target:
                import shutil
                shutil.copyfileobj(source, target)
    return destination


def _download(filename: str, folder: str, *, repo_id: str | None = None,
              archive_path: str | Path | None = None) -> Path:
    if archive_path is None:
        from huggingface_hub import hf_hub_download
        archive = Path(hf_hub_download(repo_id=_repository(repo_id), filename=filename,
                                       repo_type="dataset"))
    else:
        archive = Path(archive_path)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    return _extract(archive, Path.cwd() / folder)


def download_data(kind: str, *, repo_id: str | None = None,
                  archive_path: str | Path | None = None) -> Path:
    """Download raw or processed data into ``./Data``."""
    kind = kind.lower()
    if kind not in ("raw", "processed"):
        raise ValueError("kind must be 'raw' or 'processed'")
    return _download(f"{kind}_data.zip", "Data", repo_id=repo_id,
                     archive_path=archive_path)


def download_weights(model_name: str, *, repo_id: str | None = None,
                     archive_path: str | Path | None = None) -> Path:
    """Download model weights into ``./Weights``."""
    model = _model(model_name)
    if model in ("TabPFN_TS_3", "TimesFM_3") and archive_path is None:
        return _foundation_checkpoint(model, ask=False)
    return _download(f"weights_{model}.zip", "Weights", repo_id=repo_id,
                     archive_path=archive_path)


def download_forecasts(model_name: str, *, repo_id: str | None = None,
                       archive_path: str | Path | None = None) -> Path:
    """Download native-price forecasts into ``./Forecasts``."""
    model = "Persistence" if model_name.lower() == "persistence" else _model(model_name)
    return _download(f"forecasts_{model}.zip", "Forecasts", repo_id=repo_id,
                     archive_path=archive_path)


def _foundation_checkpoint(model: str, *, ask: bool) -> Path:
    from .foundation_models import MODEL_SPECS, download_checkpoint
    keys = {"TabPFN_TS_3": "tabpfn_ts_3", "TiRex_2": "tirex_2",
            "TimesFM_3": "timesfm_3", "Chronos_2": "chronos_2"}
    spec = MODEL_SPECS[keys[model]]
    folder = Path.cwd() / "Weights" / "FoundationModels" / spec.key
    exists = (folder / spec.checkpoint_file).is_file() if spec.checkpoint_file else (
        (folder / "config.json").is_file() and any(folder.glob("*.safetensors")))
    if spec.key == "tirex_2":
        exists = (folder / "model-config.yaml").is_file() and (folder / "model.ckpt").is_file()
    if ask and not exists:
        answer = input(f"{model} checkpoint is absent. Download pinned weights from "
                       f"{spec.repo_id}? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            raise PermissionError(f"Checkpoint download declined for {model}")
    try:
        return download_checkpoint(spec, folder.parent)
    except Exception as exc:
        if "401" not in str(exc) and "403" not in str(exc) and "gated" not in str(exc).lower():
            raise
        token = getpass.getpass("Hugging Face access token for this model (hidden): ")
        if not token:
            raise
        previous = os.environ.get("HF_TOKEN")
        os.environ["HF_TOKEN"] = token
        try:
            return download_checkpoint(spec, folder.parent)
        finally:
            if previous is None:
                os.environ.pop("HF_TOKEN", None)
            else:
                os.environ["HF_TOKEN"] = previous


def add_data(start_date: str, end_date: str, entsoe_api_key: str) -> dict:
    """Extend the existing raw and three processed variants in ``./Data``."""
    from .data_pipeline.core import ALL_FEATURES, load_config
    from .data_pipeline.sync import sync_dataset
    if not entsoe_api_key:
        raise ValueError("ENTSO-E API key is required")
    if not (Path.cwd() / "Data" / "Config" / "active_regions.json").is_file():
        raise FileNotFoundError("Download processed data before add_data")
    regions = load_config()[0]
    previous = os.environ.get("ENTSOE_API_TOKEN")
    os.environ["ENTSOE_API_TOKEN"] = entsoe_api_key
    try:
        results = {}
        for fill in ("copy", "interp", "none"):
            results[fill] = sync_dataset(regions, start_date, end_date,
                                         ALL_FEATURES, fill=fill,
                                         download=fill == "copy")
        return results
    finally:
        if previous is None:
            os.environ.pop("ENTSOE_API_TOKEN", None)
        else:
            os.environ["ENTSOE_API_TOKEN"] = previous


def reproduce_forecasts(model_name: str, *, regions: list[str] | None = None,
                        cases: list[str] | None = None, folds=(1, 2, 3),
                        batch_days: int = 32) -> list[Path]:
    """Run the archived joint models on the original held-out test folds."""
    model = _model(model_name)
    from ._inference import reproduce_foundation, reproduce_trained
    selected = cases or list(CASES)
    if any(case not in CASES for case in selected):
        raise ValueError(f"cases must be selected from {CASES}")
    if model in FOUNDATION:
        _foundation_checkpoint(model, ask=True)
        return reproduce_foundation(model, regions=regions, cases=selected,
                                    folds=folds)
    return reproduce_trained(model, regions=regions, cases=selected,
                             folds=folds, batch_days=batch_days)


def _observations(region: str, cache: dict) -> dict[np.datetime64, np.ndarray]:
    if region not in cache:
        path = (Path.cwd() / "Data" / "Processed" / "per_region_feature_interp" /
                f"{region}_DayAheadPrice.csv")
        if not path.is_file():
            raise FileNotFoundError(f"Observed prices absent: {path}. Download processed data.")
        frame = pd.read_csv(path, usecols=["timestamp_utc", "value", "source_observed"])
        stamps = pd.to_datetime(frame["timestamp_utc"], utc=True)
        values = pd.to_numeric(frame["value"], errors="coerce").to_numpy(dtype=np.float32)
        native = frame["source_observed"].to_numpy(dtype=bool) & np.isfinite(values)
        days = stamps.to_numpy(dtype="datetime64[D]")
        slots = (stamps.dt.hour.to_numpy() * 4 + stamps.dt.minute.to_numpy() // 15)
        result = {}
        for day in np.unique(days):
            if day < np.datetime64("2026-01-01"):
                continue
            row = np.full(96, np.nan, dtype=np.float32)
            selected = (days == day) & native
            row[slots[selected]] = values[selected]
            result[day] = row
        cache[region] = result
    return cache[region]


def _score_file(path: Path, cache: dict) -> tuple[dict, pd.DataFrame]:
    with np.load(path, allow_pickle=False) as z:
        dates = z["dates"].astype("datetime64[D]")
        regions = z["regions"].astype(str)
        ri, di, slots = [z[key].astype(np.int32) for key in
                         ("region_index", "day_index", "slot_utc")]
        prediction = z["prediction_EUR_per_MWh"].astype(np.float64)
    if prediction.shape != (len(ri), 3) or len(di) != len(ri) or len(slots) != len(ri):
        raise ValueError(f"Invalid forecast dimensions: {path}")
    if (ri.min() < 0 or ri.max() >= len(regions) or di.min() < 0 or
            di.max() >= len(dates) or slots.min() < 0 or slots.max() > 95):
        raise ValueError(f"Invalid forecast indices: {path}")
    truth = np.empty(len(ri), dtype=np.float64)
    for index, region in enumerate(regions):
        mask = ri == index
        obs = _observations(region, cache)
        matrix = np.stack([obs.get(day, np.full(96, np.nan)) for day in dates])
        truth[mask] = matrix[di[mask], slots[mask]]
    valid = np.isfinite(truth) & np.isfinite(prediction).all(axis=1)
    if not valid.all():
        raise ValueError(f"Forecast has {len(valid) - valid.sum()} slots without native truth: {path}")
    error = truth[:, None] - prediction
    q = np.array([.1, .5, .9])
    pinball = np.maximum(q * error, (q - 1.) * error).mean(axis=1)
    interval = prediction[:, 2] - prediction[:, 0] + 10 * np.maximum(
        prediction[:, 0] - truth, 0) + 10 * np.maximum(truth - prediction[:, 2], 0)
    absolute = np.abs(error[:, 1])
    squared = error[:, 1] ** 2
    daily = pd.DataFrame({"region": regions[ri], "date": dates[di],
                          "pinball": pinball, "interval_score": interval,
                          "MAE": absolute, "MSE": squared})
    daily = daily.groupby(["region", "date"], sort=False, as_index=False).mean()
    means = daily[["pinball", "interval_score", "MAE", "MSE"]].mean()
    return {"pinball": float(means["pinball"]),
            "interval_score": float(means["interval_score"]),
            "MAE": float(means["MAE"]), "MSE": float(means["MSE"]),
            "region_days": len(daily)}, daily


def reproduce_metrics(*, forecast_root: str | Path = "Forecasts",
                      output_path: str | Path = "Metrics/table3.csv") -> pd.DataFrame:
    """Score existing forecasts only; never load weights or run inference."""
    root = Path(forecast_root)
    if not (root / "Persistence" / "Persistence_1" / "fold1.npz").is_file():
        raise FileNotFoundError("Persistence forecast archive is absent. "
                                "Call download_forecasts('Persistence') first.")
    observation_cache: dict = {}
    rows = []
    for model in ("Persistence_1", "Persistence_2", "Persistence_3", *TRAINED, *FOUNDATION):
        for case in CASES:
            group = ("Joint/L2" if model == "PriceFM_Plus" else
                     "Joint/Include_2022" if model in TRAINED else "")
            model_folder = "Persistence" if model.startswith("Persistence_") else model
            daily_parts = []
            for fold in (1, 2, 3):
                path = (root / "Persistence" / model / f"fold{fold}.npz"
                        if model.startswith("Persistence_") else
                        root / model_folder / case / group / f"fold{fold}.npz")
                if not path.is_file():
                    raise FileNotFoundError(f"Missing forecast: {path}. Download forecasts first.")
                _, daily = _score_file(path, observation_cache)
                daily_parts.append(daily)
            daily = pd.concat(daily_parts, ignore_index=True)
            means = daily[["pinball", "interval_score", "MAE", "MSE"]].mean()
            rows.append({"model": model, "feature_case": case,
                         "Pinball": float(means["pinball"]),
                         "Interval Score": float(means["interval_score"]),
                         "MAE": float(means["MAE"]),
                         "RMSE": float(np.sqrt(means["MSE"])),
                         "region_days": len(daily)})
    result = pd.DataFrame(rows)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(target, index=False, float_format="%.8f")
    print(result.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    return result


def reproduce_figures(*, repo_id: str | None = None,
                      archive_path: str | Path | None = None) -> Path:
    """Regenerate five grouped figures and four ablation panels in ``./Figures``.

    The released plot inputs and plotting source are fetched on demand. Install
    ``PriceFM_Plus[figures]`` for the plotting dependencies.
    """
    if archive_path is None:
        from huggingface_hub import hf_hub_download
        archive = Path(hf_hub_download(repo_id=_repository(repo_id),
                                       filename="figure_source.zip",
                                       repo_type="dataset"))
    else:
        archive = Path(archive_path)
    from ._figures import regenerate
    return regenerate(archive, Path.cwd() / "Figures")


def customize(dates, feature_case="Realistic", regions=None, hyperparams=None):
    """Train, validate, test and score a custom PriceFM_Plus configuration."""
    from ._training import customize_model
    return customize_model(dates, feature_case, regions, hyperparams)
