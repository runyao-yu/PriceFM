"""Zero-shot day-ahead price forecasts: TimesFM-3, Chronos-2, TabPFN-TS-3, Moirai-2.

For every region and every target day (default 2026-01-01..2026-10-01, UTC) each
model receives the PriceFM_Plus inputs (see zs_data.py) and returns q10/q50/q90
for the 96 quarter hours of the target day, in EUR/MWh.

Output, one file per model and region (written when the region finishes):
  ZERO_SHOT/forecasts/<model>/<REGION>.csv
     timestamp_utc, target_date, issue_day, q10, q50, q90, y_true
  ZERO_SHOT/forecasts/<model>/<REGION>.json   inputs, covariates, missing-value
     counts, adapter settings, timing
Finished regions are skipped on re-run (use --force to redo).

Checkpoints are the pinned Hugging Face revisions used in
RIMAC/BatteryLife_benchmark/origin_10_50_100/run_foundation_models.py.

Usage (FM environment):
  python ZERO_SHOT/run_zero_shot.py --smoke                      # 3 days, DE_LU, all models
  python ZERO_SHOT/run_zero_shot.py --models chronos_2           # all regions
  python ZERO_SHOT/run_zero_shot.py --models timesfm_3 --regions DE_LU PL
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


from . import zs_data as D

HERE = Path(__file__).resolve().parent
TABPFN_SHA256 = "48ca82019fec74f08e15d56a157bbe728d6ec25c221f2ff2d4fd22fd4e09ec6e"
QUANTILES = [0.1, 0.5, 0.9]
HORIZON = D.STEPS


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    repo_id: str
    revision: str
    checkpoint_file: str | None = None


MODEL_SPECS = {
    "timesfm_3": ModelSpec("timesfm_3", "TimesFM-3", "google/timesfm-3.0-pytorch",
                           "43046b85ec22d584a13f8098c2ed39c889e129c2"),
    "chronos_2": ModelSpec("chronos_2", "Chronos-2", "amazon/chronos-2",
                           "29ec3766d36d6f73f0696f85560a422f50e8498c"),
    "tabpfn_ts_3": ModelSpec("tabpfn_ts_3", "TabPFN-TS-3", "Prior-Labs/tabpfn_3",
                             "24a16a89d245878b846555110985634aa2e656d7",
                             "tabpfn-v3-regressor-v3_20260506_timeseries.ckpt"),
    "moirai_2_small": ModelSpec("moirai_2_small", "Moirai-2.0-R-small",
                                "Salesforce/moirai-2.0-R-small",
                                "30f43ff08c8494f4943ae1521e9d4e94a0fbb389"),
    "tirex_2": ModelSpec("tirex_2", "TiRex-2", "NX-AI/TiRex-2",
                         "30788a311699acba7e6393ad9700ea7b7cb9bb2d"),
}
DEFAULT_BATCH = {"timesfm_3": 16, "chronos_2": 32, "tabpfn_ts_3": 1,
                 "moirai_2_small": 32, "tirex_2": 32}


def download_checkpoint(spec: ModelSpec, root: Path) -> Path:
    from huggingface_hub import hf_hub_download, snapshot_download

    local = root / spec.key
    if spec.key == "tirex_2":
        if (local / "model-config.yaml").is_file() and (local / "model.ckpt").is_file():
            return local
        return Path(snapshot_download(spec.repo_id, revision=spec.revision,
                                      local_dir=local, token=os.environ.get("HF_TOKEN") or None))
    if spec.checkpoint_file and (local / spec.checkpoint_file).is_file():
        result = local / spec.checkpoint_file
        if spec.key == "tabpfn_ts_3" and hashlib.sha256(result.read_bytes()).hexdigest() != TABPFN_SHA256:
            raise ValueError("Pinned TabPFN checkpoint checksum mismatch")
        return result
    if not spec.checkpoint_file and (local / "config.json").is_file() and any(local.glob("*.safetensors")):
        return local
    local.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None
    if spec.checkpoint_file:
        result = Path(hf_hub_download(spec.repo_id, spec.checkpoint_file, revision=spec.revision,
                                      local_dir=local, token=token))
        if spec.key == "tabpfn_ts_3" and hashlib.sha256(result.read_bytes()).hexdigest() != TABPFN_SHA256:
            raise ValueError("Pinned TabPFN checkpoint checksum mismatch")
        return result
    return Path(snapshot_download(spec.repo_id, revision=spec.revision, local_dir=local, token=token))


# ---------------------------------------------------------------------------
# Adapters: predict(windows slice) -> (B, 96, 3) in the target's raw unit.
# Covariates: cov_past (B, F, Lc) with NaN after d 12:00; cov_future (B, F, 96).
# ---------------------------------------------------------------------------
class Adapter:
    missing_policy = ""

    def __init__(self, checkpoint: Path, device: str, batch_size: int):
        self.checkpoint, self.device, self.batch_size = str(checkpoint), device, int(batch_size)

    def predict(self, target, cov_past, cov_future, has_future, names, context_start):
        raise NotImplementedError

    def settings(self) -> dict:
        return {}


class TimesFM3Adapter(Adapter):
    missing_policy = ("NaN passed in; TimesFM-3 trims leading NaN and linearly interpolates "
                      "interior NaN of target and covariates (its own preprocessing), so the "
                      "d 12:00-24:00 actuals gap is bridged between d 12:00 and the lead values.")

    def __init__(self, checkpoint, device, batch_size):
        super().__init__(checkpoint, device, batch_size)
        from timesfm3 import ModelConfig, TimesFM3Evaluator

        self.model = TimesFM3Evaluator(ModelConfig(
            checkpoint_path=self.checkpoint, per_core_batch_size=self.batch_size, device=device))
        base = list(self.model.config.quantiles)
        self.index = [int(np.flatnonzero(np.isclose(base, q))[0]) for q in QUANTILES]

    def predict(self, target, cov_past, cov_future, has_future, names, context_start):
        n = len(target)
        pf = po = None
        if len(names):
            joint = np.concatenate([cov_past, cov_future], axis=-1).astype(np.float32)
            if has_future.any():
                pf = [joint[i, has_future] for i in range(n)]
            if (~has_future).any():
                po = [cov_past[i, ~has_future].astype(np.float32) for i in range(n)]
        outputs = list(self.model.predict_batch(
            contexts=[t.astype(np.float32) for t in target], horizon=HORIZON,
            past_only_covariates=po, past_future_covariates=pf,
            return_quantiles=True, make_positive=False))
        values = np.stack([np.asarray(o.quantiles) for o in outputs])
        if values.ndim == 4:      # (N, 1, H, 9) for 2-D inputs
            values = values[:, 0]
        return values[..., self.index].astype(np.float32)

    def settings(self):
        return {"api": "TimesFM3Evaluator.predict_batch", "use_symmetric_averaging": True,
                "make_positive": False, "sort_quantiles": True,
                "covariates": "past_future_covariates (lead known), past_only otherwise"}


class Chronos2Adapter(Adapter):
    missing_policy = ("NaN passed in; Chronos-2 masks missing context and covariate values "
                      "(context_mask / future_covariates_mask inferred from NaN).")

    def __init__(self, checkpoint, device, batch_size):
        super().__init__(checkpoint, device, batch_size)
        import torch
        from chronos import BaseChronosPipeline

        self.pipeline = BaseChronosPipeline.from_pretrained(
            self.checkpoint, device_map=device, torch_dtype=torch.float32)

    def predict(self, target, cov_past, cov_future, has_future, names, context_start):
        inputs = []
        for i in range(len(target)):
            item = {"target": target[i].astype(np.float32)}
            if len(names):
                item["past_covariates"] = {nm: cov_past[i, f].astype(np.float32)
                                           for f, nm in enumerate(names)}
                future = {nm: cov_future[i, f].astype(np.float32)
                          for f, nm in enumerate(names) if has_future[f]}
                if future:
                    item["future_covariates"] = future
            inputs.append(item)
        quantiles, _ = self.pipeline.predict_quantiles(
            inputs, prediction_length=HORIZON, quantile_levels=QUANTILES,
            batch_size=self.batch_size, cross_learning=False)
        return np.stack([q.float().cpu().numpy()[0] for q in quantiles]).astype(np.float32)

    def settings(self):
        return {"api": "Chronos2Pipeline.predict_quantiles", "cross_learning": False,
                "covariates": "past_covariates + future_covariates (dict API)"}


class Moirai2Adapter(Adapter):
    missing_policy = ("NaN values are set to 0 and flagged unobserved (past_observed_target, "
                      "observed_feat_dynamic_real).")
    # uni2ts 2.0 marks Moirai-2 forecasting as univariate-only; its covariate
    # path runs but degraded DE_LU March-2026 MAE from 23 to 46 EUR/MWh in our
    # check, and the official numpy predict() fails with feat_dynamic_real.
    # Default: price only. --moirai-covariates enables the experimental path.
    use_covariates = False

    def __init__(self, checkpoint, device, batch_size):
        super().__init__(checkpoint, device, batch_size)
        from uni2ts.model.moirai2 import Moirai2Module

        self.module = Moirai2Module.from_pretrained(self.checkpoint).to(device).eval()
        base = [float(q) for q in self.module.quantile_levels]
        self.index = [int(np.flatnonzero(np.isclose(base, q))[0]) for q in QUANTILES]

    def predict(self, target, cov_past, cov_future, has_future, names, context_start):
        import torch
        from uni2ts.model.moirai2 import Moirai2Forecast

        n, lc = target.shape
        if not self.use_covariates:
            has_future = np.zeros(0, bool)
            cov_past, cov_future = cov_past[:, :0], cov_future[:, :0]
        known, past_only = has_future, ~has_future
        model = Moirai2Forecast(
            module=self.module, prediction_length=HORIZON, target_dim=1,
            feat_dynamic_real_dim=int(known.sum()), past_feat_dynamic_real_dim=int(past_only.sum()),
            context_length=lc).to(self.device).eval()

        def tensor(x, dtype=torch.float32):
            return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=self.device)

        chunks = []
        for s in range(0, n, self.batch_size):
            sl = slice(s, s + self.batch_size)
            t = target[sl]
            kwargs = {
                "past_target": tensor(np.nan_to_num(t)[..., None]),
                "past_observed_target": tensor(np.isfinite(t)[..., None], torch.bool),
                "past_is_pad": tensor(np.zeros(t.shape, bool), torch.bool),
            }
            if known.any():
                joint = np.concatenate([cov_past[sl][:, known], cov_future[sl][:, known]], axis=-1)
                joint = joint.transpose(0, 2, 1)                       # (b, time, feat)
                kwargs["feat_dynamic_real"] = tensor(np.nan_to_num(joint))
                kwargs["observed_feat_dynamic_real"] = tensor(np.isfinite(joint))
            if past_only.any():
                past = cov_past[sl][:, past_only].transpose(0, 2, 1)
                kwargs["past_feat_dynamic_real"] = tensor(np.nan_to_num(past))
                kwargs["past_observed_feat_dynamic_real"] = tensor(np.isfinite(past))
            with torch.inference_mode():
                out = model(**kwargs)
            chunks.append(out.float().cpu().numpy())
        array = np.concatenate(chunks)
        if array.ndim == 4:
            array = array[..., 0]
        array = np.take(array, self.index, axis=1)                    # (N, 3, H)
        return array.transpose(0, 2, 1).astype(np.float32)

    def settings(self):
        return {"api": "Moirai2Forecast.forward (tensor path)",
                "covariates": ("feat_dynamic_real (lead known), past_feat_dynamic_real otherwise"
                               if self.use_covariates else
                               "none: price only (Moirai-2 is univariate in uni2ts 2.0)")}


class TabPFNTS3Adapter(Adapter):
    n_estimators = None  # None = TabPFN default ("auto"); --tabpfn-estimators to trade accuracy for speed

    missing_policy = ("Covariate NaN passed in (TabPFN handles missing features natively). "
                      "TabPFN-TS requires a complete target context, so missing context prices "
                      "are linearly interpolated in time (counted in the region JSON).")

    def __init__(self, checkpoint, device, batch_size):
        super().__init__(checkpoint, device, batch_size)
        os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
        from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline

        # Real timestamps: keep the TabPFN-TS default temporal features
        # (running index, calendar, auto-seasonal).
        self.pipeline = TabPFNTSPipeline(
            tabpfn_mode=TabPFNMode.LOCAL,
            tabpfn_model_config={"model_path": self.checkpoint, "device": device,
                                 **({"n_estimators": self.n_estimators} if self.n_estimators else {})})

    def predict(self, target, cov_past, cov_future, has_future, names, context_start):
        out = np.empty((len(target), HORIZON, 3), np.float32)
        for i in range(len(target)):
            out[i] = self._one(target[i], cov_past[i], cov_future[i], has_future, names,
                               context_start[i])
        gc.collect()
        return out

    def _one(self, target, cov_past, cov_future, has_future, names, start):
        lc = len(target)
        stamps = pd.date_range(start, periods=lc + HORIZON, freq="15min").tz_localize(None)
        y = pd.Series(target).interpolate(limit_direction="both").to_numpy()
        context = pd.DataFrame({"item_id": 0, "timestamp": stamps[:lc], "target": y})
        future = pd.DataFrame({"item_id": 0, "timestamp": stamps[lc:]})
        for f, name in enumerate(names):
            context[name] = cov_past[f]
            if has_future[f]:
                future[name] = cov_future[f]      # columns absent from future = past-only, unused
        result = self.pipeline.predict_df(context, future_df=future,
                                          quantiles=QUANTILES).reset_index()
        columns = []
        for q in QUANTILES:
            match = [c for c in result.columns
                     if isinstance(c, (float, int, str)) and _is_float(c) and np.isclose(float(c), q)]
            columns.append(match[0])
        result = result.sort_values("timestamp")
        return result[columns].to_numpy(np.float32)

    def settings(self):
        return {"api": "TabPFNTSPipeline.predict_df (LOCAL)",
                "n_estimators": self.n_estimators or "auto (default)",
                "temporal_features": "TabPFN-TS defaults (RunningIndex, Calendar, AutoSeasonal)",
                "covariates": "known covariates = columns in both context and future frames"}


class TiRex2Adapter(Adapter):
    missing_policy = ("Target and covariate history: forward fill then backfill within the "
                      "observed context, then zero if entirely absent. Known-future "
                      "covariates: interpolate within the supplied target-day forecast "
                      "only, then fill residual gaps with zero. An all-missing future "
                      "channel is treated as past-only for that sample.")

    def __init__(self, checkpoint, device, batch_size):
        super().__init__(checkpoint, device, batch_size)
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        from tirex2 import load_model
        self.model = load_model(self.checkpoint, device=device,
                                compile=False, use_flex_attention=False)

    @staticmethod
    def _fill_context(values):
        return pd.Series(values).ffill().bfill().fillna(0).to_numpy(np.float32)

    @staticmethod
    def _fill_future(values):
        return pd.Series(values).interpolate(limit_direction="both").fillna(0).to_numpy(np.float32)

    def predict(self, target, cov_past, cov_future, has_future, names, context_start):
        import torch
        from tirex2 import TimeseriesType
        samples = []
        for i in range(len(target)):
            past_only, future_known = [], []
            for j in range(len(names)):
                past = cov_past[i, j]
                if not np.isfinite(past).any():
                    continue
                context = self._fill_context(past)
                future = cov_future[i, j]
                if has_future[j] and np.isfinite(future).any():
                    future_known.append(np.concatenate(
                        [context, self._fill_future(future)]))
                else:
                    past_only.append(context)
            samples.append(TimeseriesType(
                target=torch.from_numpy(self._fill_context(target[i])[None, :]),
                past_covariates=(torch.from_numpy(np.stack(past_only))
                                 if past_only else None),
                future_covariates=(torch.from_numpy(np.stack(future_known))
                                   if future_known else None)))
        raw = self.model.forecast(samples, prediction_length=HORIZON,
                                  output_type="numpy", batch_size=self.batch_size)
        output = np.stack([np.asarray(item)[0, [0, 4, 8], :].T for item in raw])
        return output.astype(np.float32)

    def settings(self):
        return {"api": "tirex2.ForecastModel.forecast", "package": "tirex-2==0.3.1",
                "compile": False, "use_flex_attention": False,
                "covariates": "past-only plus target-day-known future covariates"}


def _is_float(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


ADAPTERS = {"timesfm_3": TimesFM3Adapter, "chronos_2": Chronos2Adapter,
            "tabpfn_ts_3": TabPFNTS3Adapter, "moirai_2_small": Moirai2Adapter,
            "tirex_2": TiRex2Adapter}


def write_region(out_dir: Path, w: D.Windows, prediction: np.ndarray, meta: dict) -> None:
    times = w.target_times()
    frame = pd.DataFrame({
        "timestamp_utc": times.strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "target_date": np.repeat(w.target_dates.strftime("%Y-%m-%d"), D.STEPS),
        "issue_day": np.repeat((w.target_dates - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), D.STEPS),
        "q10": prediction[..., 0].reshape(-1),
        "q50": prediction[..., 1].reshape(-1),
        "q90": prediction[..., 2].reshape(-1),
        "y_true": w.y_true.reshape(-1),
        "native_price_mask": w.native_price_mask.reshape(-1),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    temporary = out_dir / f"{w.region}.csv.tmp"
    frame.to_csv(temporary, index=False, float_format="%.6g", na_rep="")
    temporary.replace(out_dir / f"{w.region}.csv")
    meta_target = out_dir / f"{w.region}.json"
    meta_temporary = meta_target.with_suffix(".json.tmp")
    meta_temporary.write_text(json.dumps(meta, indent=2, default=str))
    meta_temporary.replace(meta_target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["chronos_2", "timesfm_3", "tabpfn_ts_3", "tirex_2"],
                        choices=list(MODEL_SPECS))
    parser.add_argument("--regions", nargs="+", default=None)
    parser.add_argument("--start", default="2026-01-01", help="first target day (UTC)")
    parser.add_argument("--end", default="2026-10-01", help="last target day (UTC, inclusive)")
    parser.add_argument("--context-days", type=int, default=7)
    parser.add_argument("--lead", choices=("actual", "dayahead"), default="actual")
    parser.add_argument("--feature-case", type=int, choices=(1, 2, 3),
                        help="New matrix: 1 realistic, 2 TSO proxy, 3 oracle; uses interpolation fill")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path.home() / ".cache" / "pricefm_fm_checkpoints")
    parser.add_argument("--output-dir", type=Path, default=HERE / "forecasts")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--tabpfn-estimators", type=int, default=None,
                        help="TabPFN ensemble size (default: TabPFN 'auto'); smaller is faster")
    parser.add_argument("--moirai-covariates", action="store_true",
                        help="feed covariates to Moirai-2 (experimental; default price only)")
    parser.add_argument("--smoke", action="store_true",
                        help="DE_LU, 3 target days, written to <output-dir>/_smoke")
    args = parser.parse_args()

    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    regions = args.regions or D.active_regions()
    start, end, root = args.start, args.end, args.output_dir
    if args.smoke:
        regions, start, end, root = (args.regions or ["DE_LU"]), "2026-03-01", "2026-03-03", root / "_smoke"
    run_tag = (f"feature_case{args.feature_case}_interp_ctx{args.context_days}d"
               if args.feature_case else f"ctx{args.context_days}d_lead-{args.lead}")
    print(f"torch={torch.__version__} threads={torch.get_num_threads()} python={platform.python_version()} "
          f"regions={len(regions)} days={start}..{end} {run_tag}", flush=True)

    for key in args.models:
        spec = MODEL_SPECS[key]
        out_dir = root / run_tag / (key + ("_cov" if key == "moirai_2_small" and args.moirai_covariates else ""))
        pending = [r for r in regions if args.force or
                   not ((out_dir / f"{r}.json").is_file() and
                        (out_dir / f"{r}.csv").is_file())]
        if not pending:
            print(f"[{spec.label}] all regions done", flush=True)
            continue
        checkpoint = download_checkpoint(spec, args.checkpoint_dir.resolve())
        batch = args.batch_size or DEFAULT_BATCH[key]
        if key == "tabpfn_ts_3":
            TabPFNTS3Adapter.n_estimators = args.tabpfn_estimators
        adapter = ADAPTERS[key](checkpoint, args.device, batch)
        if key == "moirai_2_small":
            adapter.use_covariates = args.moirai_covariates
        for region in pending:
            w = D.build_windows(region, start, end, context_days=args.context_days,
                                lead=args.lead, feature_case=args.feature_case)
            started = time.perf_counter()
            prediction = np.full((len(w.target_dates), D.STEPS, 3), np.nan, np.float32)
            for s in range(0, len(w.target_dates), batch):
                sl = slice(s, s + batch)
                prediction[sl] = adapter.predict(w.target_past[sl], w.cov_past[sl], w.cov_future[sl],
                                                 w.has_future, w.cov_names, w.context_start[sl])
            seconds = time.perf_counter() - started
            finite = np.isfinite(prediction)
            meta = {
                "region": region, "model": spec.label, "model_key": key, "zero_shot": True,
                "repo_id": spec.repo_id, "revision": spec.revision,
                "checkpoint_file": spec.checkpoint_file, "device": args.device,
                "batch_size": batch, "torch_threads": torch.get_num_threads(),
                "target_days": [str(w.target_dates[0].date()), str(w.target_dates[-1].date())],
                "n_days": len(w.target_dates), "quantiles": QUANTILES, "unit": "EUR/MWh",
                "context_days": args.context_days,
                "feature_case": args.feature_case,
                "fill_variant": "interp" if args.feature_case else "copy",
                "inputs": {"price": "C days ending issue day 24:00 UTC",
                           "actuals": "same window, cut off at issue day 12:00 UTC",
                           "lead": ({1: "none", 2: "TSO forecasts plus commercial schedules",
                                     3: "target-day actuals"}.get(args.feature_case,
                                     f"target day ({args.lead})"))},
                "covariates_used": ([name for name, known in zip(w.cov_names, w.has_future)
                                     if known] if key == "tabpfn_ts_3" else
                                    w.cov_names if key != "moirai_2_small" or
                                    args.moirai_covariates else []),
                "covariates_with_lead": [n for n, h in zip(w.cov_names, w.has_future) if h],
                "covariates_dropped": w.dropped, "missing_counts": w.missing,
                "missing_policy": adapter.missing_policy,
                "adapter_settings": adapter.settings(),
                "prediction_nonfinite": int((~finite).sum()),
                "quantile_crossing_share": float(np.mean(np.any(np.diff(prediction, axis=-1) < 0, axis=-1))),
                "inference_seconds": seconds, "seconds_per_forecast": seconds / len(w.target_dates),
            }
            write_region(out_dir, w, prediction, meta)
            print(f"[{spec.label}] {region}: {len(w.target_dates)} days, {seconds:.1f}s "
                  f"({seconds / len(w.target_dates):.2f}s/day), covariates={meta['covariates_used']}", flush=True)
        del adapter
        gc.collect()
    print("done", flush=True)


if __name__ == "__main__":
    main()
