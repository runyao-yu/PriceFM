"""Daily forecast windows for zero-shot day-ahead price forecasting.

Inputs mirror PriceFM_Plus (case 4) on a 15-minute UTC grid. For target day D
(UTC) issued on day d = D - 1:

  price context   C days ending at d 24:00 (= D 00:00)
  actual inputs   load, solar, wind onshore, wind offshore, import sum,
                  export sum over the same C days, cut off at d 12:00
                  (the 48 quarter hours d 12:00-24:00 are unknown: NaN)
  lead inputs     the same six series over day D
                    lead="actual":   target-day actuals (PriceFM_Plus case 4)
                    lead="dayahead": day-ahead load/solar/wind forecasts for D
                                     (flows have no forecast; they stay past-only)
  target          96 quarter-hour prices of D (EUR/MWh), returned as y_true

Values are in raw units (MW, EUR/MWh) from DATASETS/processed/per_region_feature
(native hourly values copied to their quarter hours; A03 blocks expanded; true
gaps NaN). Nothing is imputed here; each model adapter declares how it treats
NaN.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
SOURCE = ROOT / "Data" / "Processed" / "per_region_feature"
INTERP_SOURCE = ROOT / "Data" / "Processed" / "per_region_feature_interp"
ACTUALS = ("ActualLoad", "ActualSolar", "ActualWindOnshore", "ActualWindOffshore",
           "ActualPowerFlowImport", "ActualPowerFlowExport")
DAYAHEAD = {"ActualLoad": "DayAheadLoad", "ActualSolar": "DayAheadSolar",
            "ActualWindOnshore": "DayAheadWindOnshore",
            "ActualWindOffshore": "DayAheadWindOffshore"}
DAYAHEAD_WITH_SCHEDULES = {**DAYAHEAD,
    "ActualPowerFlowImport": "DayAheadPowerFlowImport",
    "ActualPowerFlowExport": "DayAheadPowerFlowExport"}
STEPS = 96          # quarter hours per day
CUTOFF = 48         # actuals known up to d 12:00 -> first 48 slots of day d


def active_regions():
    return json.loads((ROOT / "Data" / "Config" / "active_regions.json").read_text())


def _series(region: str, name: str, grid: pd.DatetimeIndex,
            source: Path | None = None) -> np.ndarray:
    path = (source or SOURCE) / f"{region}_{name}.csv"
    if not path.exists():
        return np.full(len(grid), np.nan)
    frame = pd.read_csv(path, usecols=["timestamp_utc", "value"])
    index = pd.DatetimeIndex(pd.to_datetime(frame["timestamp_utc"], utc=True))
    values = pd.Series(pd.to_numeric(frame["value"], errors="coerce").to_numpy(float),
                       index=index)
    return values.reindex(grid).to_numpy(float)


@dataclass
class Windows:
    region: str
    target_dates: pd.DatetimeIndex
    target_past: np.ndarray            # (B, Lc)            price context
    cov_names: list                    # covariates kept for this region
    cov_past: np.ndarray               # (B, F, Lc)         NaN after d 12:00
    cov_future: np.ndarray             # (B, F, 96)         lead values (NaN if unknown)
    has_future: np.ndarray             # (F,) bool          covariate has a lead
    y_true: np.ndarray                 # (B, 96)
    context_start: pd.DatetimeIndex    # first context timestamp per window
    native_price_mask: np.ndarray      # (B, 96), excludes filled price slots
    dropped: dict = field(default_factory=dict)
    missing: dict = field(default_factory=dict)

    def target_times(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(np.concatenate([
            pd.date_range(day, periods=STEPS, freq="15min", tz="UTC").values
            for day in self.target_dates])).tz_localize("UTC")


def build_windows(region: str, first: str, last: str, *, context_days: int = 7,
                  lead: str = "actual", min_coverage: float = .5,
                  feature_case: int | None = None) -> Windows:
    """Windows for target days first..last (inclusive, UTC)."""
    if lead not in ("actual", "dayahead"):
        raise ValueError(lead)
    if feature_case not in (None, 1, 2, 3):
        raise ValueError(feature_case)
    if feature_case is not None:
        # Keep every series that exists during the evaluation window. An
        # absent channel remains missing rather than silently changing the
        # feature-case definition for sparsely observed regions.
        min_coverage = min(min_coverage, .001)
    source = INTERP_SOURCE if feature_case else SOURCE
    if feature_case == 1:
        lead = "none"
    elif feature_case == 2:
        lead = "dayahead"
    elif feature_case == 3:
        lead = "actual"
    days = pd.date_range(first, last, freq="D", tz="UTC")
    grid = pd.date_range(days[0] - pd.Timedelta(days=context_days),
                         days[-1] + pd.Timedelta(days=1), freq="15min",
                         inclusive="left", tz="UTC")
    price = _series(region, "DayAheadPrice", grid, source)
    price_frame = pd.read_csv(source / f"{region}_DayAheadPrice.csv",
                              usecols=["timestamp_utc", "source_observed"])
    observed = pd.Series(
        price_frame["source_observed"].astype(bool).to_numpy(),
        index=pd.DatetimeIndex(pd.to_datetime(price_frame["timestamp_utc"], utc=True)))
    native_price = observed.reindex(grid, fill_value=False).to_numpy(bool)
    actual = {name: _series(region, name, grid, source) for name in ACTUALS}
    forecast_names = DAYAHEAD_WITH_SCHEDULES if feature_case == 2 else DAYAHEAD
    forecast = ({name: _series(region, forecast_names[name], grid, source)
                 for name in forecast_names}
                if lead == "dayahead" else {})

    # Keep a covariate only if it is observed in at least `min_coverage` of the
    # forecast period's grid; a series that never or barely exists (no offshore
    # wind, PL offshore starting 2026-07) would otherwise be mostly mask.
    names, past_rows, lead_rows, has_future, dropped = [], [], [], [], {}
    for name in ACTUALS:
        share = float(np.isfinite(actual[name]).mean())
        if share < min_coverage:
            dropped[name] = f"observed share {share:.3f} < {min_coverage}"
            continue
        names.append(name)
        past_rows.append(actual[name])
        if lead == "actual":
            lead_rows.append(actual[name]); has_future.append(True)
        elif name in forecast and np.isfinite(forecast[name]).mean() >= min_coverage:
            lead_rows.append(forecast[name]); has_future.append(True)
        else:
            lead_rows.append(np.full(len(grid), np.nan)); has_future.append(False)

    lc = context_days * STEPS
    starts = ((days - grid[0]) // pd.Timedelta(minutes=15)).to_numpy() - lc
    past_idx = starts[:, None] + np.arange(lc)[None, :]
    fut_idx = starts[:, None] + lc + np.arange(STEPS)[None, :]
    target_past = price[past_idx]
    y_true = price[fut_idx]
    native_target_mask = native_price[fut_idx] & np.isfinite(y_true)
    if names:
        past = np.stack(past_rows)[:, past_idx].transpose(1, 0, 2).copy()
        future = np.stack(lead_rows)[:, fut_idx].transpose(1, 0, 2).copy()
    else:
        past = np.empty((len(days), 0, lc)); future = np.empty((len(days), 0, STEPS))
    gap = past[:, :, lc - (STEPS - CUTOFF):]
    missing = {
        "target_context_nan": int(np.isnan(target_past).sum()),
        "target_context_slots": int(target_past.size),
        "y_true_nan": int(np.isnan(y_true).sum()),
        "covariate_context_nan_before_cutoff": int(np.isnan(past[:, :, :lc - (STEPS - CUTOFF)]).sum()),
        "covariate_lead_nan": int(np.isnan(future[:, np.array(has_future, bool)]).sum()) if names else 0,
    }
    gap[:] = np.nan  # d 12:00-24:00 actuals are not yet known at issue time
    return Windows(region, days, target_past, names, past, future,
                   np.array(has_future, bool), y_true,
                   grid[starts], native_target_mask, dropped, missing)
