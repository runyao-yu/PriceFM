"""Rebuild quarter-hour features from archived native-resolution XML."""

from __future__ import annotations

import datetime as dt
import gzip
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from .core import (FEATURE_TO_FAMILY, INTERVAL, PROCESSED, PSR_FEATURE,
                  RAW, iso, load_config, sha256, utc_datetime)
from .download import load_index


OUTPUT = PROCESSED / "per_region_feature"
# How a native hourly value fills the three following quarter hours.
#   copy:   repeat the hourly value (default; the original behaviour)
#   interp: linear interpolation towards the next whole-hour value; left
#           blank when the next whole-hour slot has no reported value
#   none:   no filling; only natively reported slots carry a value
FILL_MODES = ("copy", "interp", "none")
EUR_PEG = {"BGN": 1.95583}
OUTPUTS = {"copy": OUTPUT,
           "interp": PROCESSED / "per_region_feature_interp",
           "none": PROCESSED / "per_region_feature_nofill"}


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def child(node, name):
    return next((part for part in node if local_name(part.tag) == name), None)


def descendant(node, name):
    return next((part for part in node.iter() if local_name(part.tag) == name), None)


def child_text(node, name):
    part = child(node, name) if node is not None else None
    return part.text.strip() if part is not None and part.text else ""


def resolution_minutes(text):
    matched = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", text)
    if matched is None:
        return 0
    hours, minutes, seconds = (int(part or 0) for part in matched.groups())
    return hours * 60 + minutes + seconds // 60


def effective_minutes(native_minutes, period_start, period_end, positions):
    """Recognize an hourly series mislabeled as PT15M in the XML period."""
    if native_minutes != 15:
        return native_minutes
    positions = set(positions)
    hours = (period_end - period_start).total_seconds() / 3600
    if hours < 8 or len(positions) < max(8, math.ceil(.75 * hours)):
        return native_minutes
    stamps = [period_start + dt.timedelta(minutes=15 * (p - 1))
              for p in positions]
    if all(stamp.minute == 0 and stamp.second == 0 for stamp in stamps):
        return 60
    return native_minutes


def a03_blocks(period_points, step_minutes, period_end):
    """Expand ENTSO-E curve type A03 (variable-sized blocks).

    In A03 a point holds until the next reported position, so an omitted
    position repeats the previous value; it is not missing data. Blocks are
    expanded on the effective resolution grid (hourly for whole-hour PT15M
    periods), so hourly expansion to quarter hours stays a separate step.
    """
    ordered = sorted(period_points, key=lambda item: item[1])
    step = dt.timedelta(minutes=step_minutes)
    expanded = []
    for number, (ordinal, stamp, value) in enumerate(ordered):
        until = ordered[number + 1][1] if number + 1 < len(ordered) else period_end
        cursor = stamp
        while cursor < until:
            expanded.append((ordinal, cursor, value))
            cursor += step
    return expanded


def source_entries(index, region, family, neighbor, start, end):
    return [row for row in index if row["region"] == region and
            row["family"] == family and row["neighbor"] == neighbor and
            utc_datetime(row["start"]) < end and
            utc_datetime(row["end"]) > start]


def parse_source(body, feature, domain=None, counterparty_domain=None):
    """Return explicit (time, value, native minutes, period end) and periods."""
    root = ET.fromstring(body)
    points, periods = [], []
    for series in (part for part in root.iter()
                   if local_name(part.tag) == "TimeSeries"):
        if feature in ("ActualSolar", "ActualWindOnshore",
                       "ActualWindOffshore"):
            in_zone = child_text(series, "inBiddingZone_Domain.mRID")
            out_zone = child_text(series, "outBiddingZone_Domain.mRID")
            # A75 may return both generation entering the requested zone
            # and an out-zone counterpart under the same PSR code. The
            # latter is not that region's generation.
            if (in_zone and domain is not None and in_zone != domain) or \
                    (out_zone and not in_zone):
                continue
        if feature == "DayAheadPrice":
            # A44 can carry several price sequences for one zone (DE_LU, AT:
            # sequence 1 = SDAC day-ahead, sequence 2 = a separate 15-min
            # auction). Keep SDAC only; series without a sequence are kept.
            sequence = child_text(series,
                                  "classificationSequence_AttributeInstanceComponent.position")
            if sequence and sequence != "1":
                continue
            # Day-ahead contracts only: ES also publishes its intraday
            # auctions (contract type A07, sequences 1-3) under A44.
            contract = child_text(series, "contract_MarketAgreement.type")
            if contract and contract != "A01":
                continue
        if feature in ("DayAheadPowerFlowImport", "DayAheadPowerFlowExport"):
            # A09 can also carry total/intraday schedules. Only the A01
            # day-ahead commercial exchange is a Case 5 input.
            if child_text(series, "contract_MarketAgreement.type") != "A01":
                continue
            if feature == "DayAheadPowerFlowImport":
                expected_in, expected_out = domain, counterparty_domain
            else:
                expected_in, expected_out = counterparty_domain, domain
            if (child_text(series, "in_Domain.mRID") != expected_in or
                    child_text(series, "out_Domain.mRID") != expected_out):
                continue
        curve = child_text(series, "curveType")
        psr_node = descendant(series, "MktPSRType")
        psr = child_text(psr_node, "psrType")
        expected = feature.removeprefix("Actual").removeprefix("DayAhead")
        if feature in ("ActualSolar", "ActualWindOnshore", "ActualWindOffshore",
                       "DayAheadSolar", "DayAheadWindOnshore", "DayAheadWindOffshore"):
            if PSR_FEATURE.get(psr) != expected:
                continue
        for period in (part for part in series
                       if local_name(part.tag) == "Period"):
            interval = child(period, "timeInterval")
            first = child_text(interval, "start")
            last = child_text(interval, "end")
            minutes = resolution_minutes(child_text(period, "resolution"))
            if not first or not last or minutes <= 0:
                continue
            period_start, period_end = utc_datetime(first), utc_datetime(last)
            value_tag = "price.amount" if feature == "DayAheadPrice" else "quantity"
            # Prices are stored in EUR. Bulgaria published BGN until
            # 2022-01-21; the lev is pegged at 1.95583 BGN per EUR.
            factor = 1.
            if feature == "DayAheadPrice":
                currency = child_text(series, "currency_Unit.name")
                if currency and currency != "EUR":
                    if currency not in EUR_PEG:
                        raise ValueError(f"Unsupported price currency {currency}")
                    factor = 1. / EUR_PEG[currency]
            period_points = []
            for row in (part for part in period
                        if local_name(part.tag) == "Point"):
                position = child_text(row, "position")
                value = child_text(row, value_tag)
                if not position or not value:
                    continue
                try:
                    ordinal = int(position)
                    timestamp = period_start + dt.timedelta(
                        minutes=minutes * (ordinal - 1))
                    quantity = float(value) * factor
                except ValueError:
                    continue
                if period_start <= timestamp < period_end and np.isfinite(quantity):
                    period_points.append((ordinal, timestamp, quantity))
            inferred_minutes = effective_minutes(
                minutes, period_start, period_end,
                [item[0] for item in period_points])
            if curve == "A03":
                period_points = a03_blocks(period_points, inferred_minutes,
                                           period_end)
            periods.append((period_start, period_end, inferred_minutes))
            points.extend((timestamp, quantity, inferred_minutes, period_end)
                          for _, timestamp, quantity in period_points)
    return points, periods


def read_feature_sources(index, region, feature, neighbor, start, end):
    family = FEATURE_TO_FAMILY[feature]
    areas = load_config()[1]
    domain = areas[region]
    counterparty_domain = areas[neighbor] if neighbor else None
    all_points, all_periods = [], []
    for row in source_entries(index, region, family, neighbor, start, end):
        path = RAW / row["path"]
        with gzip.open(path, "rb") as stream:
            body = stream.read()
        if sha256(body) != row["sha256_uncompressed"]:
            raise ValueError(f"Raw archive checksum mismatch: {path}")
        points, periods = parse_source(body, feature, domain,
                                       counterparty_domain)
        all_points.extend(points)
        all_periods.extend(periods)
    return all_points, all_periods


def align_quarter_hours(points, periods, start, end, fill="copy"):
    if fill not in FILL_MODES:
        raise ValueError(f"fill must be one of {FILL_MODES}")
    if (end - start).total_seconds() % 900:
        raise ValueError("Output interval must end on a quarter hour")
    count = int((end - start).total_seconds() // 900)
    explicit = {}
    for stamp, value, minutes, period_end in points:
        offset = (stamp - start).total_seconds() / 900
        if offset != int(offset):
            raise ValueError(f"Source point off the quarter-hour grid: {stamp}")
        position = int(offset)
        if 0 <= position < count:
            prior = explicit.get(position)
            if prior is None or minutes <= prior[1]:
                explicit[position] = (value, minutes, period_end)
    finer = np.zeros(count, dtype=bool)
    for first, last, minutes in periods:
        if minutes >= 60:
            continue
        left = max(0, int((first - start).total_seconds() // 900))
        right = min(count, int((last - start).total_seconds() // 900))
        if right > left:
            finer[left:right] = True
    values = np.full(count, np.nan)
    observed = np.zeros(count, dtype=bool)
    expanded = np.zeros(count, dtype=bool)
    for position, (value, minutes, _) in explicit.items():
        if minutes == 60 and finer[position]:
            continue
        values[position] = value
        observed[position] = True
    anchors = values.copy()
    for position, (value, minutes, period_end) in explicit.items():
        if fill == "none" or minutes != 60 or not observed[position]:
            continue
        if fill == "interp":
            following = position + 4
            if following >= count or not observed[following]:
                continue
            slope = (anchors[following] - value) / 4.
        for step in (1, 2, 3):
            target = position + step
            target_time = start + target * INTERVAL
            if (target >= count or target_time >= period_end or
                    finer[target] or observed[target]):
                continue
            values[target] = value if fill == "copy" else value + slope * step
            expanded[target] = True
    if np.any(observed & expanded) or np.any(expanded & ~np.isfinite(values)):
        raise AssertionError("Invalid hourly expansion flags")
    return values, observed, expanded


def make_feature(index, region, feature, neighbors, start, end, fill="copy"):
    count = int((end - start).total_seconds() // 900)
    family = FEATURE_TO_FAMILY[feature]
    if family not in ("flow_import", "flow_export",
                      "schedule_import", "schedule_export"):
        points, periods = read_feature_sources(
            index, region, feature, "", start, end)
        return align_quarter_hours(points, periods, start, end, fill)
    total = np.zeros(count)
    has_value = np.zeros(count, dtype=bool)
    source_observed = np.zeros(count, dtype=bool)
    hourly_expanded = np.zeros(count, dtype=bool)
    for neighbor in neighbors[region]:
        points, periods = read_feature_sources(
            index, region, feature, neighbor, start, end)
        values, observed, expanded = align_quarter_hours(
            points, periods, start, end, fill)
        valid = np.isfinite(values)
        total[valid] += values[valid]
        has_value |= valid
        source_observed |= observed
        hourly_expanded |= expanded
    total[~has_value] = np.nan
    return total, source_observed, hourly_expanded


def merge_feature(region, feature, start, end, values, observed, expanded,
                  fill="copy"):
    count = len(values)
    stamps = pd.date_range(start, periods=count, freq="15min", tz="UTC")
    new = pd.DataFrame({
        "timestamp_utc": stamps.strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "value": values,
        "source_observed": observed.astype("uint8"),
        "hourly_expanded": expanded.astype("uint8"),
    })
    output = OUTPUT if fill == "copy" else OUTPUTS[fill]
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{region}_{feature}.csv"
    if path.exists():
        old = pd.read_csv(path)
        if list(old.columns) != list(new.columns):
            raise ValueError(f"Unexpected processed schema: {path}")
        old = old.loc[~old["timestamp_utc"].isin(new["timestamp_utc"])]
        new = pd.concat((old, new), ignore_index=True).sort_values(
            "timestamp_utc", kind="stable")
    if new["timestamp_utc"].duplicated().any():
        raise AssertionError(f"Duplicate processed timestamp: {path}")
    times = pd.DatetimeIndex(pd.to_datetime(new["timestamp_utc"], utc=True))
    full_grid = pd.date_range(times[0], times[-1], freq="15min", tz="UTC")
    if not times.equals(full_grid):
        # Separate notebook runs may request disjoint date ranges. Keep one
        # continuous time axis and leave unrequested dates explicitly blank.
        new.index = times
        new = new.reindex(full_grid)
        new["timestamp_utc"] = full_grid.strftime("%Y-%m-%d %H:%M:%S+00:00")
        for flag in ("source_observed", "hourly_expanded"):
            new[flag] = new[flag].fillna(0).astype("uint8")
        new = new.reset_index(drop=True)
    temporary = path.with_suffix(".csv.tmp")
    new.to_csv(temporary, index=False, float_format="%.15g", na_rep="")
    temporary.replace(path)
    return {"region": region, "feature": feature, "rows": len(new),
            "reported_on_requested_interval": int(np.isfinite(values).sum()),
            "hourly_expanded_on_requested_interval": int(expanded.sum()),
            "output": str(path)}


def process_selected(regions, features, start, end, fill="copy"):
    index = load_index()
    _, _, neighbors = load_config()
    output = []
    for region in regions:
        for feature in features:
            values, observed, expanded = make_feature(
                index, region, feature, neighbors, start, end, fill)
            summary = merge_feature(region, feature, start, end,
                                    values, observed, expanded, fill)
            output.append(summary)
            print(f"Processed [{fill}] {region} {feature}: "
                  f"{summary['reported_on_requested_interval']} valid slots",
                  flush=True)
    manifest = PROCESSED / ("last_run.json" if fill == "copy"
                            else f"last_run_{fill}.json")
    manifest.write_text(json.dumps({"start": iso(start), "end": iso(end),
                                    "fill": fill,
                                    "features": list(features),
                                    "regions": list(regions),
                                    "series": output}, indent=2) + "\n",
                        encoding="utf-8")
    return output
