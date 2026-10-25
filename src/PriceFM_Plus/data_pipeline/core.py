"""Shared schema and UTC interval helpers for the ENTSO-E data archive."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path.cwd() / "Data"
RAW = ROOT / "Raw"
PROCESSED = ROOT / "Processed"
CONFIG = ROOT / "Config"
UTC = dt.timezone.utc
INTERVAL = dt.timedelta(minutes=15)
FEATURE_TO_FAMILY = {
    "DayAheadPrice": "price",
    "DayAheadLoad": "day_ahead_load",
    "DayAheadSolar": "day_ahead_renewables",
    "DayAheadWindOnshore": "day_ahead_renewables",
    "DayAheadWindOffshore": "day_ahead_renewables",
    "ActualLoad": "actual_load",
    "ActualSolar": "actual_generation",
    "ActualWindOnshore": "actual_generation",
    "ActualWindOffshore": "actual_generation",
    "ActualPowerFlowImport": "flow_import",
    "ActualPowerFlowExport": "flow_export",
    "DayAheadPowerFlowImport": "schedule_import",
    "DayAheadPowerFlowExport": "schedule_export",
}
ALL_FEATURES = tuple(FEATURE_TO_FAMILY)
FLOW_FAMILIES = frozenset(("flow_import", "flow_export",
                           "schedule_import", "schedule_export"))
PSR_FEATURE = {"B16": "Solar", "B18": "WindOffshore",
               "B19": "WindOnshore"}
PROCESSOR_VERSION = 2


def load_config():
    return tuple(json.loads((CONFIG / "active_regions.json").read_text())), \
        json.loads((CONFIG / "areas.json").read_text()), \
        json.loads((CONFIG / "neighbors.json").read_text())


def parse_date(value: str) -> dt.datetime:
    return dt.datetime.combine(dt.date.fromisoformat(value), dt.time.min,
                               tzinfo=UTC)


def inclusive_dates(start_date: str, end_date: str):
    first = parse_date(start_date)
    last = parse_date(end_date) + dt.timedelta(days=1)
    if last <= first:
        raise ValueError("END_DATE must be on or after START_DATE")
    return first, last


def utc_datetime(text: str) -> dt.datetime:
    value = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def iso(value: dt.datetime) -> str:
    return value.astimezone(UTC).isoformat()


def year_windows(start: dt.datetime, end: dt.datetime):
    cursor = start
    while cursor < end:
        boundary = dt.datetime(cursor.year + 1, 1, 1, tzinfo=UTC)
        following = min(end, boundary)
        yield cursor, following
        cursor = following


def uncovered(start: dt.datetime, end: dt.datetime, intervals):
    """Subtract cached half-open intervals from one requested interval."""
    cursor = start
    for left, right in sorted(intervals):
        if right <= cursor:
            continue
        if left > cursor:
            yield cursor, min(left, end)
        cursor = max(cursor, right)
        if cursor >= end:
            return
    if cursor < end:
        yield cursor, end


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class RequestSpec:
    region: str
    family: str
    start: dt.datetime
    end: dt.datetime
    neighbor: str = ""

    @property
    def key(self):
        return (self.region, self.family, self.neighbor)

    @property
    def relative_path(self):
        suffix = self.neighbor or "area"
        filename = f"{self.start:%Y%m%dT%H%M}_{self.end:%Y%m%dT%H%M}.xml.gz"
        return Path(self.region) / self.family / suffix / filename


def requested_keys(regions, features):
    _, _, neighbors = load_config()
    families = set(FEATURE_TO_FAMILY[name] for name in features)
    for region in regions:
        for family in sorted(families):
            if family in FLOW_FAMILIES:
                for neighbor in neighbors[region]:
                    yield region, family, neighbor
            else:
                yield region, family, ""


def validate_selection(regions, features):
    active, _, _ = load_config()
    regions = tuple(regions)
    features = tuple(features)
    if not regions or len(set(regions)) != len(regions) or not set(regions) <= set(active):
        raise ValueError("REGIONS must be distinct configured region codes")
    if not features or len(set(features)) != len(features) or \
            not set(features) <= set(ALL_FEATURES):
        raise ValueError(f"FEATURES must be distinct names from {ALL_FEATURES}")
    return regions, features
