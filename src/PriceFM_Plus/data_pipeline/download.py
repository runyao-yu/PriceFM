"""Incremental, lossless ENTSO-E XML downloader with interval coverage index."""

from __future__ import annotations

import datetime as dt
import gzip
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

from .core import (RAW, RequestSpec, atomic_json, iso, load_config,
                  requested_keys, sha256, uncovered, utc_datetime,
                  year_windows)


API_URL = "https://web-api.tp.entsoe.eu/api"
INDEX = RAW / "index.json"


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def token_from_environment():
    token = os.environ.get("ENTSOE_API_TOKEN", "").strip()
    if not token and os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                token = str(winreg.QueryValueEx(key, "ENTSOE_API_TOKEN")[0]).strip()
        except (FileNotFoundError, OSError):
            pass
    if not token:
        raise RuntimeError("ENTSOE_API_TOKEN is not configured in the user environment")
    return token


def parameters(spec, areas):
    domain = areas[spec.region]
    base = {"periodStart": spec.start.strftime("%Y%m%d%H%M"),
            "periodEnd": spec.end.strftime("%Y%m%d%H%M")}
    if spec.family == "price":
        return {"documentType": "A44", "in_Domain": domain,
                "out_Domain": domain, **base}
    if spec.family in ("day_ahead_load", "actual_load"):
        return {"documentType": "A65",
                "processType": "A01" if spec.family == "day_ahead_load" else "A16",
                "outBiddingZone_Domain": domain, **base}
    if spec.family == "day_ahead_renewables":
        return {"documentType": "A69", "processType": "A01",
                "in_Domain": domain, **base}
    if spec.family == "actual_generation":
        return {"documentType": "A75", "processType": "A16",
                "in_Domain": domain, **base}
    neighbor = areas[spec.neighbor]
    if spec.family == "flow_import":
        return {"documentType": "A11", "in_Domain": domain,
                "out_Domain": neighbor, **base}
    if spec.family == "flow_export":
        return {"documentType": "A11", "in_Domain": neighbor,
                "out_Domain": domain, **base}
    if spec.family == "schedule_import":
        return {"documentType": "A09", "contract_MarketAgreement.Type": "A01",
                "in_Domain": domain, "out_Domain": neighbor, **base}
    if spec.family == "schedule_export":
        return {"documentType": "A09", "contract_MarketAgreement.Type": "A01",
                "in_Domain": neighbor, "out_Domain": domain, **base}
    raise ValueError(spec.family)


def load_index():
    return json.loads(INDEX.read_text(encoding="utf-8")) if INDEX.exists() else []


def cached_row_usable(row, now):
    if row["status"] == "data":
        return True
    if row["status"] != "no_data":
        return False
    # A recent no-data acknowledgement can precede publication. Retry it
    # after one day while its requested end is less than 30 days old.
    recent = utc_datetime(row["end"]) > now - dt.timedelta(days=30)
    stale = utc_datetime(row["downloaded_at_utc"]) < now - dt.timedelta(days=1)
    return not (recent and stale)


def plan_missing(regions, features, start, end, index=None, now=None):
    index = load_index() if index is None else index
    now = now or dt.datetime.now(dt.timezone.utc)
    active = {}
    for row in index:
        if cached_row_usable(row, now) and (RAW / row["path"]).exists():
            key = row["region"], row["family"], row["neighbor"]
            active.setdefault(key, []).append((utc_datetime(row["start"]),
                                                 utc_datetime(row["end"])))
    planned = []
    for region, family, neighbor in requested_keys(regions, features):
        cached = active.get((region, family, neighbor), [])
        for first, last in year_windows(start, end):
            for gap_start, gap_end in uncovered(first, last, cached):
                if gap_start < gap_end:
                    planned.append(RequestSpec(region, family, gap_start,
                                               gap_end, neighbor))
    return planned


class RateGate:
    def __init__(self, requests_per_second):
        self.delay = 1. / requests_per_second
        self.next_at = 0.
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            hold = max(0., self.next_at - now)
            self.next_at = max(now, self.next_at) + self.delay
        if hold:
            time.sleep(hold)


def acknowledgement(body):
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return None, "invalid_xml"
    if local_name(root.tag) != "Acknowledgement_MarketDocument":
        return "data", ""
    reason = next((node for node in root.iter()
                   if local_name(node.tag) == "Reason"), None)
    code = ""
    if reason is not None:
        code = next((node.text or "" for node in reason
                     if local_name(node.tag) == "code"), "")
    return ("no_data" if code == "999" else None), code


def fetch(spec, token, areas, gate):
    query = parameters(spec, areas)
    query["securityToken"] = token
    url = API_URL + "?" + urllib.parse.urlencode(query)
    last_error = ""
    for attempt in range(5):
        gate.wait()
        try:
            request = urllib.request.Request(
                url, headers={"Accept": "application/xml",
                              "User-Agent": "PriceFM-Plus-incremental-archive/1.0"})
            with urllib.request.urlopen(request, timeout=180) as response:
                body = response.read()
            status, reason = acknowledgement(body)
            if status is None:
                raise RuntimeError(f"ENTSO-E response error code {reason}")
            return spec, body, status
        except urllib.error.HTTPError as exc:
            body = exc.read()
            status, reason = acknowledgement(body)
            if status == "no_data":
                return spec, body, status
            last_error = f"HTTP {exc.code}, reason {reason}"
            if exc.code not in (429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = type(exc).__name__
        time.sleep(min(30 * (attempt + 1), 120))
    raise RuntimeError(f"Request failed for {spec.region}/{spec.family}/"
                       f"{spec.neighbor or 'area'} {iso(spec.start)}–"
                       f"{iso(spec.end)}: {last_error}")


def download_missing(regions, features, start, end, *, workers=4,
                     requests_per_second=5.):
    index = load_index()
    planned = plan_missing(regions, features, start, end, index)
    print(f"Raw archive: {len(planned)} missing request intervals")
    if not planned:
        return {"planned": 0, "downloaded": 0, "no_data": 0}
    token = token_from_environment()
    _, areas, _ = load_config()
    gate = RateGate(requests_per_second)
    statuses = {"data": 0, "no_data": 0}
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, spec, token, areas, gate): spec
                   for spec in planned}
        for completed, future in enumerate(as_completed(futures), 1):
            try:
                spec, body, status = future.result()
            except RuntimeError as exc:
                # One rejected request must not abort a long archive run.
                # It stays uncovered in the index and is retried next run.
                failures.append({"region": futures[future].region,
                                 "family": futures[future].family,
                                 "neighbor": futures[future].neighbor,
                                 "start": iso(futures[future].start),
                                 "end": iso(futures[future].end),
                                 "error": str(exc)})
                print(f"FAILED {exc}", flush=True)
                continue
            relative_path = spec.relative_path
            path = RAW / relative_path
            if path.exists():
                stamp = dt.datetime.now(dt.timezone.utc).strftime(
                    "%Y%m%dT%H%M%S%fZ")
                filename = path.name.removesuffix(".xml.gz")
                relative_path = relative_path.with_name(
                    f"{filename}__{stamp}.xml.gz")
                path = RAW / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            with gzip.open(temporary, "wb") as stream:
                stream.write(body)
            temporary.replace(path)
            index.append({"region": spec.region, "family": spec.family,
                          "neighbor": spec.neighbor, "start": iso(spec.start),
                          "end": iso(spec.end), "status": status,
                          "path": str(relative_path).replace("\\", "/"),
                          "sha256_uncompressed": sha256(body),
                          "downloaded_at_utc": iso(dt.datetime.now(dt.timezone.utc))})
            atomic_json(INDEX, index)
            statuses[status] += 1
            if completed == 1 or completed % 20 == 0 or completed == len(planned):
                print(f"Raw archive: {completed}/{len(planned)} requests saved",
                      flush=True)
    atomic_json(RAW / "failed_requests.json", failures)
    return {"planned": len(planned), "downloaded": statuses["data"],
            "no_data": statuses["no_data"], "failed": len(failures)}
