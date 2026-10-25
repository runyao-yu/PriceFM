"""One-call incremental raw download, process, merge and validation."""

from __future__ import annotations

import argparse

from .check import check_processed, check_raw
from .core import ALL_FEATURES, inclusive_dates, validate_selection
from .download import download_missing
from .process import process_selected
from .availability import apply as apply_availability


def sync_dataset(regions, start_date, end_date, features, fill="copy",
                 download=True):
    regions, features = validate_selection(regions, features)
    start, end = inclusive_dates(start_date, end_date)
    downloaded = (download_missing(regions, features, start, end)
                  if download else {"skipped": True})
    try:
        raw_check = check_raw(regions, features, start, end)
    except RuntimeError as exc:
        # Failed requests are listed in raw/failed_requests.json; process
        # what is archived and leave uncovered slots blank.
        raw_check = {"warning": str(exc)}
        print(f"WARNING {exc}", flush=True)
    processed = process_selected(regions, features, start, end, fill)
    # Blank series that fail the availability rule (see availability.py).
    blanked = apply_availability(regions, features)
    processed_check = (check_processed(regions, features, start, end)
                       if fill == "copy" else [])
    return {"download": downloaded, "raw_check": raw_check,
            "processed_series": len(processed), "blanked_unavailable": len(blanked),
            "checked_series": len(processed_check)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", nargs="+", required=True)
    parser.add_argument("--start", required=True, help="Inclusive UTC date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Inclusive UTC date YYYY-MM-DD")
    parser.add_argument("--features", nargs="+", default=ALL_FEATURES)
    parser.add_argument("--fill", choices=("copy", "interp", "none"),
                        default="copy")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    print(sync_dataset(args.regions, args.start, args.end, args.features,
                       args.fill, not args.no_download))


if __name__ == "__main__":
    main()
