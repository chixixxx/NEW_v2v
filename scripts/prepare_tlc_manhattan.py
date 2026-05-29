from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from future_v2v.config import load_project_config
from future_v2v.data.tlc_manhattan import prepare_tlc_manhattan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare NYC TLC yellow taxi data for Future V2V Manhattan scenarios.")
    parser.add_argument("--month", required=True, help="Month in YYYY-MM form, for example 2025-10.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--trip-path", default=None)
    parser.add_argument("--zone-lookup-path", default=None)
    parser.add_argument("--processed-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    env = config.environment
    trip_path = args.trip_path or env.tlc_trip_path
    zone_lookup_path = args.zone_lookup_path or env.taxi_zone_lookup_path
    processed_dir = args.processed_dir or env.processed_dir
    paths = prepare_tlc_manhattan(
        trip_path=trip_path,
        zone_lookup_path=zone_lookup_path,
        processed_dir=processed_dir,
        month=args.month,
        manhattan_only=env.manhattan_only,
        tick_minutes=config.experiment.minutes_per_tick,
        time_bucket_minutes=env.time_bucket_minutes,
    )
    print("Prepared TLC Manhattan data:")
    print(f"  trip rows: {paths.trip_rows}")
    print(f"  zone time matrix: {paths.zone_time_matrix}")
    print(f"  zone lookup: {paths.zone_lookup}")
    print(f"  health report: {paths.health_report}")


if __name__ == "__main__":
    main()
