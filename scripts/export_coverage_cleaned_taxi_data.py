#!/usr/bin/env python
"""Export a coverage-filtered taxi corpus from the raw trajectory CSV."""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.data_loader import (  # noqa: E402
    DEFAULT_PROCESSED_TAXI_PATH,
    DEFAULT_SERVER_PATH,
    DEFAULT_TAXI_PATH,
    PROCESSED_REQUIRED_COLUMNS,
    _assign_episode_ids,
    load_data,
)
from core.geo import haversine_distance  # noqa: E402


def _default_out_path(coverage_km):
    base, ext = os.path.splitext(DEFAULT_PROCESSED_TAXI_PATH)
    cov_label = str(float(coverage_km)).rstrip("0").rstrip(".").replace(".", "p")
    return f"{base}_cov{cov_label}{ext}"


def _nearest_server_distances_km(df, servers_df, chunk_rows=5000):
    """Return per-row distance to the nearest configured edge server."""
    server_lats = servers_df["latitude"].to_numpy(dtype=np.float64, copy=False)
    server_lons = servers_df["longitude"].to_numpy(dtype=np.float64, copy=False)
    lat = df["latitude"].to_numpy(dtype=np.float64, copy=False)
    lon = df["longitude"].to_numpy(dtype=np.float64, copy=False)
    out = np.empty(len(df), dtype=np.float64)

    for start in range(0, len(df), int(chunk_rows)):
        end = min(start + int(chunk_rows), len(df))
        dists = haversine_distance(
            lat[start:end, None],
            lon[start:end, None],
            server_lats[None, :],
            server_lons[None, :],
        )
        out[start:end] = np.min(dists, axis=1)
    return out


def _quantiles(values, qs):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {str(q): None for q in qs}
    return {str(q): float(np.percentile(arr, q)) for q in qs}


def export_coverage_cleaned_corpus(
    out_path=None,
    raw_path=DEFAULT_TAXI_PATH,
    server_path=DEFAULT_SERVER_PATH,
    coverage_km=50.0,
    min_episode_coverage_ratio=0.8,
    active_users_limit=100,
    min_vehicle_points=100,
    v_max_kmh=200.0,
    gap_dt_hours=2.0,
    max_jump_clean_rounds=50,
    chunk_rows=5000,
):
    """Build a new cleaned corpus with edge-server coverage constraints."""
    if out_path is None:
        out_path = _default_out_path(coverage_km)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    servers_df = pd.read_csv(server_path)

    df = load_data(
        file_path=raw_path,
        processed_csv=False,
        sample_fraction=1.0,
        chunk_size=None,
        start_index=None,
        end_index=None,
        active_users_limit=None,
        min_vehicle_points=min_vehicle_points,
        v_max_kmh=v_max_kmh,
        gap_dt_hours=gap_dt_hours,
        max_jump_clean_rounds=max_jump_clean_rounds,
    )
    rows_after_base_clean = len(df)
    taxis_after_base_clean = int(df["taxi_id"].nunique())

    nearest_km = _nearest_server_distances_km(df, servers_df, chunk_rows=chunk_rows)
    df = df.copy()
    df["_nearest_server_km"] = nearest_km
    df["_in_coverage"] = df["_nearest_server_km"] <= float(coverage_km)

    episode_stats = (
        df.groupby(["taxi_id", "episode_id"], sort=False)
        .agg(rows=("taxi_id", "size"), in_coverage_ratio=("_in_coverage", "mean"))
        .reset_index()
    )
    keep_episode_keys = episode_stats.loc[
        episode_stats["in_coverage_ratio"] >= float(min_episode_coverage_ratio),
        ["taxi_id", "episode_id"],
    ]

    before_episode_rows = len(df)
    df = df.merge(
        keep_episode_keys.assign(_keep_episode=True),
        on=["taxi_id", "episode_id"],
        how="left",
    )
    keep_mask = df["_keep_episode"].eq(True)
    df = df[keep_mask].copy()
    rows_after_episode_filter = len(df)

    before_point_rows = len(df)
    df = df[df["_in_coverage"]].copy()
    rows_after_point_filter = len(df)

    df = df[["taxi_id", "date_time", "latitude", "longitude"]].copy()
    df = _assign_episode_ids(df, gap_dt_hours=gap_dt_hours)

    before_second_min_rows = len(df)
    counts = df.groupby("taxi_id").size()
    keep_ids = counts[counts >= int(min_vehicle_points)].index
    df = df[df["taxi_id"].isin(keep_ids)].reset_index(drop=True)
    rows_after_second_min = len(df)

    before_topn_rows = len(df)
    if active_users_limit is not None and int(active_users_limit) > 0:
        counts = df.groupby("taxi_id").size().sort_values(ascending=False)
        top_ids = counts.head(int(active_users_limit)).index
        df = df[df["taxi_id"].isin(top_ids)].reset_index(drop=True)

    df = df.sort_values(["taxi_id", "date_time"]).reset_index(drop=True)
    df = df[PROCESSED_REQUIRED_COLUMNS]
    df.to_csv(out_path, index=False)

    final_nearest_km = _nearest_server_distances_km(df, servers_df, chunk_rows=chunk_rows)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "raw_path": os.path.abspath(raw_path),
        "server_path": os.path.abspath(server_path),
        "out_path": os.path.abspath(out_path),
        "coverage_km": float(coverage_km),
        "min_episode_coverage_ratio": float(min_episode_coverage_ratio),
        "active_users_limit": int(active_users_limit) if active_users_limit is not None else None,
        "min_vehicle_points": int(min_vehicle_points),
        "v_max_kmh": float(v_max_kmh),
        "gap_dt_hours": float(gap_dt_hours),
        "rows_after_base_clean": int(rows_after_base_clean),
        "taxis_after_base_clean": taxis_after_base_clean,
        "episode_rows_before_filter": int(before_episode_rows),
        "rows_after_episode_filter": int(rows_after_episode_filter),
        "point_rows_before_filter": int(before_point_rows),
        "rows_after_point_filter": int(rows_after_point_filter),
        "rows_before_second_min_points": int(before_second_min_rows),
        "rows_after_second_min_points": int(rows_after_second_min),
        "rows_before_top_active": int(before_topn_rows),
        "final_rows": int(len(df)),
        "final_taxis": int(df["taxi_id"].nunique()),
        "final_episode_count": int(df.groupby(["taxi_id", "episode_id"]).ngroups) if len(df) else 0,
        "final_time_min": str(df["date_time"].min()) if len(df) else None,
        "final_time_max": str(df["date_time"].max()) if len(df) else None,
        "final_nearest_server_km_quantiles": _quantiles(
            final_nearest_km, [0, 10, 25, 50, 75, 90, 95, 99, 100]
        ),
        "rows_per_taxi_top20": {
            str(k): int(v)
            for k, v in df.groupby("taxi_id").size().sort_values(ascending=False).head(20).items()
        },
    }

    summary_path = f"{os.path.splitext(out_path)[0]}.summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return out_path, summary_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="Output CSV path.")
    parser.add_argument("--coverage-km", type=float, default=50.0)
    parser.add_argument("--min-episode-coverage-ratio", type=float, default=0.8)
    parser.add_argument("--active-users-limit", type=int, default=100)
    parser.add_argument("--min-vehicle-points", type=int, default=100)
    parser.add_argument("--v-max-kmh", type=float, default=200.0)
    parser.add_argument("--gap-dt-hours", type=float, default=2.0)
    parser.add_argument("--chunk-rows", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_coverage_cleaned_corpus(
        out_path=args.out,
        coverage_km=args.coverage_km,
        min_episode_coverage_ratio=args.min_episode_coverage_ratio,
        active_users_limit=args.active_users_limit,
        min_vehicle_points=args.min_vehicle_points,
        v_max_kmh=args.v_max_kmh,
        gap_dt_hours=args.gap_dt_hours,
        chunk_rows=args.chunk_rows,
    )
