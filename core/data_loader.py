import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_TAXI_PATH = os.path.join(DATA_DIR, "taxi_with_health_info.csv")
DEFAULT_SERVER_PATH = os.path.join(DATA_DIR, "edge_server_locations.csv")


CORE_COLUMNS = ["taxi_id", "date_time", "latitude", "longitude"]

# Earth radius (km) for Haversine
_EARTH_RADIUS_KM = 6371.0088


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; arrays in degrees."""
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return _EARTH_RADIUS_KM * c


def _remove_velocity_jump_outliers(df, v_max_kmh=200.0, max_rounds=50):
    """
    Vectorized per-round removal: drop row j when edge (j-1, j) has v_kmh > v_max and dt > 0.
    Returns (cleaned_df, total_rows_removed).
    """
    total_removed = 0
    work = df
    for _ in range(max_rounds):
        work = work.sort_values(["taxi_id", "date_time"]).reset_index(drop=True)
        if len(work) == 0:
            break
        prev_lat = work.groupby("taxi_id")["latitude"].shift(1)
        prev_lon = work.groupby("taxi_id")["longitude"].shift(1)
        prev_t = work.groupby("taxi_id")["date_time"].shift(1)
        dt = (work["date_time"] - prev_t).dt.total_seconds().to_numpy(dtype=np.float64)
        dd_km = _haversine_km(
            prev_lat.to_numpy(),
            prev_lon.to_numpy(),
            work["latitude"].to_numpy(),
            work["longitude"].to_numpy(),
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            v_kmh = np.where(dt > 0.0, (dd_km / dt) * 3600.0, np.nan)
        bad = (dt > 0.0) & np.isfinite(v_kmh) & (v_kmh > float(v_max_kmh))
        n_bad = int(np.sum(bad))
        if n_bad == 0:
            break
        total_removed += n_bad
        work = work.loc[~bad].reset_index(drop=True)
    return work, total_removed


def _assign_episode_ids(df, gap_dt_hours=2.0):
    """Per taxi_id: increment episode_id on the row after a gap where dt > gap_dt_hours."""
    gap_sec = float(gap_dt_hours) * 3600.0
    df = df.sort_values(["taxi_id", "date_time"]).reset_index(drop=True)
    blocks = []
    for _, g in df.groupby("taxi_id", sort=False):
        g = g.sort_values("date_time").reset_index(drop=True)
        n = len(g)
        ep = np.zeros(n, dtype=np.int64)
        e = 0
        ep[0] = 0
        if n > 1:
            tseries = g["date_time"].values
            for i in range(1, n):
                dt = (pd.Timestamp(tseries[i]) - pd.Timestamp(tseries[i - 1])).total_seconds()
                if dt > gap_sec:
                    e += 1
                ep[i] = e
        gg = g.copy()
        gg["episode_id"] = ep
        blocks.append(gg)
    if not blocks:
        out = df.copy()
        out["episode_id"] = np.zeros(len(out), dtype=np.int64)
        return out
    out = pd.concat(blocks, ignore_index=True)
    return out.sort_values(["taxi_id", "date_time"]).reset_index(drop=True)


def load_data(
    file_path=None,
    sample_fraction=1.0,
    chunk_size=None,
    start_index=None,
    end_index=None,
    active_users_limit=None,
    min_vehicle_points=None,
    v_max_kmh=200.0,
    gap_dt_hours=2.0,
    max_jump_clean_rounds=50,
):
    """
    Load taxi trajectory CSV, keeping only physical-movement columns.

    Phase-1 pipeline (command.md): sort → dropna → 同车时间去重 → 向量化速度毛刺清洗
    (v > v_max_kmh) → 剔除点数过少车辆 → Top‑N 活跃车 → episode_id（合法断层）→
    可选 legacy 行切片 → 可选 chunk_size。

    Parameters
    ----------
    file_path : str, optional
        Path to CSV file. Defaults to DEFAULT_TAXI_PATH.
    sample_fraction : float, optional
        Fraction of taxis to sample (0.0-1.0). Default 1.0 (all taxis). Applied before sorting.
    chunk_size : int, optional
        Truncate to first N rows of the **final** result (dev only).
    start_index, end_index : int, optional
        Legacy: applied **after** active_users_limit / min_vehicle_points, on the final DataFrame.
    active_users_limit : int, optional
        Keep only the top-N taxis by row count after prior filters. ``None`` = no limit.
    min_vehicle_points : int, optional
        Drop taxis with fewer than this many rows after jump cleaning. ``None`` = no filter.
    v_max_kmh : float
        Instantaneous speed threshold (km/h) for jump removal; default 200.
    gap_dt_hours : float
        Time gap (hours) after which ``episode_id`` increments; default 2.
    max_jump_clean_rounds : int
        Max iterations for jump removal until stable.
    """
    if file_path is None:
        file_path = DEFAULT_TAXI_PATH
    print(f"Loading data from {file_path} ...")
    df = pd.read_csv(file_path)
    n_rows_csv = len(df)

    keep = [c for c in CORE_COLUMNS if c in df.columns]
    dropped = [c for c in df.columns if c not in keep]
    if dropped:
        print(f"  Dropped {len(dropped)} non-physical columns: {dropped}")
    df = df[keep]
    n_after_column_select = len(df)

    if sample_fraction < 1.0:
        unique_taxis = df["taxi_id"].unique()
        sampled_taxis = np.random.choice(
            unique_taxis,
            size=int(len(unique_taxis) * sample_fraction),
            replace=False,
        )
        df = df[df["taxi_id"].isin(sampled_taxis)]
        print(f"  Sampled {len(sampled_taxis)} taxis.")

    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.sort_values(["taxi_id", "date_time"]).reset_index(drop=True)
    df = df.dropna(subset=["longitude", "latitude"])
    n_after_dropna = len(df)

    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=["taxi_id", "date_time"], keep="first").reset_index(drop=True)
    n_dedup_removed = n_before_dedup - len(df)
    print(f"  Time dedup: removed {n_dedup_removed:,} duplicate (taxi_id, date_time) rows -> {len(df):,} rows")

    df, n_jump_removed = _remove_velocity_jump_outliers(
        df, v_max_kmh=v_max_kmh, max_rounds=max_jump_clean_rounds
    )
    print(f"  Velocity jump clean (v > {v_max_kmh} km/h): removed {n_jump_removed:,} rows -> {len(df):,} rows")

    n_before_min = len(df)
    if min_vehicle_points is not None and int(min_vehicle_points) > 0:
        mp = int(min_vehicle_points)
        counts = df.groupby("taxi_id").size()
        keep_ids = counts[counts >= mp].index
        n_low_taxis = int((counts < mp).sum())
        df = df[df["taxi_id"].isin(keep_ids)].reset_index(drop=True)
        print(
            f"  min_vehicle_points={mp}: dropped {n_low_taxis} taxis with < {mp} points; "
            f"rows {n_before_min:,} -> {len(df):,}"
        )
    else:
        n_low_taxis = 0
        print("  min_vehicle_points: disabled (None)")

    n_before_topn = len(df)
    if active_users_limit is not None and int(active_users_limit) > 0:
        lim = int(active_users_limit)
        counts = df.groupby("taxi_id").size().sort_values(ascending=False)
        top_ids = counts.head(lim).index
        df = df[df["taxi_id"].isin(top_ids)].reset_index(drop=True)
        print(
            f"  active_users_limit={lim}: kept top {df['taxi_id'].nunique()} taxis by row count; "
            f"rows {n_before_topn:,} -> {len(df):,}"
        )
    else:
        print("  active_users_limit: disabled (None)")

    df = _assign_episode_ids(df, gap_dt_hours=gap_dt_hours)
    print(
        f"  episode_id assigned (gap > {gap_dt_hours} h); "
        f"rows={len(df):,}, global max episode_id={int(df['episode_id'].max()) if len(df) else 0}"
    )

    if start_index is not None and end_index is not None:
        df = df.iloc[start_index:end_index].reset_index(drop=True)
        print(f"  [legacy] Sliced data: [{start_index}:{end_index}], {len(df):,} records")

    if chunk_size is not None and chunk_size > 0:
        df = df.head(chunk_size)
        print(f"  Truncated to {chunk_size} rows.")

    print(
        f"  Summary: csv_rows={n_rows_csv:,}, after_columns={n_after_column_select:,}, "
        f"after_dropna={n_after_dropna:,}, dedup_removed={n_dedup_removed:,}, "
        f"jump_removed={n_jump_removed:,}, final_rows={len(df):,}, final_taxis={df['taxi_id'].nunique():,}"
    )
    print(f"  Final: {len(df):,} records, {df['taxi_id'].nunique():,} unique taxis.")
    return df


if __name__ == "__main__":
    print("=== data_loader.py local validation (command.md phase 1) ===\n")
    out = load_data(
        None,
        sample_fraction=1.0,
        chunk_size=None,
        start_index=None,
        end_index=None,
        active_users_limit=100,
        min_vehicle_points=100,
        v_max_kmh=200.0,
        gap_dt_hours=2.0,
    )
    print("\n--- Returned DataFrame head (5 rows) ---")
    print(out.head(5).to_string())
    print("\nColumns:", list(out.columns))
    print("Done.")
