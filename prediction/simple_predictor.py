"""Simple velocity-based trajectory predictor extracted from Context_Integratoin.py."""

import numpy as np
import pandas as pd
from tqdm import tqdm


def touch_taxi_last(taxi_last, taxi_id, row, current_lon, current_lat, ts):
    """Record last observed position/time/episode for local-kinematic lookahead (per simulation loop)."""
    if "episode_id" in row.index and pd.notna(row.get("episode_id")):
        pep = int(row["episode_id"])
    else:
        pep = None
    taxi_last[taxi_id] = (
        float(current_lon),
        float(current_lat),
        pd.Timestamp(ts),
        pep,
    )


def build_predict_future_time_kwargs(taxi_last, taxi_id, row, current_lon, current_lat, current_time):
    """
    Build optional time kwargs for predict_future (unseen-taxi / local kinematic branch).

    taxi_last maps taxi_id -> (prev_lon, prev_lat, prev_time, prev_episode_id).
    On new episode_id vs previous, or missing history, returns prev_lon=None so predictor stays in-place.
    """
    ts = pd.Timestamp(current_time)
    out = {"current_time": ts}
    if "episode_id" in row.index and pd.notna(row.get("episode_id")):
        cur_ep = int(row["episode_id"])
    else:
        cur_ep = None

    if taxi_id not in taxi_last:
        out["prev_lon"] = None
        return out

    plon, plat, ptime, pep = taxi_last[taxi_id]
    if cur_ep is not None and pep is not None and cur_ep != pep:
        out["prev_lon"] = None
        return out

    pt = pd.Timestamp(ptime)
    dt = (ts - pt).total_seconds()
    if dt <= 0:
        out["prev_lon"] = None
        return out

    out["prev_lon"] = plon
    out["prev_lat"] = plat
    out["prev_time"] = pt
    return out


class SimpleTrajectoryPredictor:
    def __init__(self, forecast_horizon=3):
        self.forecast_horizon = forecast_horizon
        self.velocity_factors = {}

    def fit(self, df):
        print("Fitting trajectory predictor...")
        for taxi_id in tqdm(df["taxi_id"].unique(), desc="Learning velocities"):
            taxi_data = df[df["taxi_id"] == taxi_id].sort_values("date_time")
            if len(taxi_data) < 100:
                continue
            lons = taxi_data["longitude"].values
            lats = taxi_data["latitude"].values
            dx = np.diff(lons)
            dy = np.diff(lats)
            if len(dx) > 0:
                self.velocity_factors[taxi_id] = (np.mean(dx), np.mean(dy))
        return self

    def predict_future(
        self,
        current_lon,
        current_lat,
        taxi_id,
        steps=None,
        prev_lon=None,
        prev_lat=None,
        prev_time=None,
        current_time=None,
        delta_t_prev_sec=None,
        delta_t_future_sec=None,
        **_kwargs,
    ):
        """
        Returns list of (lon, lat) for each forecast step.

        Known ``taxi_id``: legacy per-step mean (dx, dy) in degree space (unchanged).

        Unknown ``taxi_id``: time-normalized local kinematics when prev_* and dt_prev valid:
        v_lon = (lon - prev_lon) / dt_prev, v_lat = (lat - prev_lat) / dt_prev (degrees/sec);
        each step: lon += v_lon * dt_future, lat += v_lat * dt_future.
        If ``delta_t_future_sec`` is None, uses dt_prev for each step.
        """
        if steps is None:
            steps = self.forecast_horizon

        if taxi_id in self.velocity_factors:
            dx, dy = self.velocity_factors[taxi_id]
            future = []
            lon, lat = current_lon, current_lat
            for _ in range(steps):
                lon += dx
                lat += dy
                future.append((lon, lat))
            return future

        # --- Unseen taxi: local kinematic (time-normalized) ---
        dt_prev = None
        if delta_t_prev_sec is not None and float(delta_t_prev_sec) > 0:
            dt_prev = float(delta_t_prev_sec)
        elif (
            prev_lon is not None
            and prev_lat is not None
            and prev_time is not None
            and current_time is not None
        ):
            dt_prev = (
                pd.Timestamp(current_time) - pd.Timestamp(prev_time)
            ).total_seconds()

        if (
            dt_prev is None
            or dt_prev <= 0
            or prev_lon is None
            or prev_lat is None
        ):
            return [(current_lon, current_lat)] * steps

        v_lon = (float(current_lon) - float(prev_lon)) / dt_prev
        v_lat = (float(current_lat) - float(prev_lat)) / dt_prev

        def _dt_for_step(k):
            if delta_t_future_sec is None:
                return dt_prev
            if isinstance(delta_t_future_sec, (int, float, np.integer, np.floating)):
                return float(delta_t_future_sec)
            seq = list(delta_t_future_sec)
            if k < len(seq):
                return float(seq[k])
            return dt_prev

        lon, lat = float(current_lon), float(current_lat)
        out = []
        for k in range(steps):
            dtf = _dt_for_step(k)
            if dtf <= 0:
                out.append((lon, lat))
                continue
            lon += v_lon * dtf
            lat += v_lat * dtf
            out.append((lon, lat))
        return out
