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
    out["delta_t_prev_sec"] = dt
    out["delta_t_future_sec"] = dt
    out["forecast_step_dt_sec"] = dt
    return out


class SimpleTrajectoryPredictor:
    def __init__(self, forecast_horizon=3, local_velocity_alpha=0.85, max_local_dt_sec=300.0):
        self.forecast_horizon = forecast_horizon
        self.local_velocity_alpha = float(local_velocity_alpha)
        self.max_local_dt_sec = float(max_local_dt_sec)
        self.velocity_factors = {}
        self.prediction_stats = {
            "local_velocity_used": 0,
            "fusion_used": 0,
            "historical_fallback_used": 0,
            "stationary_fallback_used": 0,
            "dt_rejected": 0,
        }

    def reset_stats(self):
        for key in self.prediction_stats:
            self.prediction_stats[key] = 0

    def get_stats(self):
        return dict(self.prediction_stats)

    def _record_stat(self, key):
        if key in self.prediction_stats:
            self.prediction_stats[key] += 1

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

        Local velocity is preferred for both known and unseen taxis when a recent
        same-episode GPS interval is available and not too stale. For known
        taxis, local motion is fused with historical mean per-step displacement.

        Fallback order:
        1. valid local velocity + historical mean -> EMA fusion
        2. valid local velocity only -> local kinematics
        3. historical mean only -> historical per-step displacement
        4. neither -> stationary prediction
        """
        if steps is None:
            steps = self.forecast_horizon

        historical_step = self.velocity_factors.get(taxi_id)

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

        def _dt_for_step(k):
            if delta_t_future_sec is None:
                return dt_prev if dt_prev is not None and dt_prev > 0 else 1.0
            if isinstance(delta_t_future_sec, (int, float, np.integer, np.floating)):
                return float(delta_t_future_sec)
            seq = list(delta_t_future_sec)
            if k < len(seq):
                return float(seq[k])
            return dt_prev if dt_prev is not None and dt_prev > 0 else 1.0

        local_valid = (
            dt_prev is not None
            and 0 < dt_prev <= self.max_local_dt_sec
            and prev_lon is not None
            and prev_lat is not None
        )
        if (
            dt_prev is not None
            and dt_prev > self.max_local_dt_sec
            and prev_lon is not None
            and prev_lat is not None
        ):
            self._record_stat("dt_rejected")

        v_lon = v_lat = None
        if local_valid:
            v_lon = (float(current_lon) - float(prev_lon)) / dt_prev
            v_lat = (float(current_lat) - float(prev_lat)) / dt_prev

        if local_valid and historical_step is not None:
            mode = "fusion"
            self._record_stat("fusion_used")
            self._record_stat("local_velocity_used")
        elif local_valid:
            mode = "local"
            self._record_stat("local_velocity_used")
        elif historical_step is not None:
            mode = "historical"
            self._record_stat("historical_fallback_used")
        else:
            mode = "stationary"
            self._record_stat("stationary_fallback_used")

        lon, lat = float(current_lon), float(current_lat)
        out = []
        for k in range(steps):
            dtf = _dt_for_step(k)
            if dtf <= 0:
                dtf = dt_prev if dt_prev is not None and dt_prev > 0 else 1.0

            if mode == "fusion":
                hist_dx, hist_dy = historical_step
                local_step_lon = v_lon * dtf
                local_step_lat = v_lat * dtf
                alpha = self.local_velocity_alpha
                lon += alpha * local_step_lon + (1.0 - alpha) * float(hist_dx)
                lat += alpha * local_step_lat + (1.0 - alpha) * float(hist_dy)
            elif mode == "local":
                lon += v_lon * dtf
                lat += v_lat * dtf
            elif mode == "historical":
                hist_dx, hist_dy = historical_step
                lon += float(hist_dx)
                lat += float(hist_dy)
            out.append((lon, lat))
        return out
