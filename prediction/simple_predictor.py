"""Simple velocity-based trajectory predictor extracted from Context_Integratoin.py."""

import numpy as np
from tqdm import tqdm


class SimpleTrajectoryPredictor:
    def __init__(self, forecast_horizon=3):
        self.forecast_horizon = forecast_horizon
        self.velocity_factors = {}

    def fit(self, df):
        print("Fitting trajectory predictor...")
        for taxi_id in tqdm(df['taxi_id'].unique(), desc="Learning velocities"):
            taxi_data = df[df['taxi_id'] == taxi_id].sort_values('date_time')
            if len(taxi_data) < 100:
                continue
            lons = taxi_data['longitude'].values
            lats = taxi_data['latitude'].values
            dx = np.diff(lons)
            dy = np.diff(lats)
            if len(dx) > 0:
                self.velocity_factors[taxi_id] = (np.mean(dx), np.mean(dy))
        return self

    def predict_future(self, current_lon, current_lat, taxi_id, steps=None):
        if steps is None:
            steps = self.forecast_horizon
        if taxi_id not in self.velocity_factors:
            return [(current_lon, current_lat)] * steps
        dx, dy = self.velocity_factors[taxi_id]
        future = []
        lon, lat = current_lon, current_lat
        for _ in range(steps):
            lon += dx
            lat += dy
            future.append((lon, lat))
        return future
