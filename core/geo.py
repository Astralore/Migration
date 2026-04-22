import numpy as np


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two points in km."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def find_k_nearest_servers(lat, lon, servers_df, k=3):
    """Return k nearest servers as list of (id, dist, lat, lon) tuples."""
    distances = []
    for _, server in servers_df.iterrows():
        dist = haversine_distance(lat, lon, server['latitude'], server['longitude'])
        distances.append((server['edge_server_id'], dist, server['latitude'], server['longitude']))
    distances.sort(key=lambda x: x[1])
    return distances[:k]
