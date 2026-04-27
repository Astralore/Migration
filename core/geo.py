import numpy as np


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two points in km.

    Supports NumPy broadcasting (e.g. one user vs many servers:
    scalar lat1/lon1 with lat2/lon2 shaped (N,), or (H,1) vs (1,E) grids).
    """
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def find_k_nearest_servers(lat, lon, servers_df, k=3):
    """Return k nearest servers as list of (id, dist, lat, lon) tuples, sorted by dist ascending."""
    n = len(servers_df)
    if n == 0:
        return []
    k_req = int(k)
    if k_req <= 0:
        return []
    k_eff = min(k_req, n)
    ids = servers_df["edge_server_id"].to_numpy()
    slat = servers_df["latitude"].to_numpy(dtype=np.float64, copy=False)
    slon = servers_df["longitude"].to_numpy(dtype=np.float64, copy=False)
    dists = haversine_distance(lat, lon, slat, slon)
    if k_eff == n:
        # kind='stable' 与 pandas iterrows + list.sort 的并列键顺序一致（无损）
        order = np.argsort(dists, kind="stable")
    else:
        part = np.argpartition(dists, k_eff - 1)[:k_eff]
        order = part[np.argsort(dists[part], kind="stable")]
    return [(int(ids[i]), float(dists[i]), float(slat[i]), float(slon[i])) for i in order]
