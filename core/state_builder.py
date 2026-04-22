"""
State vector builders for microservice migration DQN / Hybrid agents.

All features are purely physical, temporal, or microservice-topology related.

16-dim base layout (DQN):
  [0-5]   Global    : lat, lon, gateway_dist, hour, weekday, n_candidates
  [6-9]   Node      : image_mb, state_mb, is_stateful, node_traffic_ratio
  [10-13] Topology  : neighbor_same_server_ratio, node_server_dist, is_entry, dag_size
  [14-15] Mobility  : avg_delta_lat, avg_delta_lon  (from trajectory prediction)

18-dim hybrid layout:
  [0-15] base (same as above)
  [16]   SA prior   : sa_proposed_server_dist
  [17]   SA prior   : sa_stay_flag
"""

import numpy as np

from core.geo import haversine_distance

MOBILITY_NORM = 0.01  # ~1.1 km in degrees; keeps feature in [-1, 1] range


def _mobility_features(user_lat, user_lon, predicted_locations):
    """Compute 2-dim normalised movement-trend features from predictions."""
    if not predicted_locations:
        return 0.0, 0.0
    avg_dlat = np.mean([lat - user_lat for lat, lon in predicted_locations])
    avg_dlon = np.mean([lon - user_lon for lat, lon in predicted_locations])
    feat_dlat = np.clip(avg_dlat / MOBILITY_NORM, -1.0, 1.0)
    feat_dlon = np.clip(avg_dlon / MOBILITY_NORM, -1.0, 1.0)
    return float(feat_dlat), float(feat_dlon)


def build_node_state(
    user_lat, user_lon, timestamp, gateway_dist,
    ms_node, dag_info, current_assignments, candidates,
    servers_info, entry_nodes_set,
    predicted_locations=None,
):
    """Build a 16-dim normalised state vector for a single microservice node."""
    node_info = dag_info['nodes'][ms_node]

    # Global features (6)
    feat_lat = user_lat / 90.0
    feat_lon = user_lon / 180.0
    feat_dist = min(gateway_dist / 50.0, 1.0)
    feat_hour = timestamp.hour / 24.0
    feat_weekday = timestamp.weekday() / 7.0
    feat_candidates = len(candidates) / 10.0

    # Node features (4)
    feat_image = node_info['image_mb'] / 200.0
    feat_state_mb = node_info['state_mb'] / 256.0
    feat_stateful = float(node_info['is_stateful'])

    node_traffic = 0.0
    max_traffic = 0.0
    for (src, dst), traffic in dag_info['edges'].items():
        if src == ms_node or dst == ms_node:
            node_traffic += traffic
        if traffic > max_traffic:
            max_traffic = traffic
    feat_traffic = node_traffic / max_traffic if max_traffic > 0 else 0.0

    # Topology context (4)
    current_server = current_assignments[ms_node]
    neighbors = set()
    for (src, dst) in dag_info['edges'].keys():
        if src == ms_node:
            neighbors.add(dst)
        elif dst == ms_node:
            neighbors.add(src)
    total_neighbors = len(neighbors)
    same_count = sum(1 for n in neighbors if current_assignments[n] == current_server)
    feat_same_ratio = same_count / total_neighbors if total_neighbors > 0 else 0.0

    srv_lat, srv_lon = servers_info[current_server]
    feat_node_dist = min(
        haversine_distance(user_lat, user_lon, srv_lat, srv_lon) / 50.0, 1.0
    )

    feat_is_entry = 1.0 if ms_node in entry_nodes_set else 0.0
    feat_dag_size = len(dag_info['nodes']) / 10.0

    # Mobility trend (2)
    feat_dlat, feat_dlon = _mobility_features(user_lat, user_lon, predicted_locations)

    return np.array([
        feat_lat, feat_lon, feat_dist, feat_hour, feat_weekday, feat_candidates,
        feat_image, feat_state_mb, feat_stateful, feat_traffic,
        feat_same_ratio, feat_node_dist, feat_is_entry, feat_dag_size,
        feat_dlat, feat_dlon,
    ], dtype=np.float32)


def build_hybrid_node_state(
    user_lat, user_lon, timestamp, gateway_dist,
    ms_node, dag_info, current_assignments, candidates,
    servers_info, entry_nodes_set,
    sa_proposed_server,
    predicted_locations=None,
):
    """18-dim state: base 16 dims + 2 SA-prior dims."""
    base = build_node_state(
        user_lat, user_lon, timestamp, gateway_dist,
        ms_node, dag_info, current_assignments, candidates,
        servers_info, entry_nodes_set,
        predicted_locations=predicted_locations,
    )

    current_server = current_assignments[ms_node]
    sa_lat, sa_lon = servers_info[sa_proposed_server]
    feat_sa_dist = min(
        haversine_distance(user_lat, user_lon, sa_lat, sa_lon) / 50.0, 1.0
    )
    feat_sa_stay = 1.0 if sa_proposed_server == current_server else 0.0

    return np.append(base, [feat_sa_dist, feat_sa_stay]).astype(np.float32)
