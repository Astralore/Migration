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

Graph-based layout (TriggerAwareGraphDQN):
  node_features : (N, 3) - [image_mb, state_mb, is_stateful] per node
  adj_matrix    : (N, N) - normalized traffic weights
  trigger_context: (2,)  - one-hot encoding of trigger type
  sa_priors     : (N, 2) - [sa_dist, sa_stay] per node
"""

import numpy as np

from core.geo import haversine_distance
from core.context import TRIGGER_PROACTIVE, TRIGGER_REACTIVE
from core.dag_utils import get_entry_nodes

MOBILITY_NORM = 0.01  # ~1.1 km in degrees; keeps feature in [-1, 1] range

# v3.7: SLA threshold for continuous risk feature calculation
# This should match the value in core/reward.py
SLA_DISTANCE_THRESHOLD_KM = 15.0


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


def build_graph_state(
    taxi_id,
    dag_info,
    current_assignments,
    servers_info,
    trigger_type,
    sa_proposal,
    candidates,
    current_lat,
    current_lon,
):
    """
    Build graph-structured state for Trigger-Aware Graph Attention DQN.

    This function constructs a complete graph representation of the microservice
    DAG, incorporating trigger-type awareness for differentiated decision-making.

    Parameters
    ----------
    taxi_id : str
        Taxi identifier (for logging/debugging).
    dag_info : dict
        DAG definition with 'nodes' and 'edges'.
    current_assignments : dict
        Current server assignment for each node.
    servers_info : dict
        Server ID -> (lat, lon) mapping.
    trigger_type : str
        Either TRIGGER_PROACTIVE or TRIGGER_REACTIVE.
    sa_proposal : dict
        SA-proposed server assignment for each node.
    candidates : list
        List of (server_id, distance) tuples for candidate servers.
    current_lat, current_lon : float
        User's current location.

    Returns
    -------
    dict with keys:
        node_features : np.ndarray, shape (N, 3)
            Per-node features: [image_mb/200, state_mb/256, is_stateful]
        adj_matrix : np.ndarray, shape (N, N)
            Normalized adjacency matrix with traffic weights.
        trigger_context : np.ndarray, shape (3,)
            v3.7: [proactive_flag, reactive_flag, risk_ratio]
            - PROACTIVE: [1.0, 0.0, risk_ratio] where risk_ratio ∈ [0,1]
            - REACTIVE:  [0.0, 1.0, 1.0] (always max risk)
            risk_ratio = min(max_entry_dist / SLA_THRESHOLD, 1.0)
        sa_priors : np.ndarray, shape (N, 2)
            Per-node SA prior: [sa_server_dist/50, sa_stay_flag]
        node_names : list
            Ordered list of node names (for index mapping).
    """
    # Build ordered node list (consistent indexing)
    node_names = sorted(dag_info['nodes'].keys())
    n_nodes = len(node_names)
    node_to_idx = {name: i for i, name in enumerate(node_names)}

    # === Node Features (N, 3) ===
    # Physical intuition: image_mb affects cold-start latency,
    # state_mb affects migration cost (especially for stateful nodes),
    # is_stateful is critical for asymmetric proactive/reactive cost.
    node_features = np.zeros((n_nodes, 3), dtype=np.float32)
    for i, node_name in enumerate(node_names):
        node_props = dag_info['nodes'][node_name]
        node_features[i, 0] = node_props['image_mb'] / 200.0
        node_features[i, 1] = node_props['state_mb'] / 256.0
        node_features[i, 2] = float(node_props['is_stateful'])

    # === Adjacency Matrix (N, N) ===
    # Traffic-weighted edges: higher traffic = stronger dependency.
    # Normalized to [0, 1] for stable training.
    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    max_traffic = max(dag_info['edges'].values()) if dag_info['edges'] else 1.0
    max_traffic = max(max_traffic, 1e-6)  # Avoid division by zero

    for (src, dst), traffic in dag_info['edges'].items():
        if src in node_to_idx and dst in node_to_idx:
            i, j = node_to_idx[src], node_to_idx[dst]
            norm_traffic = traffic / max_traffic
            adj_matrix[i, j] = norm_traffic
            adj_matrix[j, i] = norm_traffic  # Undirected for message passing

    # Add self-loops for graph convolution stability
    np.fill_diagonal(adj_matrix, 1.0)

    # === Trigger Context (3,) - v3.7: With Continuous Risk Feature ===
    # Physical intuition:
    # - PROACTIVE: system has time for background state sync → lower migration cost
    # - REACTIVE: urgent migration needed → higher cost, prioritize speed
    # 
    # v3.7 JIT Enhancement: Add continuous "risk_ratio" feature
    # =========================================================
    # Problem: Fixed binary trigger caused "Premature Handover" trap
    # - Proactive trigger fired at 5km → agent immediately migrated
    # - But migration at 5km is premature (too early, causes DAG tear)
    # 
    # Solution: Inject risk_ratio = max_entry_dist / SLA_THRESHOLD
    # - At 5km: risk_ratio = 5/15 = 0.33 → "not urgent, don't migrate yet"
    # - At 12km: risk_ratio = 12/15 = 0.80 → "getting urgent, prepare to migrate"
    # - At 15km: risk_ratio = 15/15 = 1.0 → "critical, must migrate now"
    # 
    # This allows GAT to learn distance-aware attention patterns:
    # - Low risk_ratio → focus on stability, avoid premature migration
    # - High risk_ratio → focus on latency, prioritize migration
    # =========================================================
    
    # Calculate max distance from entry nodes to user (risk indicator)
    entry_nodes = get_entry_nodes(dag_info)
    max_entry_dist = 0.0
    for node in entry_nodes:
        srv_id = current_assignments[node]
        srv_lat, srv_lon = servers_info[srv_id]
        node_dist = haversine_distance(current_lat, current_lon, srv_lat, srv_lon)
        max_entry_dist = max(max_entry_dist, node_dist)
    
    # risk_ratio: continuous risk signal, clamped to [0, 1]
    risk_ratio = min(max_entry_dist / SLA_DISTANCE_THRESHOLD_KM, 1.0)
    
    if trigger_type == TRIGGER_PROACTIVE:
        # PROACTIVE: include continuous risk_ratio for JIT decision
        trigger_context = np.array([1.0, 0.0, risk_ratio], dtype=np.float32)
    else:  # REACTIVE or fallback
        # REACTIVE: always max risk (violation already occurring)
        trigger_context = np.array([0.0, 1.0, 1.0], dtype=np.float32)

    # === SA Priors (N, 2) ===
    # Per-node guidance from Simulated Annealing:
    # - sa_dist: distance from SA-proposed server to user (quality indicator)
    # - sa_stay: whether SA suggests keeping current placement (stability)
    sa_priors = np.zeros((n_nodes, 2), dtype=np.float32)
    for i, node_name in enumerate(node_names):
        sa_server = sa_proposal[node_name]
        current_server = current_assignments[node_name]

        # SA proposed server distance to user
        sa_lat, sa_lon = servers_info[sa_server]
        sa_dist = haversine_distance(current_lat, current_lon, sa_lat, sa_lon)
        sa_priors[i, 0] = min(sa_dist / 50.0, 1.0)

        # SA stay flag: 1 if SA suggests no migration for this node
        sa_priors[i, 1] = 1.0 if sa_server == current_server else 0.0

    return {
        'node_features': node_features,
        'adj_matrix': adj_matrix,
        'trigger_context': trigger_context,
        'sa_priors': sa_priors,
        'node_names': node_names,
    }
