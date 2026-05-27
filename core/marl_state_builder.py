"""
State builders and action masks for CTDE-GAT-MARL microservice migration.

This module intentionally has no SA prior.  It only uses physical mobility,
candidate edge servers, trigger context, and the microservice DAG itself.
"""

import numpy as np

from core.context import TRIGGER_PROACTIVE
from core.dag_utils import get_service_entry_nodes, is_external_node, topological_sort
from core.geo import haversine_distance
from core.reward import dag_max_traffic_log_rpc, traffic_log_rpc_feature
from core.state_builder import MOBILITY_NORM, SLA_DISTANCE_THRESHOLD_KM


ACTION_STAY = 0
ACTION_CANDIDATE_1 = 1
ACTION_CANDIDATE_2 = 2
ACTION_CANDIDATE_3 = 3
MARL_ACTION_DIM = 4
MAX_CANDIDATES = 3


def _mobility_context(user_lat, user_lon, predicted_locations):
    if not predicted_locations:
        return np.zeros(2, dtype=np.float32)
    arr = np.asarray(predicted_locations, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        arr = np.reshape(arr, (-1, 2))
    delta = arr - np.array([user_lat, user_lon], dtype=np.float64)
    avg_dlat, avg_dlon = np.mean(delta, axis=0)
    return np.array(
        [
            float(np.clip(avg_dlat / MOBILITY_NORM, -1.0, 1.0)),
            float(np.clip(avg_dlon / MOBILITY_NORM, -1.0, 1.0)),
        ],
        dtype=np.float32,
    )


def build_marl_action_mask(candidates, current_server, *, node_movable=True):
    """
    Return a bool mask for [STAY, CANDIDATE_1, CANDIDATE_2, CANDIDATE_3].

    Invalid candidate actions are masked before softmax/sampling.  If a node is
    not movable, only STAY remains legal.  If every action is masked by an
    unexpected input, STAY is restored as a safe physical fallback.
    """
    mask = np.zeros(MARL_ACTION_DIM, dtype=bool)
    mask[ACTION_STAY] = True
    if node_movable:
        for idx in range(min(len(candidates or []), MAX_CANDIDATES)):
            server_id = candidates[idx][0]
            if server_id != current_server:
                mask[ACTION_CANDIDATE_1 + idx] = True
    if not bool(mask.any()):
        mask[ACTION_STAY] = True
    return mask


def action_to_server(action, candidates, current_server):
    """Map a masked discrete action to the target server id."""
    action = int(action)
    if action == ACTION_STAY:
        return current_server
    cand_idx = action - ACTION_CANDIDATE_1
    if 0 <= cand_idx < len(candidates or []):
        return candidates[cand_idx][0]
    return current_server


def build_marl_graph_state(
    taxi_id,
    dag_info,
    current_assignments,
    servers_info,
    trigger_type,
    candidates,
    current_lat,
    current_lon,
    predicted_locations=None,
    node_movable=None,
):
    """
    Build graph state for CTDE-GAT-MARL.

    Returns node-local features, adjacency, trigger/mobility context, candidate
    features, per-node action masks, and a stable node order.
    """
    del taxi_id  # kept for signature parity/debugging hooks.
    node_names = sorted(dag_info["nodes"].keys())
    n_nodes = len(node_names)
    node_to_idx = {name: i for i, name in enumerate(node_names)}
    entry_nodes = set(get_service_entry_nodes(dag_info))

    max_log_rpc = max(float(dag_max_traffic_log_rpc(dag_info)), 1e-6)

    topo_order = topological_sort(dag_info)
    topo_idx = {name: idx for idx, name in enumerate(topo_order)}
    in_degrees = {name: 0 for name in node_names}
    out_degrees = {name: 0 for name in node_names}
    for src, dst in dag_info["edges"]:
        if src in out_degrees:
            out_degrees[src] += 1
        if dst in in_degrees:
            in_degrees[dst] += 1
    max_degree = max(1, n_nodes - 1)
    topo_denom = max(1, n_nodes - 1)

    node_log_rpc_load = []
    node_neighbor_sets = []
    for node_name in node_names:
        load = 0.0
        neighbors = set()
        for (src, dst), traffic in dag_info["edges"].items():
            if src == node_name:
                load += traffic_log_rpc_feature(traffic)
                neighbors.add(dst)
            elif dst == node_name:
                load += traffic_log_rpc_feature(traffic)
                neighbors.add(src)
        node_log_rpc_load.append(load)
        node_neighbor_sets.append(neighbors)

    max_node_log_rpc = max(node_log_rpc_load) if node_log_rpc_load else 1.0
    max_node_log_rpc = max(float(max_node_log_rpc), 1e-6)

    node_features = np.zeros((n_nodes, 14), dtype=np.float32)
    for i, node_name in enumerate(node_names):
        props = dag_info["nodes"][node_name]
        current_server = current_assignments[node_name]
        srv_lat, srv_lon = servers_info[current_server]
        node_dist = float(haversine_distance(current_lat, current_lon, srv_lat, srv_lon))

        neighbors = node_neighbor_sets[i]
        same_neighbors = sum(
            1 for nb in neighbors
            if current_assignments.get(nb) == current_server
        )
        same_ratio = (same_neighbors / len(neighbors)) if neighbors else 0.0

        node_features[i] = np.array(
            [
                float(props["image_mb"]) / 200.0,
                float(props["state_mb"]) / 512.0,
                float(props["is_stateful"]),
                min(node_log_rpc_load[i] / max_node_log_rpc, 1.0),
                min(node_dist / 50.0, 1.0),
                1.0 if node_name in entry_nodes else 0.0,
                same_ratio,
                min(n_nodes / 10.0, 1.0),
                min(in_degrees[node_name] / max_degree, 1.0),
                min(out_degrees[node_name] / max_degree, 1.0),
                min(topo_idx.get(node_name, i) / topo_denom, 1.0),
                1.0 if out_degrees[node_name] == 0 else 0.0,
                1.0 if float(props["state_mb"]) >= 256.0 else 0.0,
                1.0 if is_external_node(node_name) else 0.0,
            ],
            dtype=np.float32,
        )

    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for (src, dst), traffic in dag_info["edges"].items():
        if src in node_to_idx and dst in node_to_idx:
            i, j = node_to_idx[src], node_to_idx[dst]
            weight = traffic_log_rpc_feature(traffic) / max_log_rpc
            adj_matrix[i, j] = weight
            adj_matrix[j, i] = weight
    np.fill_diagonal(adj_matrix, 1.0)

    if entry_nodes:
        entry_dists = [
            haversine_distance(
                current_lat,
                current_lon,
                servers_info[current_assignments[node]][0],
                servers_info[current_assignments[node]][1],
            )
            for node in entry_nodes
        ]
        risk_ratio = min(float(max(entry_dists)) / SLA_DISTANCE_THRESHOLD_KM, 1.0)
    else:
        risk_ratio = 0.0

    trigger_context = np.array(
        [1.0, 0.0, risk_ratio] if trigger_type == TRIGGER_PROACTIVE else [0.0, 1.0, 1.0],
        dtype=np.float32,
    )

    candidate_features = np.zeros((MAX_CANDIDATES, 2), dtype=np.float32)
    for idx in range(min(len(candidates or []), MAX_CANDIDATES)):
        server_id = candidates[idx][0]
        dist_km = candidates[idx][1] if len(candidates[idx]) > 1 else 0.0
        candidate_features[idx, 0] = min(float(dist_km) / 50.0, 1.0)
        candidate_features[idx, 1] = 1.0 if server_id in current_assignments.values() else 0.0

    movable = node_movable or {}
    action_masks = np.zeros((n_nodes, MARL_ACTION_DIM), dtype=bool)
    for i, node_name in enumerate(node_names):
        action_masks[i] = build_marl_action_mask(
            candidates,
            current_assignments[node_name],
            node_movable=(not is_external_node(node_name)) and bool(movable.get(node_name, True)),
        )

    return {
        "node_names": node_names,
        "node_features": node_features,
        "adj_matrix": adj_matrix,
        "trigger_context": trigger_context,
        "mobility_context": _mobility_context(current_lat, current_lon, predicted_locations),
        "candidate_features": candidate_features.reshape(-1),
        "action_masks": action_masks,
        "risk_ratio": risk_ratio,
    }
