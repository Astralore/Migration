"""
State builders and action masks for CTDE-GAT-MARL microservice migration.

P1 (v2.1): DAG-type family encoding, entry counterfactual candidate features,
entry-first action masks, coordinated with max-one migration in marl_gat.py.
"""

import numpy as np

from core.context import TRIGGER_PROACTIVE, DISTANCE_THRESHOLD_KM
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

# Feature dimensions (P1 expanded)
MARL_NODE_FEAT_DIM = 17
MARL_TRIGGER_BASE_DIM = 3
MARL_DAG_TYPE_FAMILY_DIM = 6
MARL_TRIGGER_CONTEXT_DIM = MARL_TRIGGER_BASE_DIM + MARL_DAG_TYPE_FAMILY_DIM
MARL_CANDIDATE_FEAT_PER_SERVER = 4
MARL_CANDIDATE_FEATURE_DIM = MAX_CANDIDATES * MARL_CANDIDATE_FEAT_PER_SERVER

DAG_TYPE_FAMILIES = (
    "FanIn",
    "FanOut",
    "Diamond",
    "Pipeline",
    "Data_Heavy",
    "Compute_Heavy",
)


def encode_dag_type_family(dag_type_name):
    """One-hot over coarse DAG family (6 dims)."""
    vec = np.zeros(MARL_DAG_TYPE_FAMILY_DIM, dtype=np.float32)
    if not dag_type_name:
        return vec
    for i, family in enumerate(DAG_TYPE_FAMILIES):
        if dag_type_name.startswith(family):
            vec[i] = 1.0
            return vec
    return vec


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


def _entry_counterfactual_features(
    dag_info,
    current_assignments,
    candidates,
    current_lat,
    current_lon,
    servers_info,
):
    """
    Per-candidate entry SLA counterfactual (primary entry hypothetically on candidate k).

    Returns flat candidate vector (12,) and per-node entry CF gains for primary (3,).
    """
    candidate_features = np.zeros((MAX_CANDIDATES, MARL_CANDIDATE_FEAT_PER_SERVER), dtype=np.float32)
    node_entry_cf = np.zeros((len(dag_info["nodes"]), 3), dtype=np.float32)
    entry_nodes = get_service_entry_nodes(dag_info)
    if not entry_nodes:
        return candidate_features.reshape(-1), node_entry_cf

    primary_entry = entry_nodes[0]
    node_names = sorted(dag_info["nodes"].keys())
    primary_idx = node_names.index(primary_entry) if primary_entry in node_names else None

    def _dist_to_user(server_id):
        lat, lon = servers_info[server_id]
        return float(haversine_distance(current_lat, current_lon, lat, lon))

    entry_dists = [
        _dist_to_user(current_assignments[node])
        for node in entry_nodes
    ]
    current_max_entry_dist = max(entry_dists) if entry_dists else 0.0
    thresh = max(float(DISTANCE_THRESHOLD_KM), 1e-6)

    cf_gains = []
    for idx in range(MAX_CANDIDATES):
        if idx >= len(candidates or []):
            cf_gains.append(0.0)
            continue
        server_id = candidates[idx][0]
        cand_dist = candidates[idx][1] if len(candidates[idx]) > 1 else _dist_to_user(server_id)
        candidate_features[idx, 0] = min(float(cand_dist) / 50.0, 1.0)
        candidate_features[idx, 1] = 1.0 if server_id in current_assignments.values() else 0.0

        new_entry_dists = []
        for node in entry_nodes:
            if node == primary_entry:
                new_entry_dists.append(_dist_to_user(server_id))
            else:
                new_entry_dists.append(_dist_to_user(current_assignments[node]))
        new_max = max(new_entry_dists) if new_entry_dists else cand_dist
        gain_km = max(0.0, current_max_entry_dist - new_max)
        gain_norm = min(gain_km / thresh, 1.0)
        cf_gains.append(gain_norm)
        candidate_features[idx, 2] = gain_norm
        candidate_features[idx, 3] = min(_dist_to_user(server_id) / 50.0, 1.0)

    if primary_idx is not None:
        for j in range(min(3, len(cf_gains))):
            node_entry_cf[primary_idx, j] = cf_gains[j]

    return candidate_features.reshape(-1), node_entry_cf


def build_marl_action_mask(
    candidates,
    current_server,
    *,
    node_movable=True,
    entry_first=False,
    is_entry_node=False,
):
    """
    Return a bool mask for [STAY, CANDIDATE_1, CANDIDATE_2, CANDIDATE_3].

    P1 entry-first: non-entry nodes only allow STAY.
    """
    mask = np.zeros(MARL_ACTION_DIM, dtype=bool)
    mask[ACTION_STAY] = True
    if entry_first and not is_entry_node:
        return mask
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
    dag_type=None,
    p1_entry_first=False,
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

    cf_flat, node_entry_cf_by_idx = _entry_counterfactual_features(
        dag_info,
        current_assignments,
        candidates,
        current_lat,
        current_lon,
        servers_info,
    )

    node_features = np.zeros((n_nodes, MARL_NODE_FEAT_DIM), dtype=np.float32)
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
                float(node_entry_cf_by_idx[i, 0]),
                float(node_entry_cf_by_idx[i, 1]),
                float(node_entry_cf_by_idx[i, 2]),
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

    trigger_base = np.array(
        [1.0, 0.0, risk_ratio] if trigger_type == TRIGGER_PROACTIVE else [0.0, 1.0, 1.0],
        dtype=np.float32,
    )
    dag_type_vec = encode_dag_type_family(dag_type)
    trigger_context = np.concatenate([trigger_base, dag_type_vec]).astype(np.float32)

    movable = node_movable or {}
    action_masks = np.zeros((n_nodes, MARL_ACTION_DIM), dtype=bool)
    for i, node_name in enumerate(node_names):
        action_masks[i] = build_marl_action_mask(
            candidates,
            current_assignments[node_name],
            node_movable=(not is_external_node(node_name)) and bool(movable.get(node_name, True)),
            entry_first=p1_entry_first,
            is_entry_node=node_name in entry_nodes,
        )

    return {
        "node_names": node_names,
        "node_features": node_features,
        "adj_matrix": adj_matrix,
        "trigger_context": trigger_context,
        "mobility_context": _mobility_context(current_lat, current_lon, predicted_locations),
        "candidate_features": cf_flat,
        "action_masks": action_masks,
        "risk_ratio": risk_ratio,
        "dag_type": dag_type,
    }
