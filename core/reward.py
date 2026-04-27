"""
Reward calculation for microservice DAG placement (ms 量纲 total_cost_ms).
对齐 command.md 阶段二：JIT 迁移、tearing、comm、future、SLA 跳变与截断。
"""

from core.geo import haversine_distance
from core.dag_utils import get_entry_nodes
from core.context import (
    TRIGGER_REACTIVE,
    TRIGGER_PROACTIVE,
    DISTANCE_THRESHOLD_KM,
    USER_SLA_TOLERANCE_MS,
)
from core.physics_utils import (
    calc_access_latency_ms,
    FIBER_SPEED_KM_MS,
    BASE_ROUTER_DELAY_MS,
)
import numpy as np

# --- 与 command.md 写死的物理 / 工程常量 ---
MIN_BW_MBPS = 50.0
MAX_BW_MBPS = 500.0
RPC_SIZE_MB = 0.005
MAX_TEARING_MB = 50.0
EDGE_BACKHAUL_MBPS = 1000.0
FUTURE_DECAY = 0.9
FUTURE_DIST_THRESHOLD = 15.0
SLA_PENALTY_MS = 5000.0
REWARD_CLIP_MIN = -10000.0
# Reactive 下迁移段额外系数（原 gamma 不对称语义的简化承接）
REACTIVE_MIGRATION_MULT = 1.5

# 与 context 单一真源（reward 内仍用此名便于阅读）
SLA_DISTANCE_THRESHOLD = DISTANCE_THRESHOLD_KM


def build_servers_info(servers_df):
    """Pre-build {server_id: (lat, lon)} lookup dict（列向量化，无 iterrows）。"""
    ar = servers_df[["edge_server_id", "latitude", "longitude"]].to_numpy(copy=False)
    return dict(
        zip(
            map(int, ar[:, 0]),
            zip(ar[:, 1].astype(np.float64), ar[:, 2].astype(np.float64)),
        )
    )


def calculate_microservice_reward(
    taxi_id,
    dag_info,
    current_assignments,
    previous_assignments,
    user_location,
    servers_info,
    alpha=1.0,
    beta=0.05,
    gamma=1.5,
    predicted_locations=None,
    delta=0.5,
    trigger_type=TRIGGER_REACTIVE,
):
    """
    Returns (reward, details). reward = max(-total_cost_ms, REWARD_CLIP_MIN).
    details 保留旧键名，值为 ms 或兼容字段，降低调用方断裂风险。
    """
    user_lat, user_lon = user_location
    entry_nodes = get_entry_nodes(dag_info)

    if entry_nodes:
        srv_lats = np.array(
            [servers_info[current_assignments[node]][0] for node in entry_nodes],
            dtype=np.float64,
        )
        srv_lons = np.array(
            [servers_info[current_assignments[node]][1] for node in entry_nodes],
            dtype=np.float64,
        )
        entry_distances_km = np.asarray(
            haversine_distance(user_lat, user_lon, srv_lats, srv_lons),
            dtype=np.float64,
        ).ravel()
    else:
        entry_distances_km = np.zeros(0, dtype=np.float64)

    if entry_distances_km.size:
        max_entry_dist_km = float(np.max(entry_distances_km))
        # calc_access_latency_ms 在 d>=0 上单调，与 max(calc(d) for d in ...) 数值一致
        access_latency_ms = float(calc_access_latency_ms(max_entry_dist_km))
    else:
        max_entry_dist_km = 0.0
        access_latency_ms = 0.0

    risk_ratio = min(max_entry_dist_km / SLA_DISTANCE_THRESHOLD, 1.0) if SLA_DISTANCE_THRESHOLD > 0 else 0.0
    effective_bandwidth = MIN_BW_MBPS + (MAX_BW_MBPS - MIN_BW_MBPS) * (risk_ratio ** 2)

    migration_delay_ms = 0.0
    for node, node_props in dag_info["nodes"].items():
        if current_assignments[node] != previous_assignments[node]:
            image_mb = float(node_props["image_mb"])
            state_mb = float(node_props["state_mb"])
            delta_ms = ((image_mb + state_mb) / effective_bandwidth) * 1000.0
            if trigger_type == TRIGGER_REACTIVE:
                delta_ms *= REACTIVE_MIGRATION_MULT
            migration_delay_ms += delta_ms

    max_traffic = max(dag_info["edges"].values()) if dag_info["edges"] else 0.0
    tearing_delay_ms = 0.0
    comm_delay_ms = 0.0

    edges_items = list(dag_info["edges"].items())
    if edges_items:
        traffics = np.array([float(t) for (_, _), t in edges_items], dtype=np.float64)
        src_srv = np.array([current_assignments[s] for (s, d), _ in edges_items])
        dst_srv = np.array([current_assignments[d] for (s, d), _ in edges_items])
        same = src_srv == dst_srv
        src_lat = np.array([servers_info[s][0] for s in src_srv], dtype=np.float64)
        src_lon = np.array([servers_info[s][1] for s in src_srv], dtype=np.float64)
        dst_lat = np.array([servers_info[s][0] for s in dst_srv], dtype=np.float64)
        dst_lon = np.array([servers_info[s][1] for s in dst_srv], dtype=np.float64)
        edge_dist_km = np.asarray(
            haversine_distance(src_lat, src_lon, dst_lat, dst_lon),
            dtype=np.float64,
        )
        norm_traffic = np.where(max_traffic > 0, traffics / max_traffic, 0.0)
        cross_mb = np.minimum(traffics * RPC_SIZE_MB, MAX_TEARING_MB)
        valid = ~same
        tearing_delay_ms = float(np.sum(valid * (cross_mb / EDGE_BACKHAUL_MBPS) * 1000.0))
        access_edge_ms = (np.maximum(0.0, edge_dist_km) / FIBER_SPEED_KM_MS) + BASE_ROUTER_DELAY_MS
        comm_delay_ms = float(np.sum(valid * (norm_traffic * access_edge_ms)))

    future_delay_ms = 0.0
    if predicted_locations and entry_nodes:
        pred = np.asarray(predicted_locations, dtype=np.float64)
        if pred.ndim != 2 or pred.shape[1] != 2:
            pred = np.reshape(pred, (-1, 2))
        h = pred.shape[0]
        plats = pred[:, 0][:, np.newaxis]
        plons = pred[:, 1][:, np.newaxis]
        e_lats = np.array(
            [servers_info[current_assignments[node]][0] for node in entry_nodes],
            dtype=np.float64,
        )[np.newaxis, :]
        e_lons = np.array(
            [servers_info[current_assignments[node]][1] for node in entry_nodes],
            dtype=np.float64,
        )[np.newaxis, :]
        dist_mat = haversine_distance(plats, plons, e_lats, e_lons)
        d_per_step = np.max(dist_mat, axis=1)
        w = FUTURE_DECAY ** np.arange(h, dtype=np.float64)
        excess = np.maximum(0.0, d_per_step - FUTURE_DIST_THRESHOLD)
        prop = np.where(excess <= 0.0, 0.0, excess / FIBER_SPEED_KM_MS)
        w_sum = float(np.sum(w))
        if w_sum > 0:
            future_delay_ms = float(np.dot(prop, w)) / w_sum

    sla_violations = (
        int(np.sum(entry_distances_km > SLA_DISTANCE_THRESHOLD))
        if entry_distances_km.size
        else 0
    )

    spatial_violation = max_entry_dist_km > SLA_DISTANCE_THRESHOLD
    qos_violation = access_latency_ms > USER_SLA_TOLERANCE_MS
    sla_penalty_ms = SLA_PENALTY_MS if (spatial_violation or qos_violation) else 0.0

    total_cost_ms = (
        access_latency_ms
        + migration_delay_ms
        + tearing_delay_ms
        + comm_delay_ms
        + future_delay_ms
        + sla_penalty_ms
    )

    reward = max(-total_cost_ms, REWARD_CLIP_MIN)

    details = {
        "access_latency": access_latency_ms,
        "communication_cost": comm_delay_ms,
        "migration_cost": migration_delay_ms,
        "future_penalty": future_delay_ms,
        "tearing_penalty": tearing_delay_ms,
        "sla_violations": sla_violations,
        "risk_ratio": risk_ratio,
        "state_divisor": effective_bandwidth,
        "total_cost": total_cost_ms,
        "reward": reward,
        "trigger_type": trigger_type,
        "sla_penalty_ms": sla_penalty_ms,
        "access_latency_ms": access_latency_ms,
        "total_cost_ms": total_cost_ms,
    }
    return reward, details
