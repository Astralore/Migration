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
from core.physics_utils import calc_access_latency_ms, propagation_latency_ms

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
    """Pre-build {server_id: (lat, lon)} lookup dict."""
    info = {}
    for _, row in servers_df.iterrows():
        info[row['edge_server_id']] = (row['latitude'], row['longitude'])
    return info


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

    entry_distances_km = []
    for node in entry_nodes:
        srv_id = current_assignments[node]
        srv_lat, srv_lon = servers_info[srv_id]
        entry_distances_km.append(
            haversine_distance(user_lat, user_lon, srv_lat, srv_lon)
        )

    max_entry_dist_km = max(entry_distances_km) if entry_distances_km else 0.0
    access_latency_ms = (
        max(calc_access_latency_ms(d) for d in entry_distances_km)
        if entry_distances_km
        else 0.0
    )

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

    for (src, dst), traffic in dag_info["edges"].items():
        src_server = current_assignments[src]
        dst_server = current_assignments[dst]
        if src_server == dst_server:
            continue
        src_lat, src_lon = servers_info[src_server]
        dst_lat, dst_lon = servers_info[dst_server]
        edge_dist_km = haversine_distance(src_lat, src_lon, dst_lat, dst_lon)
        norm_traffic = traffic / max_traffic if max_traffic > 0 else 0.0

        cross_mb = min(float(traffic) * RPC_SIZE_MB, MAX_TEARING_MB)
        tearing_delay_ms += (cross_mb / EDGE_BACKHAUL_MBPS) * 1000.0

        comm_delay_ms += norm_traffic * calc_access_latency_ms(edge_dist_km)

    future_delay_ms = 0.0
    if predicted_locations and entry_nodes:
        future_distances = []
        for pred_lat, pred_lon in predicted_locations:
            step_dists = []
            for node in entry_nodes:
                srv_id = current_assignments[node]
                srv_lat, srv_lon = servers_info[srv_id]
                step_dists.append(
                    haversine_distance(pred_lat, pred_lon, srv_lat, srv_lon)
                )
            future_distances.append(max(step_dists) if step_dists else 0.0)

        if future_distances:
            raw = 0.0
            weight_sum = 0.0
            for i, d in enumerate(future_distances):
                w = FUTURE_DECAY ** i
                excess_km = max(0.0, float(d) - FUTURE_DIST_THRESHOLD)
                raw += propagation_latency_ms(excess_km) * w
                weight_sum += w
            future_delay_ms = raw / weight_sum if weight_sum > 0 else 0.0

    sla_violations = 0
    for d in entry_distances_km:
        if d > SLA_DISTANCE_THRESHOLD:
            sla_violations += 1

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
