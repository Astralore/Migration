"""
Reward calculation for microservice DAG placement (ms 量纲 total_cost_ms).
对齐 command.md 阶段二：JIT 迁移、tearing、comm、future、SLA 跳变与截断。

Reward schemes (env ``REWARD_SCHEME``):
  - ``v1`` (default): base + linear + quadratic SLA; polynomial migration; -log1p objective.
  - ``v2``: pure alpha*E^2 SLA (+ optional QoS^2); exp(size/tau) migration; linear -objective/scale.
"""

import math
import os
from collections import defaultdict

from core.geo import haversine_distance

from core.dag_utils import (
    get_deployable_nodes,
    get_service_entry_nodes,
    topological_sort,
)
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
SLA_BASE_PENALTY_MS = 2000.0
SLA_PENALTY_PER_KM_MS = 500.0
SLA_QUADRATIC_PENALTY_PER_KM2_MS = 80.0
SEVERE_SLA_EXCESS_KM = 5.0
# Backward-compatible reference scale.  Actual SLA cost is now linear excess.
SLA_PENALTY_MS = SLA_BASE_PENALTY_MS + SLA_PENALTY_PER_KM_MS * 10.0
# C2：用 log 压缩真实物理代价，避免 -10 硬截断让严重违规/昂贵迁移不可区分。
REWARD_COST_SCALE_MS = 1000.0
CORE_MIGRATION_REWARD_WEIGHT = 0.75
MIGRATION_SIZE_REF_MB = 100.0
MIGRATION_SIZE_ALPHA = 0.75
MIGRATION_SIZE_POWER = 3.0
MIGRATION_STATE_ALPHA = 0.5
NONLINEAR_MIGRATION_COST_CLIP_MS = 300000.0
# D1.5: cap SLA penalty before RL (pairs with REWARD_V2_OBJECTIVE_SCALE_MS)
MAX_SLA_PENALTY_MS = float(os.environ.get("MAX_SLA_PENALTY_MS", "500000.0"))
# --- Reward v2 (REWARD_SCHEME=v2): alpha*E^2 + lambda*base*exp(size/tau) ---
def _reward_scheme_env():
    return os.environ.get("REWARD_SCHEME", "v1").strip().lower()
REWARD_V2_SLA_ALPHA_MS_PER_KM2 = float(os.environ.get("REWARD_V2_SLA_ALPHA_MS_PER_KM2", "80"))
REWARD_V2_QOS_BETA_MS_PER_KM2 = float(os.environ.get("REWARD_V2_QOS_BETA_MS_PER_KM2", "80"))
REWARD_V2_MIGRATION_LAMBDA = float(os.environ.get("REWARD_V2_MIGRATION_LAMBDA", "0.75"))
REWARD_V2_MIGRATION_TAU_MB = float(os.environ.get("REWARD_V2_MIGRATION_TAU_MB", "100"))
REWARD_V2_EXP_EXPONENT_CLIP = float(os.environ.get("REWARD_V2_EXP_EXPONENT_CLIP", "8.0"))
# v2: S=1000 STAY collapse; S=10000 ok for v2.0; v2.1 needs larger S (see run_reward_v21_*).
REWARD_V2_OBJECTIVE_SCALE_MS = float(os.environ.get("REWARD_V2_OBJECTIVE_SCALE_MS", "50000"))
REWARD_RECOVERY_BONUS_MAX = 0.0
REWARD_DISTANCE_BONUS_WEIGHT = 0.0
# Backward-compatible export; reward no longer hard-clips to this value.
REWARD_CLIP_MIN = -float("inf")
MB_TO_MBIT = 8.0
BASE_MIGRATION_OVERHEAD_MS = 200.0
# Reactive 下迁移段额外系数（原 gamma 不对称语义的简化承接）
REACTIVE_MIGRATION_MULT = 1.5
# D0: cluster trace traffic -> per-taxi RPC volume (replaces norm_traffic)
EDGE_RPC_SCALING = float(os.environ.get("EDGE_RPC_SCALING", "0.01"))
MIN_RPC_CALLS = float(os.environ.get("MIN_RPC_CALLS", "1.0"))
# D1: v2 training P_SLA uses L_e2e = L_acc + L_internal for E_q (total_cost_ms unchanged)
def _env_flag(name, default="1"):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")

# 与 HYBRID_SAC_DEBUG_STEPS 对齐：前 N 次 reward 计算打印 total_cost 分解（排障迁移惩罚 vs SLA）。
_reward_dbg_remaining = int(os.environ.get("HYBRID_SAC_DEBUG_STEPS", "0") or "0")

# 与 context 单一真源（reward 内仍用此名便于阅读）
SLA_DISTANCE_THRESHOLD = DISTANCE_THRESHOLD_KM


def reward_scheme():
    """Active reward scheme name: ``v1`` or ``v2``."""
    scheme = _reward_scheme_env()
    return "v2" if scheme in ("v2", "2", "reward_v2") else "v1"


def is_reward_v2():
    return reward_scheme() == "v2"


def use_reward_v2_internal_path():
    """v2.1: topology-aware E_q in training objective (not in report sla_penalty line)."""
    return is_reward_v2() and _env_flag("REWARD_V2_USE_INTERNAL_PATH", "1")


def edge_actual_rpc_calls(traffic):
    """Physical RPC call count scale for an edge (replaces norm_traffic)."""
    return max(MIN_RPC_CALLS, float(traffic) * EDGE_RPC_SCALING)


def traffic_log_rpc_feature(traffic):
    """log1p scaled RPC count for GAT/DQN features (D2, aligned with reward)."""
    return float(np.log1p(edge_actual_rpc_calls(traffic)))


def dag_max_traffic_log_rpc(dag_info):
    """Max per-edge log-RPC in a DAG (for normalizing adjacency weights)."""
    edges = dag_info.get("edges") or {}
    if not edges:
        return 1.0
    return max(traffic_log_rpc_feature(t) for t in edges.values())


def edge_effective_latency_ms(edge_dist_km, traffic, cross_machine=True):
    """Per-edge RPC blocking time (ms); 0 when co-located."""
    if not cross_machine:
        return 0.0
    base_ms = (max(0.0, float(edge_dist_km)) / FIBER_SPEED_KM_MS) + BASE_ROUTER_DELAY_MS
    return base_ms * edge_actual_rpc_calls(traffic)


def compute_tearing_delay_ms(dag_info, assignments):
    """Cross-server tearing/backhaul cost (ms); unchanged from pre-D0 semantics."""
    edges_items = list((dag_info.get("edges") or {}).items())
    if not edges_items:
        return 0.0
    traffics = np.array([float(t) for (_, _), t in edges_items], dtype=np.float64)
    src_srv = np.array([assignments[s] for (s, d), _ in edges_items])
    dst_srv = np.array([assignments[d] for (s, d), _ in edges_items])
    same = src_srv == dst_srv
    cross_mb = np.minimum(traffics * RPC_SIZE_MB, MAX_TEARING_MB)
    valid = ~same
    return float(np.sum(valid * (cross_mb / EDGE_BACKHAUL_MBPS) * 1000.0))


def compute_internal_critical_path_ms(dag_info, assignments, servers_info):
    """
    DAG internal end-to-end latency (ms) via critical-path max accumulation.

    Cross-machine edges contribute base_edge_ms * actual_rpc_calls; co-located edges 0.
    """
    edges = dag_info.get("edges") or {}
    if not edges:
        return 0.0

    adj = defaultdict(list)
    for src, dst in edges:
        adj[src].append(dst)

    node_latency = {node: 0.0 for node in dag_info["nodes"]}
    for src in topological_sort(dag_info):
        if src not in node_latency:
            continue
        src_server = assignments[src]
        src_lat, src_lon = servers_info[src_server]
        base_from_src = node_latency[src]
        for dst in adj[src]:
            traffic = float(edges[(src, dst)])
            if assignments[src] == assignments[dst]:
                eff_ms = 0.0
            else:
                dst_lat, dst_lon = servers_info[assignments[dst]]
                edge_dist_km = float(
                    haversine_distance(src_lat, src_lon, dst_lat, dst_lon)
                )
                eff_ms = edge_effective_latency_ms(edge_dist_km, traffic, True)
            node_latency[dst] = max(node_latency[dst], base_from_src + eff_ms)

    deployable = get_deployable_nodes(dag_info)
    if deployable:
        return float(max(node_latency[node] for node in deployable))
    return float(max(node_latency.values()) if node_latency else 0.0)


def build_servers_info(servers_df):
    """Pre-build {server_id: (lat, lon)} lookup dict（列向量化，无 iterrows）。"""
    ar = servers_df[["edge_server_id", "latitude", "longitude"]].to_numpy(copy=False)
    return dict(
        zip(
            map(int, ar[:, 0]),
            zip(ar[:, 1].astype(np.float64), ar[:, 2].astype(np.float64)),
        )
    )


def _entry_access_profile(assignments, dag_info, user_lat, user_lon, servers_info):
    """Return (max_entry_dist_km, access_latency_ms) for a placement."""
    entry_nodes = get_service_entry_nodes(dag_info)
    if not entry_nodes:
        return 0.0, 0.0
    srv_lats = np.array([servers_info[assignments[node]][0] for node in entry_nodes], dtype=np.float64)
    srv_lons = np.array([servers_info[assignments[node]][1] for node in entry_nodes], dtype=np.float64)
    entry_distances_km = np.asarray(
        haversine_distance(user_lat, user_lon, srv_lats, srv_lons),
        dtype=np.float64,
    ).ravel()
    max_entry_dist_km = float(np.max(entry_distances_km)) if entry_distances_km.size else 0.0
    return max_entry_dist_km, float(calc_access_latency_ms(max_entry_dist_km))


def estimate_dag_migration_time_s(
    dag_info,
    gateway_dist_km=0.0,
    trigger_type=TRIGGER_PROACTIVE,
    nodes=None,
):
    """
    Conservative time estimate for proactive TTV gating.

    It assumes the selected DAG nodes may migrate and uses the same physical
    transfer model as ``calculate_microservice_reward``.
    """
    risk_ratio = (
        min(max(float(gateway_dist_km) / SLA_DISTANCE_THRESHOLD, 0.0), 1.0)
        if SLA_DISTANCE_THRESHOLD > 0 else 0.0
    )
    effective_bandwidth = MIN_BW_MBPS + (MAX_BW_MBPS - MIN_BW_MBPS) * (risk_ratio ** 2)
    node_names = list(nodes) if nodes is not None else get_deployable_nodes(dag_info)
    total_s = 0.0
    for node in node_names:
        props = dag_info["nodes"][node]
        mb = float(props["image_mb"]) + float(props["state_mb"])
        delta_s = (mb * MB_TO_MBIT / effective_bandwidth) + (BASE_MIGRATION_OVERHEAD_MS / 1000.0)
        if trigger_type == TRIGGER_REACTIVE:
            delta_s *= REACTIVE_MIGRATION_MULT
        total_s += delta_s
    return total_s


def _sla_excess_km(
    max_entry_dist_km,
    access_latency_ms=0.0,
    internal_path_ms=0.0,
    *,
    use_e2e_qos=False,
):
    distance_excess_km = max(0.0, float(max_entry_dist_km) - SLA_DISTANCE_THRESHOLD)
    qos_ref_ms = float(access_latency_ms)
    if use_e2e_qos:
        qos_ref_ms += max(0.0, float(internal_path_ms))
    qos_excess_ms = max(0.0, qos_ref_ms - USER_SLA_TOLERANCE_MS)
    qos_excess_km = qos_excess_ms * FIBER_SPEED_KM_MS
    return distance_excess_km, qos_excess_km


def calculate_sla_penalty_ms(
    max_entry_dist_km,
    access_latency_ms=0.0,
    internal_path_ms=0.0,
    *,
    use_e2e_qos=None,
):
    """SLA penalty for the active ``REWARD_SCHEME``."""
    if use_e2e_qos is None:
        use_e2e_qos = use_reward_v2_internal_path()
    distance_excess_km, qos_excess_km = _sla_excess_km(
        max_entry_dist_km,
        access_latency_ms,
        internal_path_ms,
        use_e2e_qos=use_e2e_qos,
    )
    if distance_excess_km <= 0.0 and qos_excess_km <= 0.0:
        return 0.0
    if is_reward_v2():
        penalty = (distance_excess_km ** 2) * REWARD_V2_SLA_ALPHA_MS_PER_KM2
        if qos_excess_km > 0.0:
            penalty += (qos_excess_km ** 2) * REWARD_V2_QOS_BETA_MS_PER_KM2
    else:
        equivalent_excess_km = distance_excess_km + qos_excess_km
        penalty = (
            SLA_BASE_PENALTY_MS
            + equivalent_excess_km * SLA_PENALTY_PER_KM_MS
            + (equivalent_excess_km ** 2) * SLA_QUADRATIC_PENALTY_PER_KM2_MS
        )
    return float(min(max(0.0, penalty), MAX_SLA_PENALTY_MS))


def sla_penalty_gain_ms(
    old_max_entry_dist_km,
    new_max_entry_dist_km,
    old_access_latency_ms=0.0,
    new_access_latency_ms=0.0,
    old_internal_path_ms=0.0,
    new_internal_path_ms=0.0,
    *,
    use_e2e_qos=None,
):
    """Non-negative SLA penalty reduction when placement improves SLA objective."""
    if use_e2e_qos is None:
        use_e2e_qos = use_reward_v2_internal_path()
    old_penalty = calculate_sla_penalty_ms(
        old_max_entry_dist_km,
        old_access_latency_ms,
        old_internal_path_ms,
        use_e2e_qos=use_e2e_qos,
    )
    new_penalty = calculate_sla_penalty_ms(
        new_max_entry_dist_km,
        new_access_latency_ms,
        new_internal_path_ms,
        use_e2e_qos=use_e2e_qos,
    )
    return float(max(0.0, old_penalty - new_penalty))


def future_mean_excess_penalty_ms(mean_excess_km):
    """SLA penalty for mean post-threshold entry distance excess (forecast horizon)."""
    excess = max(0.0, float(mean_excess_km))
    if excess <= 0.0:
        return 0.0
    if is_reward_v2():
        return (excess ** 2) * REWARD_V2_SLA_ALPHA_MS_PER_KM2
    return excess * SLA_PENALTY_PER_KM_MS


def future_mean_excess_penalty_gain_ms(old_mean_excess_km, new_mean_excess_km):
    """Non-negative SLA penalty reduction when forecast mean excess improves."""
    return float(
        max(
            0.0,
            future_mean_excess_penalty_ms(old_mean_excess_km)
            - future_mean_excess_penalty_ms(new_mean_excess_km),
        )
    )


def migration_size_state_multiplier(image_mb, state_mb):
    """Continuous size/state penalty that replaces hard heavyweight guards."""
    transfer_mb = max(0.0, float(image_mb) + float(state_mb))
    state_mb = max(0.0, float(state_mb))
    size_ratio = transfer_mb / max(MIGRATION_SIZE_REF_MB, 1e-6)
    state_ratio = state_mb / max(MIGRATION_SIZE_REF_MB, 1e-6)
    return float(
        1.0
        + MIGRATION_SIZE_ALPHA * (size_ratio ** MIGRATION_SIZE_POWER)
        + MIGRATION_STATE_ALPHA * np.log1p(state_ratio)
    )


def _exp_migration_size_multiplier(image_mb, state_mb):
    transfer_mb = max(0.0, float(image_mb) + float(state_mb))
    tau = max(REWARD_V2_MIGRATION_TAU_MB, 1e-6)
    exponent = min(transfer_mb / tau, max(0.0, REWARD_V2_EXP_EXPONENT_CLIP))
    return float(math.exp(exponent))


def calculate_nonlinear_migration_cost_ms(raw_migration_ms, image_mb, state_mb):
    """Nonlinear migration cost for the active ``REWARD_SCHEME``."""
    base_ms = max(0.0, float(raw_migration_ms))
    if is_reward_v2():
        nonlinear_cost = (
            REWARD_V2_MIGRATION_LAMBDA
            * base_ms
            * _exp_migration_size_multiplier(image_mb, state_mb)
        )
    else:
        nonlinear_cost = base_ms * migration_size_state_multiplier(image_mb, state_mb)
    return float(min(nonlinear_cost, NONLINEAR_MIGRATION_COST_CLIP_MS))


def calculate_entry_sla_metrics(entry_nodes, assignments, user_lat, user_lon, servers_info):
    """Return max-entry SLA risk and excess-distance diagnostics for a placement."""
    if not entry_nodes:
        return {
            "primary_entry_violation": 0,
            "max_entry_violation": 0,
            "max_entry_distance_km": 0.0,
            "sla_excess_distance_km": 0.0,
            "severe_sla_violation": 0,
        }

    distances = []
    violations = []
    for node in entry_nodes:
        srv_lat, srv_lon = servers_info[assignments[node]]
        dist = float(haversine_distance(user_lat, user_lon, srv_lat, srv_lon))
        distances.append(dist)
        violations.append(
            bool(
                dist > SLA_DISTANCE_THRESHOLD
                or calc_access_latency_ms(dist) > USER_SLA_TOLERANCE_MS
            )
        )
    max_dist = max(distances) if distances else 0.0
    excess = max(0.0, max_dist - SLA_DISTANCE_THRESHOLD)
    return {
        "primary_entry_violation": int(violations[0]) if violations else 0,
        "max_entry_violation": int(any(violations)),
        "max_entry_distance_km": float(max_dist),
        "sla_excess_distance_km": float(excess),
        "severe_sla_violation": int(excess > SEVERE_SLA_EXCESS_KM),
    }


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
    """Returns (reward, details). details["total_cost_ms"] 保留真实物理 ms。"""
    user_lat, user_lon = user_location
    entry_nodes = get_service_entry_nodes(dag_info)

    max_entry_dist_km, access_latency_ms = _entry_access_profile(
        current_assignments, dag_info, user_lat, user_lon, servers_info
    )
    prev_max_entry_dist_km, prev_access_latency_ms = _entry_access_profile(
        previous_assignments, dag_info, user_lat, user_lon, servers_info
    )
    if entry_nodes:
        srv_lats = np.array([servers_info[current_assignments[node]][0] for node in entry_nodes], dtype=np.float64)
        srv_lons = np.array([servers_info[current_assignments[node]][1] for node in entry_nodes], dtype=np.float64)
        entry_distances_km = np.asarray(
            haversine_distance(user_lat, user_lon, srv_lats, srv_lons),
            dtype=np.float64,
        ).ravel()
    else:
        entry_distances_km = np.zeros(0, dtype=np.float64)

    risk_ratio = min(max_entry_dist_km / SLA_DISTANCE_THRESHOLD, 1.0) if SLA_DISTANCE_THRESHOLD > 0 else 0.0
    effective_bandwidth = MIN_BW_MBPS + (MAX_BW_MBPS - MIN_BW_MBPS) * (risk_ratio ** 2)

    migrating_targets = {}
    for node in get_deployable_nodes(dag_info):
        if current_assignments[node] != previous_assignments[node]:
            target = current_assignments[node]
            migrating_targets[target] = migrating_targets.get(target, 0) + 1

    migration_delay_ms = 0.0
    nonlinear_migration_cost_ms = 0.0
    for node in get_deployable_nodes(dag_info):
        node_props = dag_info["nodes"][node]
        if current_assignments[node] != previous_assignments[node]:
            image_mb = float(node_props["image_mb"])
            state_mb = float(node_props["state_mb"])
            target_server = current_assignments[node]
            target_concurrency = max(1, int(migrating_targets.get(target_server, 1)))
            node_bandwidth = max(effective_bandwidth / target_concurrency, 1e-6)
            # C1：字节→比特 ×8；带宽按 Mbps；每迁移节点加容器启动底噪（ms）
            delta_ms = (
                ((image_mb + state_mb) * MB_TO_MBIT / node_bandwidth) * 1000.0
                + BASE_MIGRATION_OVERHEAD_MS
            )
            if trigger_type == TRIGGER_REACTIVE:
                delta_ms *= REACTIVE_MIGRATION_MULT
            migration_delay_ms += delta_ms
            nonlinear_migration_cost_ms += calculate_nonlinear_migration_cost_ms(
                delta_ms,
                image_mb,
                state_mb,
            )

    # D0: comm = internal critical path; tearing isolated (no norm_traffic sum)
    tearing_delay_ms = compute_tearing_delay_ms(dag_info, current_assignments)
    internal_critical_path_ms = compute_internal_critical_path_ms(
        dag_info, current_assignments, servers_info
    )
    comm_delay_ms = internal_critical_path_ms

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
    l_e2e_ms = access_latency_ms + internal_critical_path_ms
    if use_reward_v2_internal_path():
        qos_violation = l_e2e_ms > USER_SLA_TOLERANCE_MS
    else:
        qos_violation = access_latency_ms > USER_SLA_TOLERANCE_MS

    # Report / SA total_cost: physical sla line (access-only E_q); L_internal already in comm.
    sla_penalty_ms = calculate_sla_penalty_ms(
        max_entry_dist_km,
        access_latency_ms,
        internal_path_ms=0.0,
        use_e2e_qos=False,
    )
    sla_penalty_objective_ms = calculate_sla_penalty_ms(
        max_entry_dist_km,
        access_latency_ms,
        internal_path_ms=internal_critical_path_ms,
        use_e2e_qos=use_reward_v2_internal_path(),
    )

    total_cost_ms = (
        access_latency_ms
        + migration_delay_ms
        + tearing_delay_ms
        + comm_delay_ms
        + future_delay_ms
        + sla_penalty_ms
    )
    if is_reward_v2():
        reward_objective_ms = sla_penalty_objective_ms + nonlinear_migration_cost_ms
    else:
        reward_objective_ms = (
            sla_penalty_objective_ms
            + CORE_MIGRATION_REWARD_WEIGHT * nonlinear_migration_cost_ms
        )

    prev_l_e2e_ms = prev_access_latency_ms
    if use_reward_v2_internal_path():
        prev_internal = compute_internal_critical_path_ms(
            dag_info, previous_assignments, servers_info
        )
        prev_l_e2e_ms = prev_access_latency_ms + prev_internal

    prev_violation = (
        prev_max_entry_dist_km > SLA_DISTANCE_THRESHOLD
        or prev_l_e2e_ms > USER_SLA_TOLERANCE_MS
    )
    resolved_violation = prev_violation and not (spatial_violation or qos_violation)
    dist_improvement = max(0.0, prev_max_entry_dist_km - max_entry_dist_km)
    distance_bonus = min(
        REWARD_DISTANCE_BONUS_WEIGHT * (dist_improvement / SLA_DISTANCE_THRESHOLD),
        REWARD_RECOVERY_BONUS_MAX,
    ) if SLA_DISTANCE_THRESHOLD > 0 else 0.0
    recovery_bonus = 1.0 if resolved_violation else 0.0
    reward_bonus = min(distance_bonus + recovery_bonus, REWARD_RECOVERY_BONUS_MAX)

    if is_reward_v2():
        scale = max(REWARD_V2_OBJECTIVE_SCALE_MS, 1e-6)
        reward = -float(max(reward_objective_ms, 0.0) / scale) + reward_bonus
    else:
        reward = (
            -float(np.log1p(max(reward_objective_ms, 0.0) / REWARD_COST_SCALE_MS))
            + reward_bonus
        )

    global _reward_dbg_remaining
    if _reward_dbg_remaining > 0:
        _reward_dbg_remaining -= 1
        print(
            "[HYBRID_SAC_DBG reward] total_cost_ms=",
            total_cost_ms,
            " migration_delay_ms=",
            migration_delay_ms,
            " nonlinear_migration_cost_ms=",
            nonlinear_migration_cost_ms,
            " tearing_delay_ms=",
            tearing_delay_ms,
            " internal_critical_path_ms=",
            internal_critical_path_ms,
            " sla_penalty_ms=",
            sla_penalty_ms,
            " reward_objective_ms=",
            reward_objective_ms,
            " access_latency_ms=",
            access_latency_ms,
            " reward=",
            reward,
            " reward_bonus=",
            reward_bonus,
            sep="",
        )

    details = {
        "access_latency": access_latency_ms,
        "communication_cost": comm_delay_ms,
        "internal_critical_path_ms": internal_critical_path_ms,
        "migration_cost": migration_delay_ms,
        "nonlinear_migration_cost_ms": nonlinear_migration_cost_ms,
        "future_penalty": future_delay_ms,
        "tearing_penalty": tearing_delay_ms,
        "sla_violations": sla_violations,
        "risk_ratio": risk_ratio,
        "state_divisor": effective_bandwidth,
        "total_cost": total_cost_ms,
        "reward_objective_ms": reward_objective_ms,
        "reward": reward,
        "trigger_type": trigger_type,
        "sla_penalty_ms": sla_penalty_ms,
        "sla_penalty_objective_ms": sla_penalty_objective_ms,
        "l_e2e_ms": l_e2e_ms,
        "access_latency_ms": access_latency_ms,
        "max_entry_distance_km": max_entry_dist_km,
        "sla_excess_distance_km": max(0.0, max_entry_dist_km - SLA_DISTANCE_THRESHOLD),
        "total_cost_ms": total_cost_ms,
        "tearing_penalty_ms": tearing_delay_ms,
        "future_penalty_ms": future_delay_ms,
        "reward_bonus": reward_bonus,
        "prev_access_latency_ms": prev_access_latency_ms,
        "prev_max_entry_dist_km": prev_max_entry_dist_km,
        "reward_scheme": reward_scheme(),
    }
    return reward, details
