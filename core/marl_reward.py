"""Reward decomposition for CTDE-GAT-MARL microservice migration."""

import os

import numpy as np

from core.dag_utils import get_deployable_nodes, get_service_entry_nodes, is_external_node
from core.geo import haversine_distance
from core.physics_utils import BASE_ROUTER_DELAY_MS, FIBER_SPEED_KM_MS
from core.reward import (
    BASE_MIGRATION_OVERHEAD_MS,
    EDGE_BACKHAUL_MBPS,
    MAX_TEARING_MB,
    MAX_BW_MBPS,
    MB_TO_MBIT,
    MIN_BW_MBPS,
    REACTIVE_MIGRATION_MULT,
    REWARD_V2_OBJECTIVE_SCALE_MS,
    effective_reward_v2_objective_scale_ms,
    RPC_SIZE_MB,
    SLA_DISTANCE_THRESHOLD,
    calculate_nonlinear_migration_cost_ms,
    calculate_microservice_reward,
    edge_effective_latency_ms,
    is_reward_v2,
)
from core.context import TRIGGER_PROACTIVE, TRIGGER_REACTIVE

# Fixed divisor for RL value/advantage scale (~7–8k ms total_cost → ~−0.7..−0.8).
# Linear in total_cost_ms; not a tunable curriculum knob.
TOTAL_COST_TRAIN_SCALE_MS = 10000.0


def _env_flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


def train_total_cost_enabled():
    """Direction A: critic/actor optimize −total_cost_ms (report KPI), not v2 scaled reward."""
    return _env_flag("MARL_TRAIN_TOTAL_COST", "0")


def total_cost_training_signal(total_cost_ms):
    """RL return aligned with SA search objective (minimize total_cost_ms)."""
    scale = max(float(TOTAL_COST_TRAIN_SCALE_MS), 1e-6)
    return -float(total_cost_ms) / scale


def lambda_schedule(step, *, warmup_steps=1000, ramp_steps=4000,
                    max_migration=0.15, max_split=0.05):
    """Warmup-to-annealing schedule for local MARL penalties."""
    step = max(0, int(step))
    if step < warmup_steps:
        return 0.0, 0.0
    progress = min(1.0, (step - warmup_steps) / max(1.0, float(ramp_steps)))
    return max_migration * progress, max_split * progress


def lambda_schedule_by_epoch(epoch, num_epochs, *, max_migration=0.15, max_split=0.05):
    """
    Epoch-level warmup schedule for local penalties.

    Epoch 0 starts at 10% of max penalties to avoid free-migration habits while
    still allowing early exploration.  The last training epoch reaches max values.
    """
    epoch = max(0, int(epoch))
    train_epochs = max(1, int(num_epochs) - 1)
    if epoch <= 0:
        return 0.1 * max_migration, 0.1 * max_split
    progress = min(1.0, epoch / float(max(1, train_epochs - 1)))
    return max_migration * progress, max_split * progress


def _effective_bandwidth(assignments, dag_info, user_location, servers_info):
    entry_nodes = get_service_entry_nodes(dag_info)
    if not entry_nodes:
        return MIN_BW_MBPS
    user_lat, user_lon = user_location
    dists = [
        haversine_distance(
            user_lat,
            user_lon,
            servers_info[assignments[node]][0],
            servers_info[assignments[node]][1],
        )
        for node in entry_nodes
    ]
    max_entry_dist_km = float(max(dists)) if dists else 0.0
    risk_ratio = (
        min(max_entry_dist_km / SLA_DISTANCE_THRESHOLD, 1.0)
        if SLA_DISTANCE_THRESHOLD > 0 else 0.0
    )
    return MIN_BW_MBPS + (MAX_BW_MBPS - MIN_BW_MBPS) * (risk_ratio ** 2)


def _local_migration_costs(dag_info, current_assignments, previous_assignments,
                           user_location, servers_info, trigger_type):
    bandwidth = _effective_bandwidth(current_assignments, dag_info, user_location, servers_info)
    deployable = set(get_deployable_nodes(dag_info))
    migrating_targets = {}
    for node in deployable:
        if current_assignments[node] != previous_assignments[node]:
            target = current_assignments[node]
            migrating_targets[target] = migrating_targets.get(target, 0) + 1

    costs = {node: 0.0 for node in dag_info["nodes"]}
    for node in deployable:
        props = dag_info["nodes"][node]
        if current_assignments[node] == previous_assignments[node]:
            continue
        mb = float(props["image_mb"]) + float(props["state_mb"])
        target_server = current_assignments[node]
        target_concurrency = max(1, int(migrating_targets.get(target_server, 1)))
        node_bandwidth = max(bandwidth / target_concurrency, 1e-6)
        raw_cost_ms = ((mb * MB_TO_MBIT / node_bandwidth) * 1000.0) + BASE_MIGRATION_OVERHEAD_MS
        if trigger_type == TRIGGER_REACTIVE:
            raw_cost_ms *= REACTIVE_MIGRATION_MULT
        costs[node] = float(
            calculate_nonlinear_migration_cost_ms(
                raw_cost_ms,
                props.get("image_mb", 0.0),
                props.get("state_mb", 0.0),
            )
        )
    return costs


def _edge_split_cost(src, dst, traffic, assignments, servers_info):
    if assignments[src] == assignments[dst]:
        return 0.0
    src_lat, src_lon = servers_info[assignments[src]]
    dst_lat, dst_lon = servers_info[assignments[dst]]
    edge_dist_km = float(haversine_distance(src_lat, src_lon, dst_lat, dst_lon))
    cross_mb = min(float(traffic) * RPC_SIZE_MB, MAX_TEARING_MB)
    tearing_ms = (cross_mb / EDGE_BACKHAUL_MBPS) * 1000.0
    comm_ms = edge_effective_latency_ms(edge_dist_km, traffic, True)
    return float(tearing_ms + comm_ms)


def _local_edge_split_costs(dag_info, current_assignments, previous_assignments, servers_info):
    """Incremental incident-edge split cost caused by the current joint migration."""
    costs = {node: 0.0 for node in dag_info["nodes"]}
    deployable = set(get_deployable_nodes(dag_info))
    if not dag_info.get("edges"):
        return costs

    for (src, dst), traffic in dag_info["edges"].items():
        if (
            current_assignments[src] == previous_assignments[src]
            and current_assignments[dst] == previous_assignments[dst]
        ):
            continue
        previous_cost = _edge_split_cost(src, dst, traffic, previous_assignments, servers_info)
        current_cost = _edge_split_cost(src, dst, traffic, current_assignments, servers_info)
        incremental_cost = max(0.0, current_cost - previous_cost)
        if incremental_cost <= 0.0:
            continue
        deployable_endpoints = [node for node in (src, dst) if node in deployable]
        if not deployable_endpoints:
            continue
        share = incremental_cost / float(len(deployable_endpoints))
        for node in deployable_endpoints:
            costs[node] += share
    return costs


def _dense_distance_bonuses(
    dag_info,
    current_assignments,
    previous_assignments,
    user_location,
    servers_info,
    trigger_type,
    *,
    proactive_bonus_per_km=0.4,
    proactive_bonus_max=6.0,
):
    """
    Reward-scale dense bonus for moving closer to the user.

    The shared reward is log-scaled, so this bonus must stay in single-digit
    reward units.  Action-time logit bias handles exploration; this post-action
    bonus reinforces successful proactive moves without dominating training.
    """
    user_lat, user_lon = user_location
    bonuses = {}
    
    for node in dag_info["nodes"]:
        if is_external_node(node):
            bonuses[node] = 0.0
            continue
        
        old_server = previous_assignments[node]
        new_server = current_assignments[node]
        
        # No migration = no bonus
        if old_server == new_server:
            bonuses[node] = 0.0
            continue
        
        # Calculate distance to user before and after migration
        old_lat, old_lon = servers_info[old_server]
        new_lat, new_lon = servers_info[new_server]
        old_dist_km = float(haversine_distance(user_lat, user_lon, old_lat, old_lon))
        new_dist_km = float(haversine_distance(user_lat, user_lon, new_lat, new_lon))
        
        # Only reward if actually getting closer
        distance_reduction_km = max(0.0, old_dist_km - new_dist_km)
        if distance_reduction_km <= 1e-6:
            bonuses[node] = 0.0
            continue
        
        # Get risk_ratio from effective bandwidth calculation
        # This reflects how close the entry nodes are to SLA violation threshold
        risk_ratio = 0.0
        entry_nodes = get_service_entry_nodes(dag_info)
        if entry_nodes:
            entry_dists = [
                haversine_distance(
                    user_lat, user_lon,
                    servers_info[current_assignments[n]][0],
                    servers_info[current_assignments[n]][1],
                )
                for n in entry_nodes
            ]
            max_entry_dist_km = float(max(entry_dists)) if entry_dists else 0.0
            risk_ratio = min(max_entry_dist_km / SLA_DISTANCE_THRESHOLD, 1.0) if SLA_DISTANCE_THRESHOLD > 0 else 0.0
        
        if trigger_type == TRIGGER_PROACTIVE:
            risk_factor = 1.0 + risk_ratio
            bonus_value = min(
                proactive_bonus_max,
                distance_reduction_km * proactive_bonus_per_km * risk_factor,
            )
        else:
            bonus_value = 0.0
            
        bonuses[node] = float(bonus_value)
    
    return bonuses


def _entry_sla_bonuses(
    dag_info,
    current_assignments,
    previous_assignments,
    user_location,
    servers_info,
    *,
    max_bonus=3.0,
):
    """Reward-scale bonus for actions that improve the entry-node SLA bottleneck."""
    bonuses = {node: 0.0 for node in dag_info["nodes"]}
    entry_nodes = get_service_entry_nodes(dag_info)
    if not entry_nodes:
        return bonuses

    user_lat, user_lon = user_location
    prev_dists = {
        node: float(haversine_distance(
            user_lat,
            user_lon,
            servers_info[previous_assignments[node]][0],
            servers_info[previous_assignments[node]][1],
        ))
        for node in entry_nodes
    }
    curr_dists = {
        node: float(haversine_distance(
            user_lat,
            user_lon,
            servers_info[current_assignments[node]][0],
            servers_info[current_assignments[node]][1],
        ))
        for node in entry_nodes
    }
    prev_max = max(prev_dists.values()) if prev_dists else 0.0
    curr_max = max(curr_dists.values()) if curr_dists else 0.0
    if curr_max >= prev_max:
        return bonuses

    prev_excess = max(0.0, prev_max - SLA_DISTANCE_THRESHOLD)
    curr_excess = max(0.0, curr_max - SLA_DISTANCE_THRESHOLD)
    improvement_ratio = max(0.0, prev_excess - curr_excess) / max(SLA_DISTANCE_THRESHOLD, 1e-6)
    if prev_max > SLA_DISTANCE_THRESHOLD and curr_max <= SLA_DISTANCE_THRESHOLD:
        improvement_ratio += 1.0
    bonus_value = min(max_bonus, max_bonus * improvement_ratio)
    if bonus_value <= 0.0:
        return bonuses

    improved_entries = [
        node for node in entry_nodes
        if current_assignments[node] != previous_assignments[node]
        and curr_dists[node] < prev_dists[node]
    ]
    if not improved_entries:
        return bonuses
    share = bonus_value / float(len(improved_entries))
    for node in improved_entries:
        bonuses[node] = float(share)
    return bonuses


def calculate_marl_rewards(
    taxi_id,
    dag_info,
    current_assignments,
    previous_assignments,
    user_location,
    servers_info,
    *,
    predicted_locations=None,
    trigger_type=TRIGGER_REACTIVE,
    lambda_migration=0.0,
    lambda_split=0.0,
    local_cost_scale_ms=1000.0,
    dense_distance_bonus=False,
):
    """
    Return shared DAG reward plus per-agent rewards and decomposition details.

    The shared reward is exactly the existing reward function's scalar reward.
    Local penalties are normalized by ``local_cost_scale_ms`` so their weights
    are comparable to the log-scaled shared reward.

    When ``MARL_TRAIN_TOTAL_COST=1`` (Direction A), ``training_reward`` and
    per-agent returns are ``−total_cost_ms / TOTAL_COST_TRAIN_SCALE_MS`` with
    no extra λ penalties — same physical cost as SA reports.
    """
    shared_reward, details = calculate_microservice_reward(
        taxi_id,
        dag_info,
        current_assignments,
        previous_assignments,
        user_location,
        servers_info,
        predicted_locations=predicted_locations,
        trigger_type=trigger_type,
    )
    effective_local_cost_scale_ms = (
        effective_reward_v2_objective_scale_ms() if is_reward_v2() else local_cost_scale_ms
    )
    migration_costs = _local_migration_costs(
        dag_info,
        current_assignments,
        previous_assignments,
        user_location,
        servers_info,
        trigger_type,
    )
    split_costs = _local_edge_split_costs(
        dag_info,
        current_assignments,
        previous_assignments,
        servers_info,
    )
    total_cost_ms = float(details["total_cost_ms"])
    use_total_cost_train = train_total_cost_enabled()
    distance_bonuses = {node: 0.0 for node in dag_info["nodes"]}
    if dense_distance_bonus:
        distance_bonuses = _dense_distance_bonuses(
            dag_info,
            current_assignments,
            previous_assignments,
            user_location,
            servers_info,
            trigger_type,
        )
    # Entry/SLA preference is now represented by the shared SLA penalty rather
    # than a hand-written local bonus.
    entry_sla_bonuses = {node: 0.0 for node in dag_info["nodes"]}

    if use_total_cost_train:
        train_signal = total_cost_training_signal(total_cost_ms)
        agent_rewards = {node: float(train_signal) for node in dag_info["nodes"]}
        training_reward = float(train_signal)
    else:
        agent_rewards = {}
        for node in dag_info["nodes"]:
            if is_external_node(node):
                agent_rewards[node] = float(shared_reward)
                continue
            local_penalty = (
                lambda_migration * (migration_costs[node] / effective_local_cost_scale_ms)
            )
            agent_rewards[node] = float(
                shared_reward
                + distance_bonuses[node]
                + entry_sla_bonuses[node]
                - local_penalty
            )
        training_reward = (
            float(np.mean(list(agent_rewards.values()))) if agent_rewards else float(shared_reward)
        )

    details = dict(details)
    details.update(
        {
            "shared_reward": float(shared_reward),
            "agent_rewards": agent_rewards,
            "local_migration_costs": migration_costs,
            "local_edge_split_costs": split_costs,
            "dense_distance_bonuses": distance_bonuses,
            "entry_sla_bonuses": entry_sla_bonuses,
            "local_migration_cost_sum": float(np.sum(list(migration_costs.values()))),
            "edge_split_cost_sum": float(np.sum(list(split_costs.values()))),
            "dense_distance_bonus_sum": float(np.sum(list(distance_bonuses.values()))),
            "entry_sla_bonus_sum": float(np.sum(list(entry_sla_bonuses.values()))),
            "lambda_migration": float(lambda_migration),
            "lambda_split": float(lambda_split),
            "training_reward": training_reward,
            "train_total_cost_mode": bool(use_total_cost_train),
            "train_total_cost_scale_ms": float(TOTAL_COST_TRAIN_SCALE_MS),
        }
    )
    return float(shared_reward), agent_rewards, details
