"""Reward decomposition for CTDE-GAT-MARL microservice migration."""

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
    RPC_SIZE_MB,
    SLA_DISTANCE_THRESHOLD,
    calculate_microservice_reward,
)
from core.context import TRIGGER_PROACTIVE, TRIGGER_REACTIVE


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

    Epoch 0 uses free migration/splitting to avoid early STAY collapse.  The last
    training epoch reaches the configured max values.  Evaluation/inference can
    pass the final epoch index if the metric should report final weights.
    """
    epoch = max(0, int(epoch))
    train_epochs = max(1, int(num_epochs) - 1)
    if epoch <= 0:
        return 0.0, 0.0
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
        cost_ms = ((mb * MB_TO_MBIT / node_bandwidth) * 1000.0) + BASE_MIGRATION_OVERHEAD_MS
        if trigger_type == TRIGGER_REACTIVE:
            cost_ms *= REACTIVE_MIGRATION_MULT
        costs[node] = float(cost_ms)
    return costs


def _edge_split_cost(src, dst, traffic, assignments, servers_info, max_traffic):
    if assignments[src] == assignments[dst]:
        return 0.0
    src_lat, src_lon = servers_info[assignments[src]]
    dst_lat, dst_lon = servers_info[assignments[dst]]
    edge_dist_km = float(haversine_distance(src_lat, src_lon, dst_lat, dst_lon))
    norm_traffic = float(traffic) / float(max_traffic)
    cross_mb = min(float(traffic) * RPC_SIZE_MB, MAX_TEARING_MB)
    tearing_ms = (cross_mb / EDGE_BACKHAUL_MBPS) * 1000.0
    comm_ms = norm_traffic * ((max(0.0, edge_dist_km) / FIBER_SPEED_KM_MS) + BASE_ROUTER_DELAY_MS)
    return float(tearing_ms + comm_ms)


def _local_edge_split_costs(dag_info, current_assignments, previous_assignments, servers_info):
    """Incremental incident-edge split cost caused by the current joint migration."""
    costs = {node: 0.0 for node in dag_info["nodes"]}
    deployable = set(get_deployable_nodes(dag_info))
    max_traffic = max(dag_info["edges"].values()) if dag_info["edges"] else 0.0
    if max_traffic <= 0:
        return costs

    for (src, dst), traffic in dag_info["edges"].items():
        if (
            current_assignments[src] == previous_assignments[src]
            and current_assignments[dst] == previous_assignments[dst]
        ):
            continue
        previous_cost = _edge_split_cost(src, dst, traffic, previous_assignments, servers_info, max_traffic)
        current_cost = _edge_split_cost(src, dst, traffic, current_assignments, servers_info, max_traffic)
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
    dense_distance_bonus=True,
):
    """
    Return shared DAG reward plus per-agent rewards and decomposition details.

    The shared reward is exactly the existing reward function's scalar reward.
    Local penalties are normalized by ``local_cost_scale_ms`` so their weights
    are comparable to the log-scaled shared reward.
    
    Distance bonus is reward-scale; proactive exploration is handled by actor
    logit bias before action selection.
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
    distance_bonuses = (
        _dense_distance_bonuses(
            dag_info,
            current_assignments,
            previous_assignments,
            user_location,
            servers_info,
            trigger_type,
        )
        if dense_distance_bonus else {node: 0.0 for node in dag_info["nodes"]}
    )

    agent_rewards = {}
    for node in dag_info["nodes"]:
        if is_external_node(node):
            agent_rewards[node] = float(shared_reward)
            continue
        local_penalty = (
            lambda_migration * (migration_costs[node] / local_cost_scale_ms)
            + lambda_split * (split_costs[node] / local_cost_scale_ms)
        )
        agent_rewards[node] = float(shared_reward + distance_bonuses[node] - local_penalty)

    details = dict(details)
    details.update(
        {
            "shared_reward": float(shared_reward),
            "agent_rewards": agent_rewards,
            "local_migration_costs": migration_costs,
            "local_edge_split_costs": split_costs,
            "dense_distance_bonuses": distance_bonuses,
            "local_migration_cost_sum": float(np.sum(list(migration_costs.values()))),
            "edge_split_cost_sum": float(np.sum(list(split_costs.values()))),
            "dense_distance_bonus_sum": float(np.sum(list(distance_bonuses.values()))),
            "lambda_migration": float(lambda_migration),
            "lambda_split": float(lambda_split),
            "training_reward": float(np.mean(list(agent_rewards.values()))) if agent_rewards else float(shared_reward),
        }
    )
    return float(shared_reward), agent_rewards, details
