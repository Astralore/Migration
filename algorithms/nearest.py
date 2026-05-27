"""Nearest-candidate heuristic baseline for microservice DAG migration."""

import copy
import time
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from core.context import TRIGGER_PROACTIVE, check_sla_violation, get_trigger_type
from core.dag_utils import (
    assign_dag_type,
    get_deployable_nodes,
    get_service_entry_nodes,
    initialize_dag_assignment,
)
from core.geo import find_k_nearest_servers, haversine_distance
from core.microservice_dags import MICROSERVICE_DAGS
from core.reward import (
    build_servers_info,
    calculate_entry_sla_metrics,
    calculate_microservice_reward,
    estimate_dag_migration_time_s,
)
from prediction.simple_predictor import build_predict_future_time_kwargs, touch_taxi_last


FORECAST_HORIZON = 15


def _entry_violation_counts(entry_nodes, assignments, user_lat, user_lon, servers_info):
    if not entry_nodes:
        return 0, 0
    primary_server = assignments[entry_nodes[0]]
    primary_lat, primary_lon = servers_info[primary_server]
    primary = int(check_sla_violation(user_lat, user_lon, primary_lat, primary_lon))
    max_entry = 0
    for node in entry_nodes:
        srv_lat, srv_lon = servers_info[assignments[node]]
        if check_sla_violation(user_lat, user_lon, srv_lat, srv_lon):
            max_entry = 1
            break
    return primary, max_entry


def run_nearest_microservice_fair(
    df,
    servers_df,
    predictor=None,
    proactive=False,
    collect_dag_proactive_stats=False,
):
    """On every trigger, co-locate all DAG nodes on the nearest candidate server."""
    servers_info = build_servers_info(servers_df)
    use_proactive = proactive and predictor is not None
    taxi_dag_type = {}
    taxi_dag_assignments = {}
    taxi_last = {}

    total_migrations = 0
    total_violations = 0
    primary_entry_violations = 0
    max_entry_violations = 0
    severe_sla_violations = 0
    total_sla_excess_distance_km = 0.0
    sla_excess_distance_history = []
    proactive_decisions = 0
    decision_count = 0
    total_reward_sum = 0.0
    reward_history = []
    total_access_latency = 0.0
    total_communication_cost = 0.0
    total_internal_path_ms = 0.0
    total_migration_cost = 0.0
    migration_decision_count = 0
    total_cost_ms_sum = 0.0
    total_sla_penalty_ms = 0.0
    total_tearing_penalty_ms = 0.0
    total_future_penalty_ms = 0.0
    total_decision_time = 0.0
    dag_stats = (
        defaultdict(lambda: {"proactive_decisions": 0, "migrated_nodes": 0})
        if collect_dag_proactive_stats else None
    )

    timestamps = sorted(df["date_time"].unique())
    df_grouped = df.groupby("date_time")
    pbar = tqdm(total=len(timestamps), desc="Nearest Microservice Migration")
    for timestamp in timestamps:
        current_rows = df_grouped.get_group(timestamp)
        for _, row in current_rows.iterrows():
            taxi_id = row["taxi_id"]
            current_lat = row["latitude"]
            current_lon = row["longitude"]
            ts = pd.Timestamp(timestamp)
            pf_kw = build_predict_future_time_kwargs(
                taxi_last, taxi_id, row, current_lon, current_lat, ts
            )

            if taxi_id not in taxi_dag_assignments:
                nearest = find_k_nearest_servers(current_lat, current_lon, servers_df, k=1)[0]
                dag_type = assign_dag_type()
                taxi_dag_type[taxi_id] = dag_type
                taxi_dag_assignments[taxi_id] = initialize_dag_assignment(dag_type, nearest[0])
                touch_taxi_last(taxi_last, taxi_id, row, current_lon, current_lat, ts)
                continue

            dag_type = taxi_dag_type[taxi_id]
            dag_info = MICROSERVICE_DAGS[dag_type]
            entry_nodes = get_service_entry_nodes(dag_info)
            if not entry_nodes:
                touch_taxi_last(taxi_last, taxi_id, row, current_lon, current_lat, ts)
                continue
            gateway = entry_nodes[0]
            gateway_server = taxi_dag_assignments[taxi_id][gateway]
            gw_lat, gw_lon = servers_info[gateway_server]
            gateway_dist = haversine_distance(current_lat, current_lon, gw_lat, gw_lon)

            sla_metrics = calculate_entry_sla_metrics(
                entry_nodes, taxi_dag_assignments[taxi_id], current_lat, current_lon, servers_info
            )
            primary_v = sla_metrics["primary_entry_violation"]
            max_v = sla_metrics["max_entry_violation"]
            primary_entry_violations += primary_v
            max_entry_violations += max_v
            total_violations += max_v
            severe_sla_violations += sla_metrics["severe_sla_violation"]
            total_sla_excess_distance_km += sla_metrics["sla_excess_distance_km"]
            sla_excess_distance_history.append(sla_metrics["sla_excess_distance_km"])

            predicted_locations = None
            if use_proactive:
                raw = predictor.predict_future(
                    current_lon, current_lat, taxi_id, steps=FORECAST_HORIZON, **pf_kw
                )
                predicted_locations = [(lat, lon) for lon, lat in raw]

            trigger_type = get_trigger_type(
                current_lat,
                current_lon,
                gw_lat,
                gw_lon,
                predicted_locations=predicted_locations,
                proactive_enabled=use_proactive,
                estimated_migration_time_s=estimate_dag_migration_time_s(
                    dag_info, gateway_dist_km=gateway_dist, trigger_type=TRIGGER_PROACTIVE
                ),
                forecast_step_dt_sec=pf_kw.get("forecast_step_dt_sec"),
            )
            if trigger_type is None:
                touch_taxi_last(taxi_last, taxi_id, row, current_lon, current_lat, ts)
                continue

            decision_count += 1
            if trigger_type == TRIGGER_PROACTIVE:
                proactive_decisions += 1

            old_assignments = copy.copy(taxi_dag_assignments[taxi_id])
            sorted_nodes = get_deployable_nodes(dag_info)
            t0 = time.perf_counter()
            nearest_server = find_k_nearest_servers(current_lat, current_lon, servers_df, k=1)[0][0]
            for node in sorted_nodes:
                taxi_dag_assignments[taxi_id][node] = nearest_server
            total_decision_time += time.perf_counter() - t0

            reward, details = calculate_microservice_reward(
                taxi_id,
                dag_info,
                taxi_dag_assignments[taxi_id],
                old_assignments,
                (current_lat, current_lon),
                servers_info,
                predicted_locations=predicted_locations,
                trigger_type=trigger_type,
            )
            total_reward_sum += reward
            reward_history.append(reward)
            total_access_latency += details["access_latency"]
            total_communication_cost += details["communication_cost"]
            total_internal_path_ms += float(details.get("internal_critical_path_ms", 0.0))
            total_migration_cost += details["migration_cost"]
            if float(details.get("migration_cost") or 0.0) > 0.0:
                migration_decision_count += 1
            total_cost_ms_sum += details["total_cost_ms"]
            total_sla_penalty_ms += details.get("sla_penalty_ms", 0.0)
            total_tearing_penalty_ms += details.get("tearing_penalty_ms", details.get("tearing_penalty", 0.0))
            total_future_penalty_ms += details.get("future_penalty_ms", details.get("future_penalty", 0.0))

            nodes_migrated = sum(
                1 for node in sorted_nodes
                if old_assignments[node] != taxi_dag_assignments[taxi_id][node]
            )
            total_migrations += nodes_migrated
            if dag_stats is not None and use_proactive and trigger_type == TRIGGER_PROACTIVE:
                dag_stats[dag_type]["proactive_decisions"] += 1
                dag_stats[dag_type]["migrated_nodes"] += nodes_migrated

            touch_taxi_last(taxi_last, taxi_id, row, current_lon, current_lat, ts)
        pbar.update(1)
    pbar.close()

    avg_ms = (total_decision_time / decision_count * 1000.0) if decision_count else 0.0
    sorted_excess = sorted(sla_excess_distance_history)
    p95_idx = int(0.95 * (len(sorted_excess) - 1)) if sorted_excess else 0
    return {
        "total_migrations": total_migrations,
        "total_violations": total_violations,
        "primary_entry_violations": primary_entry_violations,
        "max_entry_violations": max_entry_violations,
        "severe_sla_violations": severe_sla_violations,
        "total_sla_excess_distance_km": total_sla_excess_distance_km,
        "avg_sla_excess_distance_km": (
            total_sla_excess_distance_km / len(sla_excess_distance_history)
            if sla_excess_distance_history else 0.0
        ),
        "p95_sla_excess_distance_km": sorted_excess[p95_idx] if sorted_excess else 0.0,
        "proactive_decisions": proactive_decisions,
        "decision_count": decision_count,
        "total_reward": total_reward_sum,
        "total_access_latency": total_access_latency,
        "total_communication_cost": total_communication_cost,
        "total_internal_path_ms": total_internal_path_ms,
        "total_migration_cost": total_migration_cost,
        "migration_decision_count": migration_decision_count,
        "total_cost_ms_sum": total_cost_ms_sum,
        "total_sla_penalty_ms": total_sla_penalty_ms,
        "total_tearing_penalty_ms": total_tearing_penalty_ms,
        "total_future_penalty_ms": total_future_penalty_ms,
        "reward_history": reward_history,
        "total_decision_time": total_decision_time,
        "decision_count_for_latency": decision_count,
        "avg_decision_time_ms": avg_ms,
        "dag_proactive_migration_stats": dict(dag_stats or {}),
    }
