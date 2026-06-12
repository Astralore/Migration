"""
SA Microservice Migration — topology-aware Simulated Annealing baseline.
Supports both reactive and proactive (trajectory-prediction) modes.
Optimizes reported physical total_cost_ms (not log-compressed reward).
"""

import math
import copy
import os
import random
import time
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from core.microservice_dags import MICROSERVICE_DAGS
from core.colocate_pattern import (
    build_colocate_pattern_mode_a,
    count_high_traffic_colocated_edges,
)
from core.geo import haversine_distance, find_k_nearest_servers
from core.context import get_trigger_type, TRIGGER_PROACTIVE, TRIGGER_REACTIVE, check_sla_violation
from core.dag_utils import (
    assign_dag_type,
    get_deployable_nodes,
    get_service_entry_nodes,
    initialize_dag_assignment,
)
from core.reward import (
    build_servers_info,
    calculate_entry_sla_metrics,
    calculate_microservice_reward,
    estimate_dag_migration_time_s,
)
from prediction.simple_predictor import build_predict_future_time_kwargs, touch_taxi_last

FORECAST_HORIZON = 15
SA_DEFAULT_MAX_ITER = int(os.environ.get("SA_MAX_ITER", "150"))
SA_DEFAULT_TEMP = float(os.environ.get("SA_INITIAL_TEMP", "50000.0"))
SA_DEFAULT_COOLING = float(os.environ.get("SA_COOLING_RATE", "0.99"))
SA_NUM_RESTARTS = int(os.environ.get("SA_NUM_RESTARTS", "2"))


SA_NUM_RESTARTS = int(os.environ.get("SA_NUM_RESTARTS", "2"))


def _sa_colocate_mode_a_enabled():
    """Traffic-aware partial colocate neighbourhood (aligned with GAT COLOCATE mode A)."""
    raw = os.environ.get("SA_COLOCATE_MODE_A")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().lower() not in ("0", "false", "no")
    return False


def _violating_entry_nodes(entry_nodes, assignments, user_lat, user_lon, servers_info):
    violating = []
    for node in entry_nodes or []:
        server_id = assignments.get(node)
        if server_id is None:
            continue
        srv_lat, srv_lon = servers_info[server_id]
        if check_sla_violation(user_lat, user_lon, srv_lat, srv_lon):
            violating.append(node)
    return violating


def _sa_mode_a_colocate_neighbor(
    current_sol,
    dag_info,
    entry_nodes,
    candidate_server_ids,
    user_location,
    servers_info,
):
    user_lat, user_lon = user_location
    violating = _violating_entry_nodes(
        entry_nodes, current_sol, user_lat, user_lon, servers_info
    )
    pick_from = violating or list(entry_nodes or [])
    if not pick_from or not candidate_server_ids:
        return None
    entry = random.choice(pick_from)
    old_server = current_sol.get(entry)
    targets = [s for s in candidate_server_ids if s != old_server]
    if not targets:
        return None
    target = random.choice(targets)
    pattern = build_colocate_pattern_mode_a(entry, target, current_sol, dag_info)
    neighbor = dict(current_sol)
    for node in pattern:
        neighbor[node] = target
    return neighbor


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


def _sa_total_cost_ms(
    taxi_id,
    dag_info,
    assignments,
    previous_assignments,
    user_location,
    servers_info,
    *,
    predicted_locations=None,
    trigger_type=TRIGGER_REACTIVE,
):
    """Physical system cost used in experiment reports (same as total_cost_ms_sum)."""
    _, details = calculate_microservice_reward(
        taxi_id,
        dag_info,
        assignments,
        previous_assignments,
        user_location,
        servers_info,
        predicted_locations=predicted_locations,
        trigger_type=trigger_type,
    )
    return float(details["total_cost_ms"])


def _sa_colocate_start(
    deployable_nodes,
    current_assignments,
    candidate_server_ids,
    *,
    dag_info=None,
    entry_nodes=None,
    user_location=None,
    servers_info=None,
):
    start = dict(current_assignments)
    if not deployable_nodes or not candidate_server_ids:
        return start
    if (
        _sa_colocate_mode_a_enabled()
        and dag_info is not None
        and entry_nodes
        and user_location is not None
        and servers_info is not None
    ):
        neighbor = _sa_mode_a_colocate_neighbor(
            current_assignments,
            dag_info,
            entry_nodes,
            candidate_server_ids,
            user_location,
            servers_info,
        )
        if neighbor is not None:
            return neighbor
    target = random.choice(candidate_server_ids)
    for node in deployable_nodes:
        start[node] = target
    return start


def _sa_propose_neighbor(
    current_sol,
    deployable_nodes,
    candidate_server_ids,
    *,
    dag_info=None,
    entry_nodes=None,
    user_location=None,
    servers_info=None,
):
    if not deployable_nodes or not candidate_server_ids:
        return None

    if random.random() < 0.5:
        if (
            _sa_colocate_mode_a_enabled()
            and dag_info is not None
            and entry_nodes
            and user_location is not None
            and servers_info is not None
        ):
            neighbor = _sa_mode_a_colocate_neighbor(
                current_sol,
                dag_info,
                entry_nodes,
                candidate_server_ids,
                user_location,
                servers_info,
            )
        else:
            neighbor = dict(current_sol)
            target = random.choice(candidate_server_ids)
            for node in deployable_nodes:
                neighbor[node] = target
        if neighbor is not None:
            return neighbor

    node = random.choice(deployable_nodes)
    old_server = current_sol[node]
    other_servers = [s for s in candidate_server_ids if s != old_server]
    if not other_servers:
        return None
    neighbor = dict(current_sol)
    neighbor[node] = random.choice(other_servers)
    return neighbor


class MicroserviceSAReturn(tuple):
    """
    2-tuple (best_assignments, best_cost) 解包与原生元组一致；
    附加 ``sa_stats`` 字典：含接受率、劣解接受率及原始计数（不破坏现有 ``a, b = ...`` 调用）。
    """

    def __new__(cls, best_assignments, best_cost, sa_stats):
        obj = tuple.__new__(cls, (best_assignments, best_cost))
        obj.sa_stats = sa_stats
        return obj


def _sa_single_run(
    taxi_id,
    dag_info,
    start_assignments,
    candidate_server_ids,
    user_location,
    servers_info,
    previous_assignments,
    *,
    entry_nodes,
    predicted_locations,
    trigger_type,
    temp,
    cooling_rate,
    max_iter,
):
    deployable_nodes = get_deployable_nodes(dag_info)
    current_sol = dict(start_assignments)
    current_cost = _sa_total_cost_ms(
        taxi_id,
        dag_info,
        current_sol,
        previous_assignments,
        user_location,
        servers_info,
        predicted_locations=predicted_locations,
        trigger_type=trigger_type,
    )

    best_sol = dict(current_sol)
    best_cost = current_cost

    sa_accept_count = 0
    sa_worse_accept_count = 0
    sa_neighbor_count = 0

    for _iteration in range(max_iter):
        neighbor_sol = _sa_propose_neighbor(
            current_sol,
            deployable_nodes,
            candidate_server_ids,
            dag_info=dag_info,
            entry_nodes=entry_nodes,
            user_location=user_location,
            servers_info=servers_info,
        )
        if neighbor_sol is None:
            temp *= cooling_rate
            continue

        sa_neighbor_count += 1
        neighbor_cost = _sa_total_cost_ms(
            taxi_id,
            dag_info,
            neighbor_sol,
            previous_assignments,
            user_location,
            servers_info,
            predicted_locations=predicted_locations,
            trigger_type=trigger_type,
        )

        delta = neighbor_cost - current_cost
        if delta < 0:
            accept = True
        else:
            accept = random.random() < math.exp(-delta / temp) if temp > 1e-10 else False

        if accept:
            sa_accept_count += 1
            if delta > 0:
                sa_worse_accept_count += 1
            current_sol = neighbor_sol
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_sol = dict(current_sol)
                best_cost = current_cost

        temp *= cooling_rate

    return best_sol, best_cost, {
        "sa_accept_count": sa_accept_count,
        "sa_worse_accept_count": sa_worse_accept_count,
        "sa_neighbor_count": sa_neighbor_count,
    }


def microservice_simulated_annealing(
    taxi_id, dag_info, current_assignments, candidates,
    user_location, servers_info,
    previous_assignments=None,
    temp=SA_DEFAULT_TEMP, cooling_rate=SA_DEFAULT_COOLING, max_iter=SA_DEFAULT_MAX_ITER,
    predicted_locations=None,
    trigger_type=TRIGGER_REACTIVE,
    num_restarts=SA_NUM_RESTARTS,
):
    """
    Simulated Annealing over deployable microservice node placements for one DAG.

    Minimizes ``details['total_cost_ms']`` so the search objective matches experiment
    reporting (SLA + linear migration + tearing + comm + future + access).

    Neighbourhood: single-node server change or colocate-all-deployable to one candidate.
    Optional random restarts from a colocated seed improve DAG-wide coordination.
    """
    if previous_assignments is None:
        previous_assignments = current_assignments

    candidate_server_ids = [c[0] for c in candidates]
    deployable_nodes = get_deployable_nodes(dag_info)
    entry_nodes = get_service_entry_nodes(dag_info)

    start_points = [dict(current_assignments)]
    restarts = max(1, int(num_restarts))
    while len(start_points) < restarts:
        start_points.append(
            _sa_colocate_start(
                deployable_nodes,
                current_assignments,
                candidate_server_ids,
                dag_info=dag_info,
                entry_nodes=entry_nodes,
                user_location=user_location,
                servers_info=servers_info,
            )
        )

    best_sol = dict(current_assignments)
    best_cost = _sa_total_cost_ms(
        taxi_id,
        dag_info,
        best_sol,
        previous_assignments,
        user_location,
        servers_info,
        predicted_locations=predicted_locations,
        trigger_type=trigger_type,
    )

    agg_stats = {
        "sa_accept_count": 0,
        "sa_worse_accept_count": 0,
        "sa_neighbor_count": 0,
        "sa_restart_count": len(start_points),
    }

    for start_sol in start_points:
        run_temp = max(float(temp), best_cost * 0.15, 1000.0)
        sol, cost, stats = _sa_single_run(
            taxi_id,
            dag_info,
            start_sol,
            candidate_server_ids,
            user_location,
            servers_info,
            previous_assignments,
            entry_nodes=entry_nodes,
            predicted_locations=predicted_locations,
            trigger_type=trigger_type,
            temp=run_temp,
            cooling_rate=cooling_rate,
            max_iter=max_iter,
        )
        for key in ("sa_accept_count", "sa_worse_accept_count", "sa_neighbor_count"):
            agg_stats[key] += stats[key]
        if cost < best_cost:
            best_sol = sol
            best_cost = cost

    neighbor_count = agg_stats["sa_neighbor_count"]
    agg_stats["sa_accept_rate"] = (
        agg_stats["sa_accept_count"] / neighbor_count if neighbor_count > 0 else 0.0
    )
    agg_stats["sa_worse_accept_rate"] = (
        agg_stats["sa_worse_accept_count"] / neighbor_count if neighbor_count > 0 else 0.0
    )

    return MicroserviceSAReturn(best_sol, best_cost, agg_stats)


def run_sa_microservice_fair(
    df, servers_df, predictor=None, proactive=False, collect_dag_proactive_stats=False,
):
    """
    SA microservice DAG migration main simulation.

    Parameters
    ----------
    predictor : SimpleTrajectoryPredictor or None
    proactive : bool
    collect_dag_proactive_stats : bool
        **仅用于推理实验**：为 True 时在 ``use_proactive`` 且 ``TRIGGER_PROACTIVE`` 的决策上
        按 ``dag_type`` 旁路聚合 ``dag_proactive_migration_stats``。
        **训练阶段必须保持 False**（不传或默认），不建表、不计数。

    Returns
    -------
    results : dict
        Contains: total_migrations, total_violations, proactive_decisions,
                  decision_count, total_reward, reward_history,
                  dag_proactive_migration_stats（collect 时非空）.
    """
    servers_info = build_servers_info(servers_df)
    use_proactive = proactive and predictor is not None

    taxi_dag_type = {}
    taxi_dag_assignments = {}
    total_migrations = 0
    total_violations = 0
    primary_entry_violations = 0
    max_entry_violations = 0
    severe_sla_violations = 0
    total_sla_excess_distance_km = 0.0
    sla_excess_distance_history = []
    proactive_decisions = 0
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
    decision_count_for_latency = 0
    high_traffic_colocated_edges = 0
    high_traffic_critical_edges = 0

    dag_migration_stats = (
        defaultdict(lambda: {"proactive_decisions": 0, "migrated_nodes": 0})
        if collect_dag_proactive_stats else None
    )

    timestamps = sorted(df['date_time'].unique())
    df_grouped = df.groupby('date_time')
    taxi_last = {}

    decision_count = 0
    if _sa_colocate_mode_a_enabled():
        print("  SA neighbourhood: mode-A traffic-aware colocate + single-node", flush=True)
    pbar = tqdm(total=len(timestamps), desc="SA Microservice Migration")

    for timestamp in timestamps:
        current_rows = df_grouped.get_group(timestamp)
        for _, row in current_rows.iterrows():
            taxi_id = row['taxi_id']
            current_lat = row['latitude']
            current_lon = row['longitude']
            ts = pd.Timestamp(timestamp)
            pf_kw = build_predict_future_time_kwargs(
                taxi_last, taxi_id, row, current_lon, current_lat, ts
            )

            if taxi_id not in taxi_dag_assignments:
                nearest = find_k_nearest_servers(
                    current_lat, current_lon, servers_df, k=1
                )[0]
                chosen_dag = assign_dag_type()
                taxi_dag_type[taxi_id] = chosen_dag
                taxi_dag_assignments[taxi_id] = initialize_dag_assignment(
                    chosen_dag, nearest[0]
                )
                touch_taxi_last(taxi_last, taxi_id, row, current_lon, current_lat, ts)
                continue

            dag_type = taxi_dag_type[taxi_id]
            dag_info = MICROSERVICE_DAGS[dag_type]
            entry_nodes = get_service_entry_nodes(dag_info)
            if not entry_nodes:
                touch_taxi_last(taxi_last, taxi_id, row, current_lon, current_lat, ts)
                continue
            gateway_node = entry_nodes[0]

            gateway_server_id = taxi_dag_assignments[taxi_id][gateway_node]
            gw_lat, gw_lon = servers_info[gateway_server_id]
            gateway_dist = haversine_distance(
                current_lat, current_lon, gw_lat, gw_lon
            )

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
                current_lat, current_lon, gw_lat, gw_lon,
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

            candidates = find_k_nearest_servers(
                current_lat, current_lon, servers_df, k=3
            )
            old_assignments = copy.copy(taxi_dag_assignments[taxi_id])

            t_start = time.perf_counter()
            best_assignments, best_cost = microservice_simulated_annealing(
                taxi_id, dag_info,
                taxi_dag_assignments[taxi_id],
                candidates,
                user_location=(current_lat, current_lon),
                servers_info=servers_info,
                previous_assignments=old_assignments,
                predicted_locations=predicted_locations,
                trigger_type=trigger_type,
            )
            t_end = time.perf_counter()

            total_decision_time += (t_end - t_start)
            decision_count_for_latency += 1

            taxi_dag_assignments[taxi_id] = best_assignments

            reward, details = calculate_microservice_reward(
                taxi_id, dag_info, best_assignments, old_assignments,
                (current_lat, current_lon), servers_info,
                predicted_locations=predicted_locations,
                trigger_type=trigger_type,
            )
            total_reward_sum += reward
            reward_history.append(reward)

            total_access_latency += details['access_latency']
            total_communication_cost += details['communication_cost']
            total_internal_path_ms += float(details.get('internal_critical_path_ms', 0.0))
            total_migration_cost += details['migration_cost']
            if float(details.get('migration_cost') or 0.0) > 0.0:
                migration_decision_count += 1
            total_cost_ms_sum += details['total_cost_ms']
            total_sla_penalty_ms += details.get('sla_penalty_ms', 0.0)
            total_tearing_penalty_ms += details.get('tearing_penalty_ms', details.get('tearing_penalty', 0.0))
            total_future_penalty_ms += details.get('future_penalty_ms', details.get('future_penalty', 0.0))

            sorted_nodes = get_deployable_nodes(dag_info)
            nodes_migrated = sum(
                1 for n in sorted_nodes
                if old_assignments[n] != best_assignments[n]
            )
            total_migrations += nodes_migrated

            ht_colocated, ht_total = count_high_traffic_colocated_edges(
                dag_info, best_assignments
            )
            high_traffic_colocated_edges += ht_colocated
            high_traffic_critical_edges += ht_total

            if (
                collect_dag_proactive_stats
                and dag_migration_stats is not None
                and use_proactive
                and trigger_type == TRIGGER_PROACTIVE
            ):
                dag_migration_stats[dag_type]["proactive_decisions"] += 1
                dag_migration_stats[dag_type]["migrated_nodes"] += nodes_migrated

            touch_taxi_last(taxi_last, taxi_id, row, current_lon, current_lat, ts)

        pbar.update(1)
    pbar.close()

    sorted_excess = sorted(sla_excess_distance_history)
    p95_idx = int(0.95 * (len(sorted_excess) - 1)) if sorted_excess else 0
    return {
        'total_migrations': total_migrations,
        'total_violations': total_violations,
        'primary_entry_violations': primary_entry_violations,
        'max_entry_violations': max_entry_violations,
        'severe_sla_violations': severe_sla_violations,
        'total_sla_excess_distance_km': total_sla_excess_distance_km,
        'avg_sla_excess_distance_km': (
            total_sla_excess_distance_km / len(sla_excess_distance_history)
            if sla_excess_distance_history else 0.0
        ),
        'p95_sla_excess_distance_km': sorted_excess[p95_idx] if sorted_excess else 0.0,
        'proactive_decisions': proactive_decisions,
        'decision_count': decision_count,
        'total_reward': total_reward_sum,
        'total_access_latency': total_access_latency,
        'total_communication_cost': total_communication_cost,
        'total_internal_path_ms': total_internal_path_ms,
        'total_migration_cost': total_migration_cost,
        'migration_decision_count': migration_decision_count,
        'total_cost_ms_sum': total_cost_ms_sum,
        'total_sla_penalty_ms': total_sla_penalty_ms,
        'total_tearing_penalty_ms': total_tearing_penalty_ms,
        'total_future_penalty_ms': total_future_penalty_ms,
        'reward_history': reward_history,
        'total_decision_time': total_decision_time,
        'decision_count_for_latency': decision_count_for_latency,
        'avg_decision_time_ms': (total_decision_time / decision_count_for_latency * 1000) if decision_count_for_latency > 0 else 0,
        'dag_proactive_migration_stats': (
            {k: dict(v) for k, v in dag_migration_stats.items()} if dag_migration_stats else {}
        ),
        'high_traffic_colocated_edges': high_traffic_colocated_edges,
        'high_traffic_critical_edges': high_traffic_critical_edges,
        'high_traffic_colocated_ratio': (
            high_traffic_colocated_edges / high_traffic_critical_edges
            if high_traffic_critical_edges > 0 else 0.0
        ),
        'sa_colocate_mode_a': _sa_colocate_mode_a_enabled(),
    }
