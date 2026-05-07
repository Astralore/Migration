"""
SA Microservice Migration — topology-aware Simulated Annealing baseline.
Supports both reactive and proactive (trajectory-prediction) modes.
Implements asymmetric migration cost based on trigger type.
"""

import math
import copy
import random
import time
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from core.microservice_dags import MICROSERVICE_DAGS
from core.geo import haversine_distance, find_k_nearest_servers
from core.context import get_trigger_type, TRIGGER_PROACTIVE, TRIGGER_REACTIVE
from core.dag_utils import get_entry_nodes, assign_dag_type, initialize_dag_assignment, topological_sort
from core.reward import build_servers_info, calculate_microservice_reward
from prediction.simple_predictor import build_predict_future_time_kwargs, touch_taxi_last

FORECAST_HORIZON = 15  # Extended horizon for better proactive detection


class MicroserviceSAReturn(tuple):
    """
    2-tuple (best_assignments, best_cost) 解包与原生元组一致；
    附加 ``sa_stats`` 字典：含接受率、劣解接受率及原始计数（不破坏现有 ``a, b = ...`` 调用）。
    """

    def __new__(cls, best_assignments, best_cost, sa_stats):
        obj = tuple.__new__(cls, (best_assignments, best_cost))
        obj.sa_stats = sa_stats
        return obj


def microservice_simulated_annealing(
    taxi_id, dag_info, current_assignments, candidates,
    user_location, servers_info,
    previous_assignments=None,
    temp=3000.0, cooling_rate=0.97, max_iter=50,
    predicted_locations=None,
    trigger_type=TRIGGER_REACTIVE,
):
    """
    Simulated Annealing over all microservice node placements for one DAG.

    When *predicted_locations* is provided the cost function includes the
    future topology violation penalty so that SA also optimises for predicted
    user movement.

    Returns
    -------
    MicroserviceSAReturn
        可像 ``(best_assignments, best_cost)`` 一样解包为两项；
        统计指标见 ``result.sa_stats``（含 ``sa_accept_rate``、``sa_worse_accept_rate`` 等）。
    """
    if previous_assignments is None:
        previous_assignments = current_assignments

    candidate_server_ids = [c[0] for c in candidates]
    all_nodes = list(dag_info['nodes'].keys())

    current_sol = dict(current_assignments)
    current_reward, _ = calculate_microservice_reward(
        taxi_id, dag_info, current_sol, previous_assignments,
        user_location, servers_info,
        predicted_locations=predicted_locations,
        trigger_type=trigger_type,
    )
    current_cost = -current_reward

    best_sol = dict(current_sol)
    best_cost = current_cost

    sa_accept_count = 0
    sa_worse_accept_count = 0
    sa_neighbor_count = 0

    for _iteration in range(max_iter):
        node = random.choice(all_nodes)
        old_server = current_sol[node]

        other_servers = [s for s in candidate_server_ids if s != old_server]
        if not other_servers:
            temp *= cooling_rate
            continue
        new_server = random.choice(other_servers)

        sa_neighbor_count += 1

        neighbor_sol = dict(current_sol)
        neighbor_sol[node] = new_server

        neighbor_reward, _ = calculate_microservice_reward(
            taxi_id, dag_info, neighbor_sol, previous_assignments,
            user_location, servers_info,
            predicted_locations=predicted_locations,
            trigger_type=trigger_type,
        )
        neighbor_cost = -neighbor_reward

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

    sa_accept_rate = (
        sa_accept_count / sa_neighbor_count if sa_neighbor_count > 0 else 0.0
    )
    sa_worse_accept_rate = (
        sa_worse_accept_count / sa_neighbor_count if sa_neighbor_count > 0 else 0.0
    )

    sa_stats = {
        "sa_accept_count": sa_accept_count,
        "sa_worse_accept_count": sa_worse_accept_count,
        "sa_neighbor_count": sa_neighbor_count,
        "sa_accept_rate": sa_accept_rate,
        "sa_worse_accept_rate": sa_worse_accept_rate,
    }

    return MicroserviceSAReturn(best_sol, best_cost, sa_stats)


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
    proactive_decisions = 0
    total_reward_sum = 0.0
    reward_history = []

    # Cost breakdown tracking
    total_access_latency = 0.0
    total_communication_cost = 0.0
    total_migration_cost = 0.0
    
    # 时延探针初始化
    total_decision_time = 0.0
    decision_count_for_latency = 0

    dag_migration_stats = (
        defaultdict(lambda: {"proactive_decisions": 0, "migrated_nodes": 0})
        if collect_dag_proactive_stats else None
    )

    timestamps = sorted(df['date_time'].unique())
    df_grouped = df.groupby('date_time')
    taxi_last = {}

    decision_count = 0
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
            entry_nodes = get_entry_nodes(dag_info)
            gateway_node = entry_nodes[0]

            gateway_server_id = taxi_dag_assignments[taxi_id][gateway_node]
            gw_lat, gw_lon = servers_info[gateway_server_id]
            gateway_dist = haversine_distance(
                current_lat, current_lon, gw_lat, gw_lon
            )

            # --- SCORING: Real violation count (independent of trigger) ---
            if gateway_dist > 15.0:
                total_violations += 1

            # --- Trajectory prediction ---
            predicted_locations = None
            if use_proactive:
                raw = predictor.predict_future(
                    current_lon, current_lat, taxi_id, steps=FORECAST_HORIZON, **pf_kw
                )
                predicted_locations = [(lat, lon) for lon, lat in raw]

            # --- Get trigger type (REACTIVE / PROACTIVE / None) ---
            trigger_type = get_trigger_type(
                current_lat, current_lon, gw_lat, gw_lon,
                predicted_locations=predicted_locations,
                proactive_enabled=use_proactive,
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

            # 时延探针：计时 SA 决策过程
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

            # Recalculate to get cost breakdown details
            reward, details = calculate_microservice_reward(
                taxi_id, dag_info, best_assignments, old_assignments,
                (current_lat, current_lon), servers_info,
                predicted_locations=predicted_locations,
                trigger_type=trigger_type,
            )
            total_reward_sum += reward
            reward_history.append(reward)

            # Accumulate cost breakdown
            total_access_latency += details['access_latency']
            total_communication_cost += details['communication_cost']
            total_migration_cost += details['migration_cost']

            # 与 Hybrid SAC 同口径：拓扑序节点集合上比对迁移数
            sorted_nodes = topological_sort(dag_info)
            nodes_migrated = sum(
                1 for n in sorted_nodes
                if old_assignments[n] != best_assignments[n]
            )
            total_migrations += nodes_migrated

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

    return {
        'total_migrations': total_migrations,
        'total_violations': total_violations,
        'proactive_decisions': proactive_decisions,
        'decision_count': decision_count,
        'total_reward': total_reward_sum,
        'total_access_latency': total_access_latency,
        'total_communication_cost': total_communication_cost,
        'total_migration_cost': total_migration_cost,
        'reward_history': reward_history,
        # 时延信息
        'total_decision_time': total_decision_time,
        'decision_count_for_latency': decision_count_for_latency,
        'avg_decision_time_ms': (total_decision_time / decision_count_for_latency * 1000) if decision_count_for_latency > 0 else 0,
        'dag_proactive_migration_stats': (
            {k: dict(v) for k, v in dag_migration_stats.items()} if dag_migration_stats else {}
        ),
    }
