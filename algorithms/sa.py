"""
SA Microservice Migration — topology-aware Simulated Annealing baseline.
Supports both reactive and proactive (trajectory-prediction) modes.
"""

import math
import copy
import random
from tqdm import tqdm

from core.microservice_dags import MICROSERVICE_DAGS
from core.geo import haversine_distance, find_k_nearest_servers
from core.context import check_sla_violation, check_proactive_sla_violation
from core.dag_utils import get_entry_nodes, assign_dag_type, initialize_dag_assignment
from core.reward import build_servers_info, calculate_microservice_reward

FORECAST_HORIZON = 5


def microservice_simulated_annealing(
    taxi_id, dag_info, current_assignments, candidates,
    user_location, servers_info,
    previous_assignments=None,
    temp=100.0, cooling_rate=0.95, max_iter=30,
    predicted_locations=None,
):
    """
    Simulated Annealing over all microservice node placements for one DAG.

    When *predicted_locations* is provided the cost function includes the
    future topology violation penalty so that SA also optimises for predicted
    user movement.

    Returns (best_assignments, best_cost).
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
    )
    current_cost = -current_reward

    best_sol = dict(current_sol)
    best_cost = current_cost

    for _iteration in range(max_iter):
        node = random.choice(all_nodes)
        old_server = current_sol[node]

        other_servers = [s for s in candidate_server_ids if s != old_server]
        if not other_servers:
            temp *= cooling_rate
            continue
        new_server = random.choice(other_servers)

        neighbor_sol = dict(current_sol)
        neighbor_sol[node] = new_server

        neighbor_reward, _ = calculate_microservice_reward(
            taxi_id, dag_info, neighbor_sol, previous_assignments,
            user_location, servers_info,
            predicted_locations=predicted_locations,
        )
        neighbor_cost = -neighbor_reward

        delta = neighbor_cost - current_cost
        if delta < 0:
            accept = True
        else:
            accept = random.random() < math.exp(-delta / temp) if temp > 1e-10 else False

        if accept:
            current_sol = neighbor_sol
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_sol = dict(current_sol)
                best_cost = current_cost

        temp *= cooling_rate

    return best_sol, best_cost


def run_sa_microservice_fair(df, servers_df, predictor=None, proactive=False):
    """
    SA microservice DAG migration main simulation.

    Parameters
    ----------
    predictor : SimpleTrajectoryPredictor or None
    proactive : bool

    Returns
    -------
    results : dict
    """
    servers_info = build_servers_info(servers_df)
    use_proactive = proactive and predictor is not None

    taxi_dag_type = {}
    taxi_dag_assignments = {}
    total_migrations = 0
    total_violations = 0
    total_reward_sum = 0.0
    reward_history = []

    timestamps = sorted(df['date_time'].unique())
    df_grouped = df.groupby('date_time')

    decision_count = 0
    pbar = tqdm(total=len(timestamps), desc="SA Microservice Migration")

    for timestamp in timestamps:
        current_rows = df_grouped.get_group(timestamp)
        for _, row in current_rows.iterrows():
            taxi_id = row['taxi_id']
            current_lat = row['latitude']
            current_lon = row['longitude']

            if taxi_id not in taxi_dag_assignments:
                nearest = find_k_nearest_servers(
                    current_lat, current_lon, servers_df, k=1
                )[0]
                chosen_dag = assign_dag_type()
                taxi_dag_type[taxi_id] = chosen_dag
                taxi_dag_assignments[taxi_id] = initialize_dag_assignment(
                    chosen_dag, nearest[0]
                )
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

            current_dag_reward, _ = calculate_microservice_reward(
                taxi_id, dag_info,
                taxi_dag_assignments[taxi_id], taxi_dag_assignments[taxi_id],
                (current_lat, current_lon), servers_info,
            )

            # --- Trajectory prediction ---
            predicted_locations = None
            if use_proactive:
                raw = predictor.predict_future(
                    current_lon, current_lat, taxi_id, steps=FORECAST_HORIZON
                )
                predicted_locations = [(lat, lon) for lon, lat in raw]

            # --- Trigger decision (may include proactive look-ahead) ---
            if use_proactive:
                should_migrate = check_proactive_sla_violation(
                    current_lat, current_lon, gw_lat, gw_lon,
                    current_dag_reward, predicted_locations,
                )
            else:
                should_migrate = check_sla_violation(
                    current_lat, current_lon, gw_lat, gw_lon,
                    current_dag_reward,
                )

            if not should_migrate:
                continue

            candidates = find_k_nearest_servers(
                current_lat, current_lon, servers_df, k=3
            )
            old_assignments = copy.copy(taxi_dag_assignments[taxi_id])

            best_assignments, best_cost = microservice_simulated_annealing(
                taxi_id, dag_info,
                taxi_dag_assignments[taxi_id],
                candidates,
                user_location=(current_lat, current_lon),
                servers_info=servers_info,
                previous_assignments=old_assignments,
                predicted_locations=predicted_locations,
            )

            taxi_dag_assignments[taxi_id] = best_assignments

            reward = -best_cost
            total_reward_sum += reward
            reward_history.append(reward)

            nodes_migrated = sum(
                1 for n in dag_info['nodes']
                if old_assignments[n] != best_assignments[n]
            )
            total_migrations += nodes_migrated
            decision_count += 1

        pbar.update(1)
    pbar.close()

    return {
        'total_migrations': total_migrations,
        'total_violations': total_violations,
        'total_reward': total_reward_sum,
        'reward_history': reward_history,
        'decision_count': decision_count,
    }
