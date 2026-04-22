"""
Hybrid RL-Refined SA Microservice Migration.
SA generates a global draft -> DQN reviews per-node (Stay / Follow SA / Nearest).
Supports both reactive and proactive (trajectory-prediction) modes.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import tqdm
import random
import copy

from core.microservice_dags import MICROSERVICE_DAGS
from core.geo import haversine_distance, find_k_nearest_servers
from core.context import check_sla_violation, check_proactive_sla_violation
from core.dag_utils import get_entry_nodes, topological_sort, assign_dag_type, initialize_dag_assignment
from core.reward import build_servers_info, calculate_microservice_reward
from core.state_builder import build_hybrid_node_state
from algorithms.sa import microservice_simulated_annealing
from algorithms.dqn import optimize_model

FORECAST_HORIZON = 5


class HybridMicroserviceDQN(nn.Module):
    """
    action_size=3:
      0 = Stay        (keep current server)
      1 = Follow SA   (accept SA proposal)
      2 = Nearest     (move to absolute nearest candidate)
    """
    def __init__(self, input_size=18, hidden_size=128, action_size=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x):
        return self.network(x)


def run_hybrid_microservice_fair(df, servers_df, predictor=None, proactive=False):
    """
    Hybrid RL-Refined SA microservice migration main simulation.

    Parameters
    ----------
    predictor : SimpleTrajectoryPredictor or None
        Trajectory predictor (must already be fitted).
    proactive : bool
        When True and predictor is provided, enables look-ahead trigger,
        future-penalty reward, and mobility-trend state features.

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}  |  Proactive: {use_proactive}")
    q_network = HybridMicroserviceDQN(input_size=18, hidden_size=128, action_size=3).to(device)
    target_network = HybridMicroserviceDQN(input_size=18, hidden_size=128, action_size=3).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)

    memory = deque(maxlen=5000)
    epsilon = 0.3
    epsilon_decay = 0.995
    epsilon_min = 0.01
    batch_size = 32
    gamma_td = 0.95
    target_update_freq = 50

    loss_history = []
    reward_history = []
    epsilon_history = []

    timestamps = sorted(df['date_time'].unique())
    df_grouped = df.groupby('date_time')

    decision_count = 0
    pbar = tqdm(total=len(timestamps), desc="Hybrid Microservice Migration")

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
            entry_nodes_set = set(entry_nodes)
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

            # --- Trajectory prediction (lon,lat) -> (lat,lon) ---
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

            decision_count += 1

            # Phase A: SA global draft
            candidates = find_k_nearest_servers(
                current_lat, current_lon, servers_df, k=3
            )
            old_assignments = copy.copy(taxi_dag_assignments[taxi_id])

            sa_proposal, _sa_cost = microservice_simulated_annealing(
                taxi_id, dag_info,
                taxi_dag_assignments[taxi_id],
                candidates,
                user_location=(current_lat, current_lon),
                servers_info=servers_info,
                previous_assignments=old_assignments,
                predicted_locations=predicted_locations,
            )

            # Phase B: DQN per-node review
            sorted_nodes = topological_sort(dag_info)
            node_transitions = []
            nearest_server_id = candidates[0][0]

            for ms_node in sorted_nodes:
                sa_proposed_server = sa_proposal[ms_node]

                state = build_hybrid_node_state(
                    current_lat, current_lon, timestamp, gateway_dist,
                    ms_node, dag_info, taxi_dag_assignments[taxi_id],
                    candidates, servers_info, entry_nodes_set,
                    sa_proposed_server,
                    predicted_locations=predicted_locations,
                )

                if random.random() < epsilon:
                    action = random.randint(0, 2)
                else:
                    with torch.no_grad():
                        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                        q_values = q_network(state_t)
                        action = q_values.argmax().item()

                current_node_server = taxi_dag_assignments[taxi_id][ms_node]
                if action == 0:
                    target_server = current_node_server
                elif action == 1:
                    target_server = sa_proposed_server
                else:
                    target_server = nearest_server_id

                taxi_dag_assignments[taxi_id][ms_node] = target_server
                node_transitions.append((state, action))

            reward, _details = calculate_microservice_reward(
                taxi_id, dag_info,
                taxi_dag_assignments[taxi_id], old_assignments,
                (current_lat, current_lon), servers_info,
                predicted_locations=predicted_locations,
            )
            total_reward_sum += reward
            reward_history.append(reward)

            nodes_migrated = sum(
                1 for n in sorted_nodes
                if old_assignments[n] != taxi_dag_assignments[taxi_id][n]
            )
            total_migrations += nodes_migrated

            for i, (s, a) in enumerate(node_transitions):
                is_last = (i == len(node_transitions) - 1)
                if is_last:
                    next_s = s
                    step_reward = reward
                    done = True
                else:
                    next_s = node_transitions[i + 1][0]
                    step_reward = 0.0
                    done = False
                memory.append((s, a, step_reward, next_s, done))

            loss_val = optimize_model(
                memory, q_network, target_network, optimizer, device,
                batch_size=batch_size, gamma=gamma_td,
            )
            if loss_val is not None:
                loss_history.append(loss_val)
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay

            epsilon_history.append(epsilon)

            if decision_count % target_update_freq == 0:
                target_network.load_state_dict(q_network.state_dict())

        pbar.update(1)
    pbar.close()

    return {
        'total_migrations': total_migrations,
        'total_violations': total_violations,
        'total_reward': total_reward_sum,
        'loss_history': loss_history,
        'reward_history': reward_history,
        'epsilon_history': epsilon_history,
        'decision_count': decision_count,
    }
