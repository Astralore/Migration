"""
DQN Microservice Migration — DAG-aware collaborative migration via Deep Q-Network.
Supports both reactive and proactive (trajectory-prediction) modes.
Implements asymmetric migration cost based on trigger type.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import copy

import pandas as pd
from tqdm import tqdm

from core.microservice_dags import MICROSERVICE_DAGS
from core.geo import haversine_distance, find_k_nearest_servers
from core.context import get_trigger_type, TRIGGER_PROACTIVE, check_sla_violation
from core.dag_utils import (
    assign_dag_type,
    get_deployable_nodes,
    get_service_entry_nodes,
    initialize_dag_assignment,
    topological_sort,
)
from core.reward import build_servers_info, calculate_microservice_reward, estimate_dag_migration_time_s
from core.state_builder import build_node_state
from prediction.simple_predictor import build_predict_future_time_kwargs, touch_taxi_last

FORECAST_HORIZON = 15  # Extended horizon for better proactive detection


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


class MicroserviceDQN(nn.Module):
    def __init__(self, input_size=16, hidden_size=128, action_size=4):
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


def optimize_model(memory, policy_net, target_net, optimizer, device,
                   batch_size=32, gamma=0.95):
    """Sample a batch from replay buffer and perform one DQN training step."""
    if len(memory) < batch_size:
        return None

    batch = random.sample(memory, batch_size)
    states_b = torch.FloatTensor(np.array([e[0] for e in batch])).to(device)
    actions_b = torch.LongTensor([e[1] for e in batch]).to(device)
    rewards_b = torch.FloatTensor([e[2] for e in batch]).to(device)
    next_states_b = torch.FloatTensor(np.array([e[3] for e in batch])).to(device)
    dones_b = torch.BoolTensor([e[4] for e in batch]).to(device)

    current_q = policy_net(states_b).gather(1, actions_b.unsqueeze(1))
    with torch.no_grad():
        next_q = target_net(next_states_b).max(1)[0].unsqueeze(1)
    target_q = rewards_b.unsqueeze(1) + (gamma * next_q * ~dones_b.unsqueeze(1))

    loss = F.mse_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def _save_dqn_checkpoint(path, q_net, target_net):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "q_network": q_net.state_dict(),
            "target_network": target_net.state_dict(),
        },
        path,
    )
    print(f"  [DQN SAVE] Weights saved to {path}")


def _load_dqn_checkpoint(path, q_net, target_net, device):
    ckpt = torch.load(path, map_location=device)
    q_net.load_state_dict(ckpt["q_network"])
    target_net.load_state_dict(ckpt["target_network"])
    print(f"  [DQN LOAD] Weights loaded from {path}")


def run_dqn_microservice_fair(
    df,
    servers_df,
    predictor=None,
    proactive=False,
    inference_mode=False,
    checkpoint_path=None,
    save_checkpoint_path=None,
):
    """
    DQN microservice DAG collaborative migration main simulation.

    Parameters
    ----------
    predictor : SimpleTrajectoryPredictor or None
        Trajectory predictor (must already be fitted).
    proactive : bool
        When True and predictor is provided, enables look-ahead trigger
        and future-penalty reward.

    Returns
    -------
    results : dict
        Contains: total_migrations, total_violations, proactive_decisions,
                  decision_count, total_reward, loss_history, etc.
    """
    servers_info = build_servers_info(servers_df)
    use_proactive = proactive and predictor is not None

    taxi_dag_type = {}
    taxi_dag_assignments = {}
    total_migrations = 0
    total_violations = 0
    primary_entry_violations = 0
    max_entry_violations = 0
    proactive_decisions = 0
    total_reward_sum = 0.0

    # Cost breakdown tracking
    total_access_latency = 0.0
    total_communication_cost = 0.0
    total_migration_cost = 0.0
    total_cost_ms_sum = 0.0
    total_sla_penalty_ms = 0.0
    total_tearing_penalty_ms = 0.0
    total_future_penalty_ms = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}  |  Proactive: {use_proactive}")
    q_network = MicroserviceDQN(input_size=16, hidden_size=128, action_size=4).to(device)
    target_network = MicroserviceDQN(input_size=16, hidden_size=128, action_size=4).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)

    memory = deque(maxlen=5000)
    if inference_mode:
        if checkpoint_path is None:
            raise ValueError("inference_mode=True 但未提供 checkpoint_path")
        _load_dqn_checkpoint(checkpoint_path, q_network, target_network, device)
        epsilon = 0.0
    else:
        epsilon = 0.3
    epsilon_decay = 0.995
    epsilon_min = 0.01
    batch_size = 32
    gamma_td = 0.95
    target_update_freq = 50

    loss_history = []
    reward_history = []
    epsilon_history = []

    total_decision_time = 0.0

    timestamps = sorted(df['date_time'].unique())
    df_grouped = df.groupby('date_time')
    taxi_last = {}

    decision_count = 0
    pbar = tqdm(total=len(timestamps), desc="DQN Microservice Migration")

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
            entry_nodes_set = set(entry_nodes)
            gateway_node = entry_nodes[0]

            gateway_server_id = taxi_dag_assignments[taxi_id][gateway_node]
            gw_lat, gw_lon = servers_info[gateway_server_id]
            gateway_dist = haversine_distance(
                current_lat, current_lon, gw_lat, gw_lon
            )

            # --- SCORING: max-entry SLA count, with primary-entry retained for diagnostics ---
            primary_v, max_v = _entry_violation_counts(
                entry_nodes, taxi_dag_assignments[taxi_id], current_lat, current_lon, servers_info
            )
            primary_entry_violations += primary_v
            max_entry_violations += max_v
            total_violations += max_v

            # --- Trajectory prediction (lon,lat) -> (lat,lon) ---
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
            sorted_nodes = topological_sort(dag_info)
            node_transitions = []

            t_decision_start = time.perf_counter()
            for ms_node in sorted_nodes:
                state = build_node_state(
                    current_lat, current_lon, timestamp, gateway_dist,
                    ms_node, dag_info, taxi_dag_assignments[taxi_id],
                    candidates, servers_info, entry_nodes_set,
                    predicted_locations=predicted_locations,
                )

                if random.random() < epsilon:
                    action = random.randint(0, 3)
                else:
                    with torch.no_grad():
                        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                        q_values = q_network(state_t)
                        action = q_values.argmax().item()

                current_node_server = taxi_dag_assignments[taxi_id][ms_node]
                if action == 0:
                    new_server = current_node_server
                elif action == 1 and len(candidates) >= 1:
                    new_server = candidates[0][0]
                elif action == 2 and len(candidates) >= 2:
                    new_server = candidates[1][0]
                elif action == 3 and len(candidates) >= 3:
                    new_server = candidates[2][0]
                else:
                    new_server = candidates[0][0] if candidates else current_node_server

                taxi_dag_assignments[taxi_id][ms_node] = new_server
                node_transitions.append((state, action))

            total_decision_time += time.perf_counter() - t_decision_start

            # Reward with asymmetric migration cost
            reward, details = calculate_microservice_reward(
                taxi_id, dag_info,
                taxi_dag_assignments[taxi_id], old_assignments,
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
            total_cost_ms_sum += details['total_cost_ms']
            total_sla_penalty_ms += details.get('sla_penalty_ms', 0.0)
            total_tearing_penalty_ms += details.get('tearing_penalty_ms', details.get('tearing_penalty', 0.0))
            total_future_penalty_ms += details.get('future_penalty_ms', details.get('future_penalty', 0.0))

            deployable_nodes = get_deployable_nodes(dag_info)
            nodes_migrated = sum(
                1 for n in deployable_nodes
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

            if not inference_mode:
                loss_val = optimize_model(
                    memory, q_network, target_network, optimizer, device,
                    batch_size=batch_size, gamma=gamma_td,
                )
                if loss_val is not None:
                    loss_history.append(loss_val)
                    if epsilon > epsilon_min:
                        epsilon *= epsilon_decay

            epsilon_history.append(epsilon)

            if not inference_mode and decision_count % target_update_freq == 0:
                target_network.load_state_dict(q_network.state_dict())

            touch_taxi_last(taxi_last, taxi_id, row, current_lon, current_lat, ts)

        pbar.update(1)
    pbar.close()

    if not inference_mode and save_checkpoint_path:
        _save_dqn_checkpoint(save_checkpoint_path, q_network, target_network)

    avg_ms = (
        (total_decision_time / decision_count * 1000.0) if decision_count > 0 else 0.0
    )
    return {
        'total_migrations': total_migrations,
        'total_violations': total_violations,
        'primary_entry_violations': primary_entry_violations,
        'max_entry_violations': max_entry_violations,
        'proactive_decisions': proactive_decisions,
        'decision_count': decision_count,
        'total_reward': total_reward_sum,
        'total_access_latency': total_access_latency,
        'total_communication_cost': total_communication_cost,
        'total_migration_cost': total_migration_cost,
        'total_cost_ms_sum': total_cost_ms_sum,
        'total_sla_penalty_ms': total_sla_penalty_ms,
        'total_tearing_penalty_ms': total_tearing_penalty_ms,
        'total_future_penalty_ms': total_future_penalty_ms,
        'loss_history': loss_history,
        'reward_history': reward_history,
        'epsilon_history': epsilon_history,
        'total_decision_time': total_decision_time,
        'decision_count_for_latency': decision_count,
        'avg_decision_time_ms': avg_ms,
    }
