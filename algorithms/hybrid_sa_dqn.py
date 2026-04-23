"""
Hybrid RL-Refined SA Microservice Migration with Trigger-Aware Graph Attention.

Key Innovation: TriggerAwareGraphDQN uses graph attention mechanism that
conditions on the trigger type (PROACTIVE vs REACTIVE) to learn different
attention patterns for stateful vs stateless nodes.

SA generates a global draft -> Graph-DQN reviews per-node (Stay / Follow SA / Nearest).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from tqdm import tqdm
import random
import copy

from core.microservice_dags import MICROSERVICE_DAGS
from core.geo import haversine_distance, find_k_nearest_servers
from core.context import get_trigger_type, TRIGGER_PROACTIVE
from core.dag_utils import get_entry_nodes, topological_sort, assign_dag_type, initialize_dag_assignment
from core.reward import build_servers_info, calculate_microservice_reward
from core.state_builder import build_graph_state
from algorithms.sa import microservice_simulated_annealing

FORECAST_HORIZON = 15


class TriggerAwareGraphDQN(nn.Module):
    """
    Trigger-Conditioned Graph Attention Network for Microservice Migration.

    Architecture:
    1. Feature Enrichment: Concatenate trigger_context to each node's features
    2. Physics-Aware Attention: Compute attention weights considering:
       - Node connectivity (adjacency matrix)
       - Stateful flag (critical for migration cost)
       - Trigger type (PROACTIVE allows cheaper state sync)
    3. Message Passing: Aggregate neighbor features with attention weights
    4. Per-Node Decision: Output Q-values for each node's action

    Physical Intuition:
    - PROACTIVE trigger: network learns to be more aggressive with stateful
      nodes because background sync is cheap
    - REACTIVE trigger: network learns to prioritize stateless nodes or
      accept SA's conservative proposals for stateful nodes
    """

    def __init__(self, node_feat_dim=3, trigger_dim=2, sa_prior_dim=2,
                 hidden_dim=64, attention_heads=1, action_size=3):
        super().__init__()

        self.node_feat_dim = node_feat_dim
        self.trigger_dim = trigger_dim
        self.sa_prior_dim = sa_prior_dim
        self.hidden_dim = hidden_dim
        self.action_size = action_size

        # Input dimension after concatenating trigger context to node features
        enriched_dim = node_feat_dim + trigger_dim  # 3 + 2 = 5

        # === Layer 1: Initial Node Embedding ===
        # Transform raw features into latent space
        self.node_encoder = nn.Sequential(
            nn.Linear(enriched_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # === Layer 2: Physics-Aware Attention ===
        # Key insight: attention should consider BOTH topology (adj_matrix)
        # and node properties (especially is_stateful under different triggers)

        # Attention score computation: combines learned attention with topology
        self.attention_query = nn.Linear(hidden_dim, hidden_dim)
        self.attention_key = nn.Linear(hidden_dim, hidden_dim)

        # Trigger-conditioned attention modulation
        # This allows the network to learn different attention patterns for
        # PROACTIVE vs REACTIVE scenarios
        self.trigger_attention_gate = nn.Sequential(
            nn.Linear(trigger_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # Stateful-aware attention bias
        # Physical intuition: stateful nodes need more careful attention
        # when trigger is REACTIVE (expensive migration)
        self.stateful_attention_bias = nn.Parameter(torch.zeros(1))

        # === Layer 3: Message Passing Aggregation ===
        self.message_transform = nn.Linear(hidden_dim, hidden_dim)

        # === Layer 4: Graph Readout (per-node embedding) ===
        self.graph_norm = nn.LayerNorm(hidden_dim)

        # === Layer 5: Action Head ===
        # Combines node embedding with SA prior for final decision
        action_input_dim = hidden_dim + sa_prior_dim  # 64 + 2 = 66
        self.action_head = nn.Sequential(
            nn.Linear(action_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size),
        )

        # Store computed embeddings for per-node retrieval
        self._node_embeddings = None
        self._sa_priors = None

    def compute_graph_embeddings(self, node_features, adj_matrix, trigger_context, sa_priors):
        """
        Compute graph embeddings for all nodes in one forward pass.

        Parameters
        ----------
        node_features : torch.Tensor, shape (N, 3)
        adj_matrix : torch.Tensor, shape (N, N)
        trigger_context : torch.Tensor, shape (2,)
        sa_priors : torch.Tensor, shape (N, 2)

        Returns
        -------
        node_embeddings : torch.Tensor, shape (N, hidden_dim)
        """
        n_nodes = node_features.shape[0]
        device = node_features.device

        # === Step 1: Feature Enrichment ===
        # Broadcast trigger_context to all nodes: (N, 2)
        trigger_broadcast = trigger_context.unsqueeze(0).expand(n_nodes, -1)
        # Concatenate: (N, 5) = (N, 3) + (N, 2)
        enriched_features = torch.cat([node_features, trigger_broadcast], dim=1)

        # === Step 2: Initial Node Embedding ===
        # (N, 5) -> (N, hidden_dim)
        node_hidden = self.node_encoder(enriched_features)

        # === Step 3: Physics-Aware Attention ===
        # Compute attention scores
        queries = self.attention_query(node_hidden)  # (N, hidden_dim)
        keys = self.attention_key(node_hidden)       # (N, hidden_dim)

        # Raw attention: Q @ K^T / sqrt(d)
        attention_scores = torch.matmul(queries, keys.t()) / (self.hidden_dim ** 0.5)

        # Trigger-conditioned gating: modulate attention based on trigger type
        # Physical intuition: PROACTIVE allows more exploration (softer attention),
        # REACTIVE needs focused attention on critical nodes
        trigger_gate = self.trigger_attention_gate(trigger_context)  # (hidden_dim,)
        gate_factor = trigger_gate.mean()  # Scalar gating factor

        # Apply topology mask: only attend to connected nodes
        # adj_matrix acts as a soft mask (traffic-weighted connections)
        topology_mask = adj_matrix.clone()
        topology_mask[topology_mask == 0] = -1e9  # Mask disconnected pairs

        # Combine attention with topology
        attention_scores = attention_scores * gate_factor + topology_mask

        # Stateful-aware bias: increase attention to/from stateful nodes
        # Extract is_stateful flag (3rd feature, index 2)
        is_stateful = node_features[:, 2]  # (N,)

        # Create stateful attention bias matrix
        # When REACTIVE, stateful nodes get extra attention (they're expensive to move)
        is_reactive = 1.0 - trigger_context[0]  # 1 if REACTIVE, 0 if PROACTIVE
        stateful_bias = torch.outer(is_stateful, is_stateful) * self.stateful_attention_bias * is_reactive
        attention_scores = attention_scores + stateful_bias

        # Softmax normalization
        attention_weights = F.softmax(attention_scores, dim=1)  # (N, N)

        # === Step 4: Message Passing ===
        # Aggregate neighbor features weighted by attention
        messages = self.message_transform(node_hidden)  # (N, hidden_dim)
        aggregated = torch.matmul(attention_weights, messages)  # (N, hidden_dim)

        # Residual connection + normalization
        node_embeddings = self.graph_norm(node_hidden + aggregated)

        # Store for per-node retrieval
        self._node_embeddings = node_embeddings
        self._sa_priors = sa_priors

        return node_embeddings

    def forward(self, node_index):
        """
        Get Q-values for a specific node's action.

        Must call compute_graph_embeddings() first to populate embeddings.

        Parameters
        ----------
        node_index : int
            Index of the node to make decision for.

        Returns
        -------
        q_values : torch.Tensor, shape (action_size,)
        """
        if self._node_embeddings is None:
            raise RuntimeError("Must call compute_graph_embeddings() before forward()")

        # Get this node's embedding and SA prior
        node_emb = self._node_embeddings[node_index]  # (hidden_dim,)
        sa_prior = self._sa_priors[node_index]         # (2,)

        # Concatenate embedding with SA prior
        action_input = torch.cat([node_emb, sa_prior], dim=0)  # (hidden_dim + 2,)

        # Compute Q-values
        q_values = self.action_head(action_input)  # (action_size,)

        return q_values

    def forward_all_nodes(self):
        """
        Get Q-values for all nodes at once (for batch processing).

        Returns
        -------
        all_q_values : torch.Tensor, shape (N, action_size)
        """
        if self._node_embeddings is None:
            raise RuntimeError("Must call compute_graph_embeddings() before forward_all_nodes()")

        # Concatenate all embeddings with their SA priors
        action_inputs = torch.cat([self._node_embeddings, self._sa_priors], dim=1)

        # Compute Q-values for all nodes
        all_q_values = self.action_head(action_inputs)

        return all_q_values


def optimize_graph_model(memory, policy_net, target_net, optimizer, device,
                         batch_size=32, gamma=0.95):
    """
    Experience replay optimization for TriggerAwareGraphDQN.

    Memory entries contain: (graph_state_dict, node_idx, action, reward, next_graph_state_dict, done)
    """
    if len(memory) < batch_size:
        return None

    batch = random.sample(memory, batch_size)

    total_loss = 0.0
    optimizer.zero_grad()

    for (graph_state, node_idx, action, reward, next_graph_state, done) in batch:
        # Convert to tensors
        node_feat = torch.FloatTensor(graph_state['node_features']).to(device)
        adj = torch.FloatTensor(graph_state['adj_matrix']).to(device)
        trigger = torch.FloatTensor(graph_state['trigger_context']).to(device)
        sa_prior = torch.FloatTensor(graph_state['sa_priors']).to(device)

        # Current Q-value
        policy_net.compute_graph_embeddings(node_feat, adj, trigger, sa_prior)
        current_q = policy_net.forward(node_idx)[action]

        # Target Q-value
        with torch.no_grad():
            if done:
                target_q = torch.tensor(reward, dtype=torch.float32, device=device)
            else:
                next_node_feat = torch.FloatTensor(next_graph_state['node_features']).to(device)
                next_adj = torch.FloatTensor(next_graph_state['adj_matrix']).to(device)
                next_trigger = torch.FloatTensor(next_graph_state['trigger_context']).to(device)
                next_sa_prior = torch.FloatTensor(next_graph_state['sa_priors']).to(device)

                target_net.compute_graph_embeddings(next_node_feat, next_adj, next_trigger, next_sa_prior)
                next_q = target_net.forward(node_idx).max()
                target_q = torch.tensor(reward, dtype=torch.float32, device=device) + gamma * next_q

        loss = F.mse_loss(current_q, target_q)
        total_loss += loss

    # Average loss and backprop
    avg_loss = total_loss / batch_size
    avg_loss.backward()
    optimizer.step()

    return avg_loss.item()


def run_hybrid_microservice_fair(df, servers_df, predictor=None, proactive=False):
    """
    Hybrid RL-Refined SA microservice migration with Trigger-Aware Graph Attention.

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
    proactive_decisions = 0
    total_reward_sum = 0.0

    total_access_latency = 0.0
    total_communication_cost = 0.0
    total_migration_cost = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}  |  Proactive: {use_proactive}  |  Model: TriggerAwareGraphDQN")

    # Initialize Graph-DQN networks
    q_network = TriggerAwareGraphDQN(
        node_feat_dim=3,
        trigger_dim=2,
        sa_prior_dim=2,
        hidden_dim=64,
        action_size=3,
    ).to(device)

    target_network = TriggerAwareGraphDQN(
        node_feat_dim=3,
        trigger_dim=2,
        sa_prior_dim=2,
        hidden_dim=64,
        action_size=3,
    ).to(device)
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
    pbar = tqdm(total=len(timestamps), desc="Hybrid Graph-DQN Migration")

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

            # --- SCORING: Real violation count ---
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

            # --- Get trigger type ---
            trigger_type = get_trigger_type(
                current_lat, current_lon, gw_lat, gw_lon,
                current_dag_reward, predicted_locations,
                proactive_enabled=use_proactive,
            )

            if trigger_type is None:
                continue

            decision_count += 1
            if trigger_type == TRIGGER_PROACTIVE:
                proactive_decisions += 1

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
                trigger_type=trigger_type,
            )

            # Phase B: Build graph state and compute embeddings
            graph_state = build_graph_state(
                taxi_id, dag_info,
                taxi_dag_assignments[taxi_id],
                servers_info, trigger_type,
                sa_proposal, candidates,
                current_lat, current_lon,
            )

            # Convert to tensors and compute graph embeddings
            node_feat_t = torch.FloatTensor(graph_state['node_features']).to(device)
            adj_t = torch.FloatTensor(graph_state['adj_matrix']).to(device)
            trigger_t = torch.FloatTensor(graph_state['trigger_context']).to(device)
            sa_prior_t = torch.FloatTensor(graph_state['sa_priors']).to(device)

            with torch.no_grad():
                q_network.compute_graph_embeddings(node_feat_t, adj_t, trigger_t, sa_prior_t)

            # Phase C: Per-node decisions following topological order
            sorted_nodes = topological_sort(dag_info)
            node_to_idx = {name: i for i, name in enumerate(graph_state['node_names'])}
            node_transitions = []
            nearest_server_id = candidates[0][0]

            for ms_node in sorted_nodes:
                node_idx = node_to_idx[ms_node]
                sa_proposed_server = sa_proposal[ms_node]

                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = random.randint(0, 2)
                else:
                    with torch.no_grad():
                        q_values = q_network.forward(node_idx)
                        action = q_values.argmax().item()

                # Execute action
                current_node_server = taxi_dag_assignments[taxi_id][ms_node]
                if action == 0:
                    target_server = current_node_server
                elif action == 1:
                    target_server = sa_proposed_server
                else:
                    target_server = nearest_server_id

                taxi_dag_assignments[taxi_id][ms_node] = target_server
                node_transitions.append((node_idx, action))

            # Compute reward
            reward, details = calculate_microservice_reward(
                taxi_id, dag_info,
                taxi_dag_assignments[taxi_id], old_assignments,
                (current_lat, current_lon), servers_info,
                predicted_locations=predicted_locations,
                trigger_type=trigger_type,
            )
            total_reward_sum += reward
            reward_history.append(reward)

            total_access_latency += details['access_latency']
            total_communication_cost += details['communication_cost']
            total_migration_cost += details['migration_cost']

            nodes_migrated = sum(
                1 for n in sorted_nodes
                if old_assignments[n] != taxi_dag_assignments[taxi_id][n]
            )
            total_migrations += nodes_migrated

            # Build next graph state for replay buffer
            next_graph_state = build_graph_state(
                taxi_id, dag_info,
                taxi_dag_assignments[taxi_id],
                servers_info, trigger_type,
                sa_proposal, candidates,
                current_lat, current_lon,
            )

            # Store transitions in replay buffer
            for i, (node_idx, action) in enumerate(node_transitions):
                is_last = (i == len(node_transitions) - 1)
                if is_last:
                    step_reward = reward
                    done = True
                else:
                    step_reward = 0.0
                    done = False

                memory.append((
                    graph_state,
                    node_idx,
                    action,
                    step_reward,
                    next_graph_state,
                    done,
                ))

            # Optimize
            loss_val = optimize_graph_model(
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
        'proactive_decisions': proactive_decisions,
        'decision_count': decision_count,
        'total_reward': total_reward_sum,
        'total_access_latency': total_access_latency,
        'total_communication_cost': total_communication_cost,
        'total_migration_cost': total_migration_cost,
        'loss_history': loss_history,
        'reward_history': reward_history,
        'epsilon_history': epsilon_history,
    }


# Keep old class for backward compatibility (can be removed later)
class HybridMicroserviceDQN(nn.Module):
    """Legacy MLP-based DQN (deprecated, use TriggerAwareGraphDQN)."""
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
