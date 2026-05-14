"""
CTDE-GAT-MARL for mobile edge microservice migration.

This is an independent multi-agent method: no SA proposal, no FOLLOW_SA action,
and no behavior cloning.  Each microservice node is an agent with a shared actor;
training uses a centralized critic over the pooled DAG embedding and joint action.
"""

from collections import defaultdict, deque
import copy
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from core.context import TRIGGER_PROACTIVE, get_trigger_type, check_sla_violation
from core.dag_utils import (
    assign_dag_type,
    get_service_entry_nodes,
    initialize_dag_assignment,
    is_external_node,
    topological_sort,
)
from core.geo import find_k_nearest_servers, haversine_distance
from core.marl_reward import calculate_marl_rewards, lambda_schedule_by_epoch
from core.marl_state_builder import (
    MARL_ACTION_DIM,
    action_to_server,
    build_marl_graph_state,
)
from core.microservice_dags import MICROSERVICE_DAGS
from core.reward import build_servers_info, estimate_dag_migration_time_s
from prediction.simple_predictor import build_predict_future_time_kwargs, touch_taxi_last


FORECAST_HORIZON = 15
MAX_NODES = 12


class GraphEncoder(nn.Module):
    """Lightweight GAT-style message passing encoder for small DAGs."""

    def __init__(self, node_feat_dim, trigger_dim=3, mobility_dim=2,
                 candidate_dim=6, hidden_dim=64, output_dim=64):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(node_feat_dim + trigger_dim + mobility_dim + candidate_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.self_proj = nn.Linear(hidden_dim, output_dim)
        self.neighbor_proj = nn.Linear(hidden_dim, output_dim)
        self.out_norm = nn.LayerNorm(output_dim)

    def forward(self, node_features, adj_matrix, trigger_context, mobility_context, candidate_features):
        n = node_features.shape[0]
        ctx = torch.cat([trigger_context, mobility_context, candidate_features], dim=-1)
        ctx = ctx.unsqueeze(0).expand(n, -1)
        h = self.input_proj(torch.cat([node_features, ctx], dim=-1))
        denom = adj_matrix.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        msg = torch.matmul(adj_matrix, h) / denom
        out = self.self_proj(h) + self.neighbor_proj(msg)
        return F.relu(self.out_norm(out))


class SharedNodeActor(nn.Module):
    def __init__(self, embedding_dim=64, trigger_dim=3, hidden_dim=128, action_dim=MARL_ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + trigger_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, node_embeddings, trigger_context):
        n = node_embeddings.shape[0]
        trigger = trigger_context.unsqueeze(0).expand(n, -1)
        return self.net(torch.cat([node_embeddings, trigger], dim=-1))


class CentralCritic(nn.Module):
    def __init__(self, embedding_dim=64, trigger_dim=3, mobility_dim=2,
                 candidate_dim=6, max_nodes=MAX_NODES, action_dim=MARL_ACTION_DIM,
                 hidden_dim=128):
        super().__init__()
        self.max_nodes = max_nodes
        self.action_dim = action_dim
        joint_dim = max_nodes * action_dim
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + trigger_dim + mobility_dim + candidate_dim + joint_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph_embedding, trigger_context, mobility_context, candidate_features, joint_action_onehot):
        x = torch.cat(
            [graph_embedding, trigger_context, mobility_context, candidate_features, joint_action_onehot],
            dim=-1,
        )
        return self.net(x).squeeze(-1)


def _apply_action_mask(logits, action_mask):
    mask = torch.as_tensor(action_mask, dtype=torch.bool, device=logits.device)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    mask = mask.clone()
    fallback = ~mask.any(dim=-1)
    if fallback.any():
        mask[fallback, 0] = True
    return logits.masked_fill(~mask, -1e9), int((~mask).sum().item()), int(fallback.sum().item())


def _state_to_tensors(state, device):
    return {
        "node_features": torch.FloatTensor(state["node_features"]).to(device),
        "adj_matrix": torch.FloatTensor(state["adj_matrix"]).to(device),
        "trigger_context": torch.FloatTensor(state["trigger_context"]).to(device),
        "mobility_context": torch.FloatTensor(state["mobility_context"]).to(device),
        "candidate_features": torch.FloatTensor(state["candidate_features"]).to(device),
        "action_masks": torch.BoolTensor(state["action_masks"]).to(device),
    }


def _joint_onehot(actions, max_nodes=MAX_NODES, action_dim=MARL_ACTION_DIM, device=None):
    out = torch.zeros(max_nodes, action_dim, device=device)
    for i, action in enumerate(actions[:max_nodes]):
        out[i, int(action)] = 1.0
    return out.reshape(-1)


def _save_marl_checkpoint(path, encoder, actor, critic):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
        },
        path,
    )
    print(f"  [GAT-MARL SAVE] Weights saved to {path}")


def _load_marl_checkpoint(path, encoder, actor, critic, device):
    ckpt = torch.load(path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])
    print(f"  [GAT-MARL LOAD] Weights loaded from {path}")


def _optimize_marl(memory, encoder, actor, critic, optimizer, device,
                   batch_size=32, entropy_coef=0.08):
    if len(memory) < batch_size:
        return None
    batch = random.sample(memory, batch_size)
    total_loss = torch.tensor(0.0, device=device)
    policy_losses = []
    value_losses = []
    entropies = []

    optimizer.zero_grad()
    for transition in batch:
        tensors = _state_to_tensors(transition["state"], device)
        embeddings = encoder(
            tensors["node_features"],
            tensors["adj_matrix"],
            tensors["trigger_context"],
            tensors["mobility_context"],
            tensors["candidate_features"],
        )
        graph_embedding = embeddings.mean(dim=0)
        logits = actor(embeddings, tensors["trigger_context"])
        masked_logits, _, _ = _apply_action_mask(logits, tensors["action_masks"])
        log_probs = F.log_softmax(masked_logits, dim=-1)
        probs = F.softmax(masked_logits, dim=-1)
        actions = torch.LongTensor(transition["actions"]).to(device)
        chosen_log_prob = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        joint = _joint_onehot(transition["actions"], device=device)
        value = critic(
            graph_embedding,
            tensors["trigger_context"],
            tensors["mobility_context"],
            tensors["candidate_features"],
            joint,
        )
        training_reward = torch.tensor(
            float(transition.get("training_reward", transition["shared_reward"])),
            dtype=torch.float32,
            device=device,
        )
        agent_rewards = torch.FloatTensor(transition.get("agent_reward_list", [])).to(device)
        if agent_rewards.numel() != actions.numel():
            agent_rewards = training_reward.expand_as(chosen_log_prob)
        advantage = agent_rewards - value.detach()
        policy_loss = -(chosen_log_prob * advantage).mean() - entropy_coef * entropy
        value_loss = F.smooth_l1_loss(value, training_reward)
        total_loss = total_loss + policy_loss + value_loss
        policy_losses.append(float(policy_loss.detach().item()))
        value_losses.append(float(value_loss.detach().item()))
        entropies.append(float(entropy.detach().item()))

    avg_loss = total_loss / float(batch_size)
    avg_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(actor.parameters()) + list(critic.parameters()),
        max_norm=1.0,
    )
    optimizer.step()
    return {
        "loss": float(avg_loss.detach().item()),
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
    }


def _dag_complexity_key(dag_info):
    n = len(dag_info["nodes"])
    e = len(dag_info["edges"])
    if n >= 8 or e >= 10:
        return "complex"
    if n >= 5 or e >= 5:
        return "medium"
    return "simple"


def run_marl_gat_microservice(
    df,
    servers_df,
    predictor=None,
    proactive=False,
    num_epochs=4,
    inference_mode=False,
    checkpoint_path=None,
    save_checkpoint_path=None,
    collect_dag_proactive_stats=False,
    max_lambda_migration=None,
    max_lambda_split=None,
):
    """Run CTDE-GAT-MARL under the same trigger/reward protocol as baselines."""
    servers_info = build_servers_info(servers_df)
    use_proactive = proactive and predictor is not None
    if max_lambda_migration is None:
        max_lambda_migration = 0.15 if use_proactive else 0.25  # 硬调 Reactive 到 0.25
    if max_lambda_split is None:
        max_lambda_split = 0.05 if use_proactive else 0.10  # 硬调 Reactive 到 0.10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}  |  Proactive: {use_proactive}  |  Model: CTDE-GAT-MARL  |  Lambda: migration={max_lambda_migration:.3f}, split={max_lambda_split:.3f}")

    hidden_dim = 64
    encoder = GraphEncoder(node_feat_dim=14, hidden_dim=hidden_dim, output_dim=hidden_dim).to(device)
    actor = SharedNodeActor(embedding_dim=hidden_dim).to(device)
    critic = CentralCritic(embedding_dim=hidden_dim).to(device)
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(actor.parameters()) + list(critic.parameters()),
        lr=3e-4,
    )

    if inference_mode:
        if checkpoint_path is None:
            raise ValueError("inference_mode=True 但未提供 checkpoint_path")
        _load_marl_checkpoint(checkpoint_path, encoder, actor, critic, device)
        num_epochs = 1

    memory = deque(maxlen=10000)
    batch_size = 32
    epsilon = 0.25 if not inference_mode else 0.0
    epsilon_min = 0.02
    epsilon_decay = 0.996

    loss_history = []
    reward_history = []
    epsilon_history = []
    entropy_history = []
    lambda_migration_history = []
    lambda_split_history = []
    dense_distance_bonus_sum = 0.0

    total_decision_time = 0.0
    decision_count = 0
    total_migrations = 0
    total_violations = 0
    proactive_decisions = 0
    total_reward_sum = 0.0
    total_access_latency = 0.0
    total_communication_cost = 0.0
    total_migration_cost = 0.0
    total_cost_ms_sum = 0.0
    total_sla_penalty_ms = 0.0
    total_tearing_penalty_ms = 0.0
    total_future_penalty_ms = 0.0
    local_migration_cost_sum = 0.0
    edge_split_cost_sum = 0.0
    invalid_action_masked_count = 0
    action_mask_fallback_count = 0
    agent_decision_count = 0
    controlled_agent_decision_count = 0
    pinned_agent_decision_count = 0
    stay_action_count = 0
    all_agents_migrated_decisions = 0
    controlled_migrations = 0
    controlled_all_agents_migrated_decisions = 0
    candidate_action_counts = {str(i): 0 for i in range(MARL_ACTION_DIM)}
    joint_action_distribution = defaultdict(int)
    dag_complexity_stats = defaultdict(lambda: {"decision_count": 0, "total_cost_ms_sum": 0.0, "migrations": 0})
    dag_type_stats = defaultdict(
        lambda: {
            "decision_count": 0,
            "total_cost_ms_sum": 0.0,
            "migrations": 0,
            "controlled_migrations": 0,
            "all_agents_migrated_decisions": 0,
            "controlled_all_agents_migrated_decisions": 0,
        }
    )
    dag_proactive_stats = (
        defaultdict(lambda: {"proactive_decisions": 0, "migrated_nodes": 0})
        if collect_dag_proactive_stats else None
    )

    timestamps = sorted(df["date_time"].unique())
    df_grouped = df.groupby("date_time")
    global_step = 0

    for epoch in range(num_epochs):
        is_eval_epoch = inference_mode or (epoch == num_epochs - 1)
        taxi_dag_type = {}
        taxi_dag_assignments = {}
        taxi_last = {}
        if is_eval_epoch:
            total_decision_time = 0.0
            decision_count = 0
            total_migrations = 0
            total_violations = 0
            proactive_decisions = 0
            total_reward_sum = 0.0
            total_access_latency = 0.0
            total_communication_cost = 0.0
            total_migration_cost = 0.0
            total_cost_ms_sum = 0.0
            total_sla_penalty_ms = 0.0
            total_tearing_penalty_ms = 0.0
            total_future_penalty_ms = 0.0
            local_migration_cost_sum = 0.0
            edge_split_cost_sum = 0.0
            dense_distance_bonus_sum = 0.0
            invalid_action_masked_count = 0
            action_mask_fallback_count = 0
            agent_decision_count = 0
            controlled_agent_decision_count = 0
            pinned_agent_decision_count = 0
            stay_action_count = 0
            all_agents_migrated_decisions = 0
            controlled_migrations = 0
            controlled_all_agents_migrated_decisions = 0
            candidate_action_counts = {str(i): 0 for i in range(MARL_ACTION_DIM)}
            joint_action_distribution = defaultdict(int)
            dag_complexity_stats = defaultdict(lambda: {"decision_count": 0, "total_cost_ms_sum": 0.0, "migrations": 0})
            dag_type_stats = defaultdict(
                lambda: {
                    "decision_count": 0,
                    "total_cost_ms_sum": 0.0,
                    "migrations": 0,
                    "controlled_migrations": 0,
                    "all_agents_migrated_decisions": 0,
                    "controlled_all_agents_migrated_decisions": 0,
                }
            )
            dag_proactive_stats = (
                defaultdict(lambda: {"proactive_decisions": 0, "migrated_nodes": 0})
                if collect_dag_proactive_stats else None
            )
            encoder.eval()
            actor.eval()
            critic.eval()
        else:
            encoder.train()
            actor.train()
            critic.train()
        if inference_mode or is_eval_epoch:
            epoch_lm, epoch_ls = max_lambda_migration, max_lambda_split
        else:
            epoch_lm, epoch_ls = lambda_schedule_by_epoch(
                epoch,
                num_epochs,
                max_migration=max_lambda_migration,
                max_split=max_lambda_split,
            )

        pbar = tqdm(total=len(timestamps), desc=f"GAT-MARL Epoch {epoch + 1}/{num_epochs}{' [EVAL]' if is_eval_epoch else ''}")
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
                gateway_node = entry_nodes[0]
                gateway_server = taxi_dag_assignments[taxi_id][gateway_node]
                gw_lat, gw_lon = servers_info[gateway_server]
                gateway_dist = float(haversine_distance(current_lat, current_lon, gw_lat, gw_lon))

                if check_sla_violation(current_lat, current_lon, gw_lat, gw_lon):
                    total_violations += 1

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

                candidates = find_k_nearest_servers(current_lat, current_lon, servers_df, k=3)
                old_assignments = copy.copy(taxi_dag_assignments[taxi_id])
                state = build_marl_graph_state(
                    taxi_id,
                    dag_info,
                    taxi_dag_assignments[taxi_id],
                    servers_info,
                    trigger_type,
                    candidates,
                    current_lat,
                    current_lon,
                    predicted_locations=predicted_locations,
                )
                sorted_nodes = topological_sort(dag_info)
                controlled_nodes = [node for node in sorted_nodes if not is_external_node(node)]
                node_to_idx = {name: i for i, name in enumerate(state["node_names"])}
                actions = []

                t0 = time.perf_counter()
                with torch.no_grad():
                    tensors = _state_to_tensors(state, device)
                    embeddings = encoder(
                        tensors["node_features"],
                        tensors["adj_matrix"],
                        tensors["trigger_context"],
                        tensors["mobility_context"],
                        tensors["candidate_features"],
                    )
                    logits = actor(embeddings, tensors["trigger_context"])
                    masked_logits, masked_count, fallback_count = _apply_action_mask(
                        logits, tensors["action_masks"]
                    )
                    invalid_action_masked_count += masked_count
                    action_mask_fallback_count += fallback_count
                    probs = F.softmax(masked_logits, dim=-1)

                    for ms_node in sorted_nodes:
                        node_idx = node_to_idx[ms_node]
                        if is_external_node(ms_node):
                            action = 0
                        elif (not is_eval_epoch) and random.random() < epsilon:
                            legal = torch.nonzero(tensors["action_masks"][node_idx], as_tuple=False).flatten()
                            action = int(legal[torch.randint(0, len(legal), (1,), device=device)].item())
                        else:
                            action = int(torch.argmax(probs[node_idx]).item())
                        actions.append(action)
                        current_server = taxi_dag_assignments[taxi_id][ms_node]
                        target_server = (
                            current_server
                            if is_external_node(ms_node)
                            else action_to_server(action, candidates, current_server)
                        )
                        taxi_dag_assignments[taxi_id][ms_node] = target_server

                total_decision_time += time.perf_counter() - t0
                agent_decision_count += len(actions)
                controlled_agent_decision_count += len(controlled_nodes)
                pinned_agent_decision_count += len(actions) - len(controlled_nodes)
                stay_action_count += sum(1 for action in actions if int(action) == 0)
                for action in actions:
                    key = str(int(action))
                    candidate_action_counts[key] = candidate_action_counts.get(key, 0) + 1
                joint_action_distribution["-".join(map(str, actions))] += 1

                lm, ls = epoch_lm, epoch_ls
                lambda_migration_history.append(lm)
                lambda_split_history.append(ls)
                shared_reward, agent_rewards, details = calculate_marl_rewards(
                    taxi_id,
                    dag_info,
                    taxi_dag_assignments[taxi_id],
                    old_assignments,
                    (current_lat, current_lon),
                    servers_info,
                    predicted_locations=predicted_locations,
                    trigger_type=trigger_type,
                    lambda_migration=lm,
                    lambda_split=ls,
                    dense_distance_bonus_max=8.0 if use_proactive else 0.0,  # 增加 Proactive 激励
                )
                del agent_rewards
                total_reward_sum += shared_reward
                reward_history.append(shared_reward)
                total_access_latency += details["access_latency"]
                total_communication_cost += details["communication_cost"]
                total_migration_cost += details["migration_cost"]
                total_cost_ms_sum += details["total_cost_ms"]
                total_sla_penalty_ms += details.get("sla_penalty_ms", 0.0)
                total_tearing_penalty_ms += details.get("tearing_penalty_ms", details.get("tearing_penalty", 0.0))
                total_future_penalty_ms += details.get("future_penalty_ms", details.get("future_penalty", 0.0))
                local_migration_cost_sum += details.get("local_migration_cost_sum", 0.0)
                edge_split_cost_sum += details.get("edge_split_cost_sum", 0.0)
                dense_distance_bonus_sum += details.get("dense_distance_bonus_sum", 0.0)

                nodes_migrated = sum(
                    1 for n in sorted_nodes
                    if old_assignments[n] != taxi_dag_assignments[taxi_id][n]
                )
                controlled_nodes_migrated = sum(
                    1 for n in controlled_nodes
                    if old_assignments[n] != taxi_dag_assignments[taxi_id][n]
                )
                total_migrations += nodes_migrated
                controlled_migrations += controlled_nodes_migrated
                if nodes_migrated == len(actions) and actions:
                    all_agents_migrated_decisions += 1
                if (
                    controlled_nodes
                    and controlled_nodes_migrated == len(controlled_nodes)
                ):
                    controlled_all_agents_migrated_decisions += 1
                if dag_proactive_stats is not None and use_proactive and trigger_type == TRIGGER_PROACTIVE:
                    dag_proactive_stats[dag_type]["proactive_decisions"] += 1
                    dag_proactive_stats[dag_type]["migrated_nodes"] += controlled_nodes_migrated
                ckey = _dag_complexity_key(dag_info)
                dag_complexity_stats[ckey]["decision_count"] += 1
                dag_complexity_stats[ckey]["total_cost_ms_sum"] += details["total_cost_ms"]
                dag_complexity_stats[ckey]["migrations"] += nodes_migrated
                dag_type_stats[dag_type]["decision_count"] += 1
                dag_type_stats[dag_type]["total_cost_ms_sum"] += details["total_cost_ms"]
                dag_type_stats[dag_type]["migrations"] += nodes_migrated
                dag_type_stats[dag_type]["controlled_migrations"] += controlled_nodes_migrated
                if nodes_migrated == len(actions) and actions:
                    dag_type_stats[dag_type]["all_agents_migrated_decisions"] += 1
                if controlled_nodes and controlled_nodes_migrated == len(controlled_nodes):
                    dag_type_stats[dag_type]["controlled_all_agents_migrated_decisions"] += 1

                if not is_eval_epoch:
                    agent_reward_list = [
                        details["agent_rewards"][node]
                        for node in state["node_names"]
                    ]
                    memory.append({
                        "state": state,
                        "actions": actions,
                        "shared_reward": shared_reward,
                        "training_reward": details.get("training_reward", shared_reward),
                        "agent_reward_list": agent_reward_list,
                    })
                    info = _optimize_marl(memory, encoder, actor, critic, optimizer, device, batch_size=batch_size)
                    if info:
                        loss_history.append(info["loss"])
                        entropy_history.append(info["entropy"])
                        epsilon = max(epsilon_min, epsilon * epsilon_decay)
                    global_step += 1
                epsilon_history.append(epsilon)
                touch_taxi_last(taxi_last, taxi_id, row, current_lon, current_lat, ts)
            pbar.update(1)
        pbar.close()

    if (not inference_mode) and save_checkpoint_path:
        _save_marl_checkpoint(save_checkpoint_path, encoder, actor, critic)

    avg_ms = (total_decision_time / decision_count * 1000.0) if decision_count > 0 else 0.0
    avg_agents = (agent_decision_count / decision_count) if decision_count > 0 else 0.0
    avg_controlled_agents = (
        controlled_agent_decision_count / decision_count if decision_count > 0 else 0.0
    )
    avg_pinned_agents = (
        pinned_agent_decision_count / decision_count if decision_count > 0 else 0.0
    )
    avg_migrated_agents = (total_migrations / decision_count) if decision_count > 0 else 0.0
    avg_controlled_migrated_agents = (
        controlled_migrations / decision_count if decision_count > 0 else 0.0
    )
    all_agents_migrated_ratio = (
        all_agents_migrated_decisions / decision_count if decision_count > 0 else 0.0
    )
    controlled_all_agents_migrated_ratio = (
        controlled_all_agents_migrated_decisions / decision_count if decision_count > 0 else 0.0
    )
    stay_action_ratio = stay_action_count / agent_decision_count if agent_decision_count > 0 else 0.0
    return {
        "total_migrations": total_migrations,
        "total_violations": total_violations,
        "proactive_decisions": proactive_decisions,
        "decision_count": decision_count,
        "total_reward": total_reward_sum,
        "total_access_latency": total_access_latency,
        "total_communication_cost": total_communication_cost,
        "total_migration_cost": total_migration_cost,
        "total_cost_ms_sum": total_cost_ms_sum,
        "total_sla_penalty_ms": total_sla_penalty_ms,
        "total_tearing_penalty_ms": total_tearing_penalty_ms,
        "total_future_penalty_ms": total_future_penalty_ms,
        "loss_history": loss_history,
        "reward_history": reward_history,
        "epsilon_history": epsilon_history,
        "entropy_history": entropy_history,
        "lambda_migration_history": lambda_migration_history,
        "lambda_split_history": lambda_split_history,
        "total_decision_time": total_decision_time,
        "decision_count_for_latency": decision_count,
        "avg_decision_time_ms": avg_ms,
        "avg_agents_per_decision": avg_agents,
        "controlled_agents_per_decision": avg_controlled_agents,
        "pinned_agents_per_decision": avg_pinned_agents,
        "avg_migrated_agents_per_decision": avg_migrated_agents,
        "avg_controlled_migrated_agents_per_decision": avg_controlled_migrated_agents,
        "controlled_migrations": controlled_migrations,
        "all_agents_migrated_decisions": all_agents_migrated_decisions,
        "all_agents_migrated_ratio": all_agents_migrated_ratio,
        "controlled_all_agents_migrated_decisions": controlled_all_agents_migrated_decisions,
        "controlled_all_agents_migrated_ratio": controlled_all_agents_migrated_ratio,
        "stay_action_ratio": stay_action_ratio,
        "candidate_action_counts": dict(candidate_action_counts),
        "joint_action_distribution": dict(joint_action_distribution),
        "local_migration_cost_sum": local_migration_cost_sum,
        "edge_split_cost_sum": edge_split_cost_sum,
        "dense_distance_bonus_sum": dense_distance_bonus_sum,
        "cost_by_dag_complexity": {
            k: {
                **v,
                "avg_total_cost_ms": (
                    v["total_cost_ms_sum"] / v["decision_count"]
                    if v["decision_count"] > 0 else 0.0
                ),
            }
            for k, v in dag_complexity_stats.items()
        },
        "cost_by_dag_type": {
            k: {
                **v,
                "avg_total_cost_ms": (
                    v["total_cost_ms_sum"] / v["decision_count"]
                    if v["decision_count"] > 0 else 0.0
                ),
                "all_agents_migrated_ratio": (
                    v["all_agents_migrated_decisions"] / v["decision_count"]
                    if v["decision_count"] > 0 else 0.0
                ),
                "controlled_all_agents_migrated_ratio": (
                    v["controlled_all_agents_migrated_decisions"] / v["decision_count"]
                    if v["decision_count"] > 0 else 0.0
                ),
            }
            for k, v in dag_type_stats.items()
        },
        "migrations_by_dag_type": {
            k: v["migrations"] for k, v in dag_type_stats.items()
        },
        "controlled_migrations_by_dag_type": {
            k: v["controlled_migrations"] for k, v in dag_type_stats.items()
        },
        "invalid_action_masked_count": invalid_action_masked_count,
        "action_mask_fallback_count": action_mask_fallback_count,
        "lambda_migration": lambda_migration_history[-1] if lambda_migration_history else 0.0,
        "lambda_split": lambda_split_history[-1] if lambda_split_history else 0.0,
        "dag_proactive_migration_stats": dict(dag_proactive_stats or {}),
    }
