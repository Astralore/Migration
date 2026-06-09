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

from core.context import (
    DISTANCE_THRESHOLD_KM,
    TRIGGER_PROACTIVE,
    TRIGGER_REACTIVE,
    get_trigger_type,
    check_sla_violation,
)
from core.dag_utils import (
    assign_dag_type,
    get_deployable_nodes,
    get_service_entry_nodes,
    initialize_dag_assignment,
    is_external_node,
    topological_sort,
)
from core.geo import find_k_nearest_servers, haversine_distance
from core.marl_reward import (
    calculate_marl_rewards,
    lambda_schedule_by_epoch,
    reactive_dense_bonus_enabled,
    train_total_cost_enabled,
)
from core.reward_curriculum import (
    apply_curriculum_for_epoch,
    curriculum_enabled,
    internal_gamma_curriculum_enabled,
    reset_curriculum,
)
from core.marl_state_builder import (
    MARL_ACTION_DIM,
    MARL_CANDIDATE_FEATURE_DIM,
    MARL_NODE_FEAT_DIM,
    MARL_TRIGGER_CONTEXT_DIM,
    action_to_server,
    build_marl_graph_state,
)
from core.microservice_dags import MICROSERVICE_DAGS
from core.reward import (
    BASE_MIGRATION_OVERHEAD_MS,
    EDGE_BACKHAUL_MBPS,
    MAX_BW_MBPS,
    MAX_TEARING_MB,
    MB_TO_MBIT,
    MIN_BW_MBPS,
    REACTIVE_MIGRATION_MULT,
    RPC_SIZE_MB,
    SLA_PENALTY_PER_KM_MS,
    build_servers_info,
    calculate_entry_sla_metrics,
    calculate_nonlinear_migration_cost_ms,
    dag_current_sla_violation,
    _entry_access_profile,
    calc_access_latency_ms,
    calculate_sla_penalty_ms,
    compute_internal_critical_path_ms,
    estimate_dag_migration_time_s,
    future_mean_excess_penalty_gain_ms,
    is_reward_v2,
    REWARD_V2_OBJECTIVE_SCALE_MS,
    effective_reward_v2_objective_scale_ms,
    sla_penalty_gain_ms,
    use_reward_v2_internal_path,
    edge_effective_latency_ms,
)
from core.physics_utils import BASE_ROUTER_DELAY_MS, FIBER_SPEED_KM_MS
from prediction.simple_predictor import build_predict_future_time_kwargs, touch_taxi_last


FORECAST_HORIZON = 15
MAX_NODES = 12
CF_SLA_WEIGHT = 0.002
CF_FUTURE_WEIGHT = 0.0015
CF_TOPOLOGY_WEIGHT = 0.001
CF_COST_SCALE_MS = 1000.0
# When True, CF gate uses train-only score floor (MARL_CF_TRAIN_SCORE_FLOOR).
_cf_gate_use_train_score_floor = False


def _env_flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


def _cf_cost_scale_ms():
    """Align counterfactual migration normalization with v2 reward scale."""
    return effective_reward_v2_objective_scale_ms() if is_reward_v2() else CF_COST_SCALE_MS
CF_SCORE_EPS = 1e-6
CF_SLA_ENTRY_SCORE_FLOOR = -0.2
PROACTIVE_MAX_MIGRATABLE_MB = 100.0
STATEFUL_FUTURE_GAIN_DISCOUNT = 0.1
PROACTIVE_HEAVY_SLA_GAIN_FLOOR_MS = 20000.0
LIGHTWEIGHT_ENTRY_SCORE_FLOOR = -0.2
LIGHTWEIGHT_ENTRY_MIN_SLA_GAIN_MS = 2500.0
LIGHTWEIGHT_ENTRY_MIN_BIAS = 0.2


def _env_float(name, default):
    raw = os.environ.get(name)
    return float(raw) if raw is not None and raw != "" else default


def _env_int(name, default):
    raw = os.environ.get(name)
    return int(raw) if raw is not None and raw != "" else default


# Overridable via env for ablations (e.g. phase C0: MAX=2, budget=6000).
PROACTIVE_MIGRATION_BUDGET_MS = _env_float("PROACTIVE_MIGRATION_BUDGET_MS", 3000.0)
PROACTIVE_MAX_MIGRATIONS_PER_DECISION = _env_int("PROACTIVE_MAX_MIGRATIONS_PER_DECISION", 1)
MIN_SLA_GAIN_COST_RATIO = 0.30
HIGH_COST_MIGRATION_MS = 3000.0
HIGH_COST_MIN_SLA_GAIN_MS = 5000.0
REACTIVE_MAX_MIGRATIONS_PER_DECISION = _env_int("REACTIVE_MAX_MIGRATIONS_PER_DECISION", 1)


def _unlimited_migration_cap(cap_setting):
    """Non-positive env value disables per-step count cap (budget / ROI still apply)."""
    return int(cap_setting) <= 0


def _reactive_migration_cap(controlled_count, violating_entry_count):
    if _unlimited_migration_cap(REACTIVE_MAX_MIGRATIONS_PER_DECISION):
        return max(1, int(controlled_count))
    return min(
        REACTIVE_MAX_MIGRATIONS_PER_DECISION,
        controlled_count,
        max(1, int(violating_entry_count)),
    )


def _passthrough_guard_stats():
    """Placeholder stats when v2 bypasses rule-based guards (soft reward only)."""
    return {
        "counterfactual_score_sum": 0.0,
        "sla_improving_action_count": 0,
        "cost_guard_blocked_count": 0,
        "non_entry_distance_only_blocked_count": 0,
    }


def _bypass_rule_based_guards():
    """Reward v2/v2.1: bypass rule guards unless B1.1 CF gate is active."""
    if _use_cf_sla_gain_gate():
        return False
    return is_reward_v2()


def _use_p1_coordination():
    """
    P1: SA-aligned action space — entry-first masks + at most one migration per decision.

    Enabled when MARL_P1=1 (default for Reward v2.1 internal-path runs).
    """
    raw = os.environ.get("MARL_P1")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().lower() not in ("0", "false", "no")
    return is_reward_v2() and use_reward_v2_internal_path()


def _apply_p1_max_one_migration(actions, sorted_nodes, node_to_idx, probs, entry_nodes):
    """Keep a single migration per joint decision; entry nodes win ties."""
    migrate_slots = []
    for i, (node, action) in enumerate(zip(sorted_nodes, actions)):
        if int(action) > 0 and not is_external_node(node):
            migrate_slots.append((i, node, int(action)))
    if len(migrate_slots) <= 1:
        return list(actions), 0

    entry_set = set(entry_nodes or [])
    scored = []
    for i, node, action in migrate_slots:
        idx = node_to_idx[node]
        score = float(probs[idx, action].item()) if probs is not None else 0.0
        if node in entry_set:
            score += 1000.0
        scored.append((score, i))
    scored.sort(key=lambda item: -item[0])
    keep_i = scored[0][1]

    new_actions = list(actions)
    suppressed = 0
    for i, node, _action in migrate_slots:
        if i != keep_i:
            new_actions[i] = 0
            suppressed += 1
    return new_actions, suppressed


def _use_soft_cf_bias():
    """
    P2: soft counterfactual logit nudge (v2 bypass path).

    Default off when CF gate is on (B1.1b); otherwise on under v2 bypass.
    """
    if _use_cf_sla_gain_gate() and os.environ.get("MARL_SOFT_CF_BIAS") is None:
        return False
    raw = os.environ.get("MARL_SOFT_CF_BIAS")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().lower() not in ("0", "false", "no")
    return is_reward_v2() and _bypass_rule_based_guards()


def _soft_cf_bias_scales():
    pos_scale = _env_float("MARL_SOFT_CF_BIAS_SCALE", 0.06)
    neg_scale = _env_float("MARL_SOFT_CF_NEG_BIAS_SCALE", 0.04)
    max_pos = _env_float("MARL_SOFT_CF_MAX_BIAS", 0.10)
    max_neg = _env_float("MARL_SOFT_CF_MAX_NEG_BIAS", 0.08)
    return pos_scale, neg_scale, max_pos, max_neg


def _apply_soft_counterfactual_bias(
    masked_logits,
    sorted_nodes,
    node_to_idx,
    action_masks,
    assignments,
    candidates,
    user_lat,
    user_lon,
    servers_info,
    dag_info,
    predicted_locations,
    lambda_migration,
    lambda_split,
    trigger_type,
):
    """Light logit prior/penalty from counterfactual ROI (no hard clipping)."""
    pos_scale, neg_scale, max_pos, max_neg = _soft_cf_bias_scales()
    biased_logits = masked_logits.clone()
    bias_count = 0
    neg_bias_count = 0
    best_action_counts = defaultdict(int)
    score_sum = 0.0
    sla_improving_count = 0
    cost_guard_blocked_count = 0
    non_entry_distance_only_blocked_count = 0
    cf_ctx = _build_counterfactual_context(
        dag_info,
        assignments,
        candidates,
        user_lat,
        user_lon,
        servers_info,
        predicted_locations=predicted_locations,
        trigger_type=trigger_type,
    )

    for node in sorted_nodes:
        if is_external_node(node):
            continue
        node_idx = node_to_idx[node]
        best_action = 0
        best_score = -float("inf")
        for action in range(1, MARL_ACTION_DIM):
            if not bool(action_masks[node_idx, action].item()):
                continue
            scored = _score_counterfactual_action(
                dag_info,
                node,
                action,
                assignments,
                candidates,
                servers_info,
                cf_ctx,
                lambda_migration,
                lambda_split,
            )
            if scored is None:
                continue
            score_sum += scored["score"]
            if scored["sla_gain_ms"] > 0.0:
                sla_improving_count += 1
            if scored["non_entry_distance_only"] and scored["score"] <= CF_SCORE_EPS:
                non_entry_distance_only_blocked_count += 1
            if scored["score"] > CF_SCORE_EPS:
                bias = min(max_pos, pos_scale * scored["score"])
                biased_logits[node_idx, action] = biased_logits[node_idx, action] + float(bias)
                bias_count += 1
            elif scored["score"] < -CF_SCORE_EPS:
                penalty = min(max_neg, neg_scale * abs(scored["score"]))
                biased_logits[node_idx, action] = biased_logits[node_idx, action] - float(penalty)
                neg_bias_count += 1
            if scored["score"] > best_score:
                best_score = scored["score"]
                best_action = action
        best_action_counts[str(best_action)] += 1

    return biased_logits, bias_count, dict(best_action_counts), {
        "counterfactual_score_sum": float(score_sum),
        "sla_improving_action_count": int(sla_improving_count),
        "cost_guard_blocked_count": int(cost_guard_blocked_count),
        "non_entry_distance_only_blocked_count": int(non_entry_distance_only_blocked_count),
        "soft_cf_neg_bias_count": int(neg_bias_count),
    }


def _node_transfer_mb(dag_info, node):
    props = dag_info["nodes"].get(node, {})
    return float(props.get("image_mb", 0.0)) + float(props.get("state_mb", 0.0))


def _node_state_mb(dag_info, node):
    return float(dag_info["nodes"].get(node, {}).get("state_mb", 0.0))


def _is_heavy_for_proactive(dag_info, node):
    return _node_transfer_mb(dag_info, node) > PROACTIVE_MAX_MIGRATABLE_MB


def _is_lightweight_entry_rescue(dag_info, node, cf_ctx):
    return (
        node in cf_ctx["entry_nodes"]
        and _node_state_mb(dag_info, node) <= 0.0
    )


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


def _use_sla_violation_gate():
    """
    B1: hard action gate — when current DAG has no SLA violation, only STAY is legal.

    Default off when B1.1a CF gate is on; disable with ``MARL_SLA_VIOLATION_GATE=0``.
    """
    if _use_cf_sla_gain_gate() and os.environ.get("MARL_SLA_VIOLATION_GATE") is None:
        return False
    raw = os.environ.get("MARL_SLA_VIOLATION_GATE")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().lower() not in ("0", "false", "no")
    return is_reward_v2()


def _use_cf_sla_gain_gate():
    """
    B1.1: hard CF action gate before actor sampling.

    Default on for Reward v2/v2.1; disable with ``MARL_CF_SLA_GAIN_GATE=0``.
    """
    raw = os.environ.get("MARL_CF_SLA_GAIN_GATE")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().lower() not in ("0", "false", "no")
    return is_reward_v2()


def _cf_gate_mode():
    """
    CF gate criterion: ``score`` (B1.1b, default) or ``sla_gain`` (B1.1a legacy).
    """
    mode = os.environ.get("MARL_CF_GATE_MODE", "score").strip().lower()
    return mode if mode in ("score", "sla_gain") else "score"


def _cf_physical_migration_cost_enabled():
    """E: CF/gate scores migration at 1.0x; reported total_cost_ms keeps REACTIVE_MIGRATION_MULT."""
    return _env_flag("MARL_CF_PHYSICAL_MIGRATION_COST", "1")


def _cf_migration_mult_for_scoring(trigger_type):
    if trigger_type == TRIGGER_PROACTIVE:
        return 1.0
    if _cf_physical_migration_cost_enabled():
        return 1.0
    return REACTIVE_MIGRATION_MULT


def _cf_score_floor():
    """Minimum counterfactual score for non-rescue migrate slots (B1.1b)."""
    if _cf_gate_use_train_score_floor:
        train_floor = os.environ.get("MARL_CF_TRAIN_SCORE_FLOOR", "").strip()
        if train_floor:
            return float(train_floor)
    return _env_float("MARL_CF_SCORE_FLOOR", CF_SCORE_EPS)


def _entry_distance_violating(node, cf_ctx):
    """True only when entry distance exceeds SLA threshold."""
    return node in cf_ctx["violating_entries"]


def _cf_migrate_action_allowed(scored, node, cf_ctx, *, gate_mode=None, action=None):
    """Whether a migrate slot passes the CF gate (B1.1b tightened)."""
    if scored is None:
        return False
    gate_mode = gate_mode or _cf_gate_mode()
    if gate_mode == "sla_gain":
        return float(scored["sla_gain_ms"]) > CF_SCORE_EPS

    is_entry = node in cf_ctx["entry_nodes"]
    score = float(scored["score"])
    sla_gain = float(scored["sla_gain_ms"])
    score_floor = _cf_score_floor()

    # B1.1b: truly violating entry → nearest candidate (action=1) rescue only
    if (
        gate_mode == "score"
        and is_entry
        and _entry_distance_violating(node, cf_ctx)
        and action is not None
        and int(action) == 1
    ):
        return True

    if scored.get("non_entry_distance_only") and score <= score_floor:
        return False

    lightweight_rescue = (
        scored.get("is_lightweight_entry_rescue")
        and sla_gain >= LIGHTWEIGHT_ENTRY_MIN_SLA_GAIN_MS
        and score > LIGHTWEIGHT_ENTRY_SCORE_FLOOR
    )
    if lightweight_rescue:
        return True

    # General: positive SLA gain and score above floor
    if sla_gain > 0.0 and score > score_floor:
        return True

    return False


def _apply_sla_violation_action_gate(action_masks):
    """
    Force stay-only masks when SLA is green. Returns (new_masks, agents_forced_stay).
    """
    mask = action_masks.clone()
    agents_forced = 0
    for i in range(mask.shape[0]):
        if bool(mask[i, 1:].any().item()):
            mask[i, 1:] = False
            mask[i, 0] = True
            agents_forced += 1
    return mask, agents_forced


def _apply_cf_sla_gain_action_gate(
    action_masks,
    sorted_nodes,
    node_to_idx,
    dag_info,
    assignments,
    candidates,
    user_lat,
    user_lon,
    servers_info,
    trigger_type,
    lambda_migration,
    lambda_split,
):
    """
    B1.1: mask migrate slots failing CF gate (B1.1b ``score`` mode by default).
    Entry violating nodes may keep one best sla-improving candidate.
    Returns (new_masks, stats_dict).
    """
    mask = action_masks.clone()
    gate_mode = _cf_gate_mode()
    cf_ctx = _build_counterfactual_context(
        dag_info,
        assignments,
        candidates,
        user_lat,
        user_lon,
        servers_info,
        predicted_locations=None,
        trigger_type=trigger_type,
    )
    blocked_slots = 0
    improving_slots = 0
    forced_stay_agents = 0
    entry_rescue_slots = 0
    nodes_without_migrate = []

    for node in sorted_nodes:
        if is_external_node(node):
            continue
        node_idx = node_to_idx.get(node)
        if node_idx is None:
            continue
        kept_migrate = False
        for action in range(1, MARL_ACTION_DIM):
            if not bool(mask[node_idx, action].item()):
                continue
            scored = _score_counterfactual_action(
                dag_info,
                node,
                action,
                assignments,
                candidates,
                servers_info,
                cf_ctx,
                lambda_migration,
                lambda_split,
            )
            if _cf_migrate_action_allowed(
                scored, node, cf_ctx, gate_mode=gate_mode, action=action
            ):
                improving_slots += 1
                kept_migrate = True
            else:
                mask[node_idx, action] = False
                blocked_slots += 1
        if not kept_migrate:
            nodes_without_migrate.append(node)
            mask[node_idx, 0] = True
            forced_stay_agents += 1

    # B1.1b: entry rescue — nearest candidate for distance-violating entries only
    if gate_mode == "score":
        for node in sorted(
            cf_ctx["violating_entries"],
            key=lambda n: cf_ctx["entry_distances"].get(n, 0.0),
            reverse=True,
        ):
            if node not in nodes_without_migrate:
                continue
            node_idx = node_to_idx.get(node)
            if node_idx is None:
                continue
            if bool(action_masks[node_idx, 1].item()):
                mask[node_idx, 1] = True
                improving_slots += 1
                entry_rescue_slots += 1
                forced_stay_agents = max(0, forced_stay_agents - 1)
                nodes_without_migrate.remove(node)
                continue
            best_entry = None
            for action in range(1, MARL_ACTION_DIM):
                if not bool(action_masks[node_idx, action].item()):
                    continue
                scored = _score_counterfactual_action(
                    dag_info,
                    node,
                    action,
                    assignments,
                    candidates,
                    servers_info,
                    cf_ctx,
                    lambda_migration,
                    lambda_split,
                )
                if _cf_migrate_action_allowed(
                    scored, node, cf_ctx, gate_mode=gate_mode, action=action
                ):
                    score = float(scored["score"]) if scored else 0.0
                    if best_entry is None or score > best_entry[0]:
                        best_entry = (score, action)
            if best_entry is not None:
                _score, action = best_entry
                mask[node_idx, action] = True
                improving_slots += 1
                entry_rescue_slots += 1
                forced_stay_agents = max(0, forced_stay_agents - 1)
                nodes_without_migrate.remove(node)

    return mask, {
        "cf_gate_blocked_slots": blocked_slots,
        "cf_gate_improving_slots": improving_slots,
        "cf_gate_forced_stay_agents": forced_stay_agents,
        "cf_gate_entry_rescue_slots": entry_rescue_slots,
        "cf_gate_mode": gate_mode,
    }


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


def _load_marl_checkpoint(path, encoder, actor, critic, device, *, strict=True):
    ckpt = torch.load(path, map_location=device)
    enc_res = encoder.load_state_dict(ckpt["encoder"], strict=strict)
    act_res = actor.load_state_dict(ckpt["actor"], strict=strict)
    crt_res = critic.load_state_dict(ckpt["critic"], strict=strict)
    if not strict:
        print(
            f"  [GAT-MARL LOAD] Partial load from {path} "
            f"(enc_missing={len(enc_res.missing_keys)} "
            f"act_missing={len(act_res.missing_keys)} "
            f"crt_missing={len(crt_res.missing_keys)})"
        )
    else:
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


def _new_epoch_counters():
    """Per-epoch behavioral counters; reset at the start of each MARL epoch."""
    return {
        "total_decision_time": 0.0,
        "decision_count": 0,
        "total_migrations": 0,
        "total_violations": 0,
        "primary_entry_violations": 0,
        "max_entry_violations": 0,
        "severe_sla_violations": 0,
        "total_sla_excess_distance_km": 0.0,
        "sla_excess_distance_history": [],
        "proactive_decisions": 0,
        "total_reward_sum": 0.0,
        "total_access_latency": 0.0,
        "total_communication_cost": 0.0,
        "total_internal_path_ms": 0.0,
        "total_migration_cost": 0.0,
        "migration_decision_count": 0,
        "total_cost_ms_sum": 0.0,
        "total_sla_penalty_ms": 0.0,
        "total_tearing_penalty_ms": 0.0,
        "total_future_penalty_ms": 0.0,
        "local_migration_cost_sum": 0.0,
        "edge_split_cost_sum": 0.0,
        "dense_distance_bonus_sum": 0.0,
        "entry_sla_bonus_sum": 0.0,
        "proactive_logit_bias_count": 0,
        "proactive_best_bias_action_counts": {str(i): 0 for i in range(MARL_ACTION_DIM)},
        "reactive_action_clipped_count": 0,
        "counterfactual_score_sum": 0.0,
        "entry_node_migration_count": 0,
        "sla_improving_action_count": 0,
        "cost_guard_blocked_count": 0,
        "non_entry_distance_only_blocked_count": 0,
        "invalid_action_masked_count": 0,
        "action_mask_fallback_count": 0,
        "agent_decision_count": 0,
        "controlled_agent_decision_count": 0,
        "pinned_agent_decision_count": 0,
        "stay_action_count": 0,
        "all_agents_migrated_decisions": 0,
        "controlled_migrations": 0,
        "controlled_all_agents_migrated_decisions": 0,
        "candidate_action_counts": {str(i): 0 for i in range(MARL_ACTION_DIM)},
        "joint_action_distribution": defaultdict(int),
        "dag_complexity_stats": defaultdict(
            lambda: {"decision_count": 0, "total_cost_ms_sum": 0.0, "migrations": 0}
        ),
        "dag_type_stats": defaultdict(
            lambda: {
                "decision_count": 0,
                "total_cost_ms_sum": 0.0,
                "migrations": 0,
                "controlled_migrations": 0,
                "all_agents_migrated_decisions": 0,
                "controlled_all_agents_migrated_decisions": 0,
            }
        ),
        "dag_proactive_stats": defaultdict(lambda: {"proactive_decisions": 0, "migrated_nodes": 0}),
        "mask_only_stay_agent_count": 0,
        "agents_with_migrate_option_count": 0,
        "legal_migrate_slots_sum": 0,
        "decisions_with_migrate_option_count": 0,
        "epsilon_greedy_samples": 0,
        "argmax_samples": 0,
        "p1_multi_migration_suppressed_count": 0,
        "sla_gate_no_violation_decisions": 0,
        "sla_gate_violation_decisions": 0,
        "sla_gate_forced_stay_agents": 0,
        "cf_gate_blocked_slots": 0,
        "cf_gate_improving_slots": 0,
        "cf_gate_forced_stay_agents": 0,
        "decisions_with_cf_improve_option_count": 0,
    }


def _snapshot_epoch_stats(epoch_index, is_eval_epoch, epsilon, counters, *, loss_history, entropy_history):
    """Compact per-epoch summary for exploration vs eval diagnostics."""
    dc = int(counters["decision_count"])
    adc = int(counters["agent_decision_count"])
    sorted_excess = sorted(counters["sla_excess_distance_history"])
    p95_idx = int(0.95 * (len(sorted_excess) - 1)) if sorted_excess else 0
    mask_only = int(counters["mask_only_stay_agent_count"])
    return {
        "epoch": int(epoch_index),
        "phase": "eval" if is_eval_epoch else "train",
        "is_eval_epoch": bool(is_eval_epoch),
        "epsilon_end": float(epsilon),
        "decision_count": dc,
        "total_migrations": int(counters["total_migrations"]),
        "stay_action_ratio": (counters["stay_action_count"] / adc) if adc > 0 else 0.0,
        "candidate_action_counts": dict(counters["candidate_action_counts"]),
        "cost_guard_blocked_count": int(counters["cost_guard_blocked_count"]),
        "reactive_action_clipped_count": int(counters["reactive_action_clipped_count"]),
        "epsilon_greedy_samples": int(counters["epsilon_greedy_samples"]),
        "argmax_samples": int(counters["argmax_samples"]),
        "mask_only_stay_agent_count": mask_only,
        "mask_only_stay_agent_ratio": (mask_only / adc) if adc > 0 else 0.0,
        "agents_with_migrate_option_count": int(counters["agents_with_migrate_option_count"]),
        "decisions_with_migrate_option_count": int(counters["decisions_with_migrate_option_count"]),
        "decisions_with_migrate_option_ratio": (
            counters["decisions_with_migrate_option_count"] / dc if dc > 0 else 0.0
        ),
        "avg_legal_migrate_actions_per_decision": (
            counters["legal_migrate_slots_sum"] / dc if dc > 0 else 0.0
        ),
        "avg_total_cost_ms": (counters["total_cost_ms_sum"] / dc) if dc > 0 else 0.0,
        "severe_sla_violations": int(counters["severe_sla_violations"]),
        "p95_sla_excess_distance_km": sorted_excess[p95_idx] if sorted_excess else 0.0,
        "avg_loss": float(np.mean(loss_history)) if loss_history else None,
        "avg_entropy": float(np.mean(entropy_history)) if entropy_history else None,
        "p1_multi_migration_suppressed_count": int(counters.get("p1_multi_migration_suppressed_count", 0)),
        "soft_cf_bias_count": int(counters.get("soft_cf_bias_count", 0)),
        "soft_cf_neg_bias_count": int(counters.get("soft_cf_neg_bias_count", 0)),
        "sla_gate_no_violation_decisions": int(counters.get("sla_gate_no_violation_decisions", 0)),
        "sla_gate_violation_decisions": int(counters.get("sla_gate_violation_decisions", 0)),
        "sla_gate_forced_stay_agents": int(counters.get("sla_gate_forced_stay_agents", 0)),
        "cf_gate_blocked_slots": int(counters.get("cf_gate_blocked_slots", 0)),
        "cf_gate_improving_slots": int(counters.get("cf_gate_improving_slots", 0)),
        "cf_gate_forced_stay_agents": int(counters.get("cf_gate_forced_stay_agents", 0)),
        "decisions_with_cf_improve_option_count": int(
            counters.get("decisions_with_cf_improve_option_count", 0)
        ),
        "reward_curriculum": counters.get("reward_curriculum"),
    }


def _build_marl_return_dict(
    counters,
    *,
    loss_history,
    reward_history,
    epsilon_history,
    entropy_history,
    lambda_migration_history,
    lambda_split_history,
    epoch_stats,
    collect_dag_proactive_stats=False,
):
    """Build result dict from counters (top-level = final epoch, usually eval)."""
    decision_count = int(counters["decision_count"])
    agent_decision_count = int(counters["agent_decision_count"])
    dag_complexity_stats = counters["dag_complexity_stats"]
    dag_type_stats = counters["dag_type_stats"]
    dag_proactive_stats = counters["dag_proactive_stats"]
    sla_excess_distance_history = counters["sla_excess_distance_history"]
    candidate_action_counts = counters["candidate_action_counts"]
    joint_action_distribution = counters["joint_action_distribution"]
    proactive_best_bias_action_counts = counters["proactive_best_bias_action_counts"]

    avg_ms = (
        counters["total_decision_time"] / decision_count * 1000.0 if decision_count > 0 else 0.0
    )
    avg_agents = agent_decision_count / decision_count if decision_count > 0 else 0.0
    avg_controlled_agents = (
        counters["controlled_agent_decision_count"] / decision_count if decision_count > 0 else 0.0
    )
    avg_pinned_agents = (
        counters["pinned_agent_decision_count"] / decision_count if decision_count > 0 else 0.0
    )
    avg_migrated_agents = (
        counters["total_migrations"] / decision_count if decision_count > 0 else 0.0
    )
    avg_controlled_migrated_agents = (
        counters["controlled_migrations"] / decision_count if decision_count > 0 else 0.0
    )
    all_agents_migrated_ratio = (
        counters["all_agents_migrated_decisions"] / decision_count if decision_count > 0 else 0.0
    )
    controlled_all_agents_migrated_ratio = (
        counters["controlled_all_agents_migrated_decisions"] / decision_count
        if decision_count > 0 else 0.0
    )
    stay_action_ratio = (
        counters["stay_action_count"] / agent_decision_count if agent_decision_count > 0 else 0.0
    )
    sorted_excess = sorted(sla_excess_distance_history)
    p95_idx = int(0.95 * (len(sorted_excess) - 1)) if sorted_excess else 0
    adc = agent_decision_count
    mask_only = int(counters["mask_only_stay_agent_count"])
    dc = decision_count

    return {
        "total_migrations": counters["total_migrations"],
        "total_violations": counters["total_violations"],
        "primary_entry_violations": counters["primary_entry_violations"],
        "max_entry_violations": counters["max_entry_violations"],
        "severe_sla_violations": counters["severe_sla_violations"],
        "total_sla_excess_distance_km": counters["total_sla_excess_distance_km"],
        "avg_sla_excess_distance_km": (
            counters["total_sla_excess_distance_km"] / len(sla_excess_distance_history)
            if sla_excess_distance_history else 0.0
        ),
        "p95_sla_excess_distance_km": sorted_excess[p95_idx] if sorted_excess else 0.0,
        "proactive_decisions": counters["proactive_decisions"],
        "decision_count": decision_count,
        "total_reward": counters["total_reward_sum"],
        "total_access_latency": counters["total_access_latency"],
        "total_communication_cost": counters["total_communication_cost"],
        "total_internal_path_ms": counters["total_internal_path_ms"],
        "total_migration_cost": counters["total_migration_cost"],
        "migration_decision_count": counters["migration_decision_count"],
        "total_cost_ms_sum": counters["total_cost_ms_sum"],
        "total_sla_penalty_ms": counters["total_sla_penalty_ms"],
        "total_tearing_penalty_ms": counters["total_tearing_penalty_ms"],
        "total_future_penalty_ms": counters["total_future_penalty_ms"],
        "loss_history": loss_history,
        "reward_history": reward_history,
        "epsilon_history": epsilon_history,
        "entropy_history": entropy_history,
        "lambda_migration_history": lambda_migration_history,
        "lambda_split_history": lambda_split_history,
        "total_decision_time": counters["total_decision_time"],
        "decision_count_for_latency": decision_count,
        "avg_decision_time_ms": avg_ms,
        "avg_agents_per_decision": avg_agents,
        "controlled_agents_per_decision": avg_controlled_agents,
        "pinned_agents_per_decision": avg_pinned_agents,
        "avg_migrated_agents_per_decision": avg_migrated_agents,
        "avg_controlled_migrated_agents_per_decision": avg_controlled_migrated_agents,
        "controlled_migrations": counters["controlled_migrations"],
        "all_agents_migrated_decisions": counters["all_agents_migrated_decisions"],
        "all_agents_migrated_ratio": all_agents_migrated_ratio,
        "controlled_all_agents_migrated_decisions": counters["controlled_all_agents_migrated_decisions"],
        "controlled_all_agents_migrated_ratio": controlled_all_agents_migrated_ratio,
        "stay_action_ratio": stay_action_ratio,
        "candidate_action_counts": dict(candidate_action_counts),
        "joint_action_distribution": dict(joint_action_distribution),
        "local_migration_cost_sum": counters["local_migration_cost_sum"],
        "edge_split_cost_sum": counters["edge_split_cost_sum"],
        "dense_distance_bonus_sum": counters["dense_distance_bonus_sum"],
        "entry_sla_bonus_sum": counters["entry_sla_bonus_sum"],
        "proactive_logit_bias_count": counters["proactive_logit_bias_count"],
        "proactive_best_bias_action_counts": dict(proactive_best_bias_action_counts),
        "reactive_action_clipped_count": counters["reactive_action_clipped_count"],
        "counterfactual_score_sum": counters["counterfactual_score_sum"],
        "entry_node_migration_count": counters["entry_node_migration_count"],
        "sla_improving_action_count": counters["sla_improving_action_count"],
        "cost_guard_blocked_count": counters["cost_guard_blocked_count"],
        "non_entry_distance_only_blocked_count": counters["non_entry_distance_only_blocked_count"],
        "mask_only_stay_agent_count": mask_only,
        "mask_only_stay_agent_ratio": (mask_only / adc) if adc > 0 else 0.0,
        "agents_with_migrate_option_count": int(counters["agents_with_migrate_option_count"]),
        "decisions_with_migrate_option_count": int(counters["decisions_with_migrate_option_count"]),
        "decisions_with_migrate_option_ratio": (
            counters["decisions_with_migrate_option_count"] / dc if dc > 0 else 0.0
        ),
        "avg_legal_migrate_actions_per_decision": (
            counters["legal_migrate_slots_sum"] / dc if dc > 0 else 0.0
        ),
        "epsilon_greedy_samples": int(counters["epsilon_greedy_samples"]),
        "argmax_samples": int(counters["argmax_samples"]),
        "epoch_stats": epoch_stats,
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
        "invalid_action_masked_count": counters["invalid_action_masked_count"],
        "action_mask_fallback_count": counters["action_mask_fallback_count"],
        "lambda_migration": lambda_migration_history[-1] if lambda_migration_history else 0.0,
        "lambda_split": lambda_split_history[-1] if lambda_split_history else 0.0,
        "dag_proactive_migration_stats": (
            dict(dag_proactive_stats) if collect_dag_proactive_stats else {}
        ),
    }


def _server_distance_to_user(server_id, user_lat, user_lon, servers_info):
    srv_lat, srv_lon = servers_info[server_id]
    return float(haversine_distance(user_lat, user_lon, srv_lat, srv_lon))


def _edge_split_delta_ms(src, dst, traffic, assignments, node, target_server, servers_info):
    src_server = target_server if src == node else assignments[src]
    dst_server = target_server if dst == node else assignments[dst]
    old_same = assignments[src] == assignments[dst]
    new_same = src_server == dst_server
    if old_same == new_same:
        return 0.0

    def split_cost(server_a, server_b):
        if server_a == server_b:
            return 0.0
        lat_a, lon_a = servers_info[server_a]
        lat_b, lon_b = servers_info[server_b]
        dist_km = float(haversine_distance(lat_a, lon_a, lat_b, lon_b))
        cross_mb = min(float(traffic) * RPC_SIZE_MB, MAX_TEARING_MB)
        tearing_ms = (cross_mb / EDGE_BACKHAUL_MBPS) * 1000.0
        comm_ms = edge_effective_latency_ms(dist_km, traffic, True)
        return float(tearing_ms + comm_ms)

    old_cost = split_cost(assignments[src], assignments[dst])
    new_cost = split_cost(src_server, dst_server)
    return float(new_cost - old_cost)


def _build_counterfactual_context(
    dag_info,
    assignments,
    candidates,
    user_lat,
    user_lon,
    servers_info,
    predicted_locations=None,
    trigger_type=TRIGGER_PROACTIVE,
):
    deployable = set(get_deployable_nodes(dag_info))
    entry_nodes = [node for node in get_service_entry_nodes(dag_info) if node in deployable]
    entry_distances = {
        node: _server_distance_to_user(assignments[node], user_lat, user_lon, servers_info)
        for node in entry_nodes
    }
    current_max_entry_dist = max(entry_distances.values()) if entry_distances else 0.0
    current_access_latency_ms = float(calc_access_latency_ms(current_max_entry_dist))
    current_internal_path_ms = float(
        compute_internal_critical_path_ms(dag_info, assignments, servers_info)
    )
    bottleneck_entries = {
        node for node, dist in entry_distances.items()
        if dist >= current_max_entry_dist - 1e-6
    }
    violating_entries = {
        node for node, dist in entry_distances.items()
        if dist > DISTANCE_THRESHOLD_KM
    }
    risk_ratio = (
        min(current_max_entry_dist / DISTANCE_THRESHOLD_KM, 1.0)
        if DISTANCE_THRESHOLD_KM > 0 else 0.0
    )
    bandwidth = MIN_BW_MBPS + (MAX_BW_MBPS - MIN_BW_MBPS) * (risk_ratio ** 2)
    max_traffic = max(dag_info["edges"].values()) if dag_info["edges"] else 0.0

    incident_edges = defaultdict(list)
    for (src, dst), traffic in dag_info["edges"].items():
        incident_edges[src].append((src, dst, traffic))
        incident_edges[dst].append((src, dst, traffic))

    candidate_distances = {}
    for action in range(MARL_ACTION_DIM):
        if action == 0:
            continue
        cand_idx = action - 1
        if 0 <= cand_idx < len(candidates or []):
            server_id = candidates[cand_idx][0]
            candidate_distances[action] = _server_distance_to_user(
                server_id, user_lat, user_lon, servers_info
            )

    migration_costs = {}
    migration_mult = _cf_migration_mult_for_scoring(trigger_type)
    for node in deployable:
        props = dag_info["nodes"][node]
        mb = float(props["image_mb"]) + float(props["state_mb"])
        migration_costs[node] = (
            ((mb * MB_TO_MBIT / max(bandwidth, 1e-6)) * 1000.0)
            + BASE_MIGRATION_OVERHEAD_MS
        ) * migration_mult

    pred_arr = None
    current_future_max = 0.0
    if predicted_locations:
        pred_arr = np.asarray(predicted_locations, dtype=np.float64)
        if pred_arr.ndim != 2 or pred_arr.shape[1] != 2:
            pred_arr = np.reshape(pred_arr, (-1, 2))
        if pred_arr.size > 0 and entry_nodes:
            future_max = []
            for lat, lon in pred_arr:
                dists = [
                    _server_distance_to_user(assignments[node], float(lat), float(lon), servers_info)
                    for node in entry_nodes
                ]
                future_max.append(max(dists) if dists else 0.0)
            current_future_max = float(np.mean(np.maximum(0.0, np.asarray(future_max) - DISTANCE_THRESHOLD_KM)))

    return {
        "deployable": deployable,
        "entry_nodes": set(entry_nodes),
        "primary_entry": entry_nodes[0] if entry_nodes else None,
        "entry_distances": entry_distances,
        "current_max_entry_dist": current_max_entry_dist,
        "current_access_latency_ms": current_access_latency_ms,
        "current_internal_path_ms": current_internal_path_ms,
        "bottleneck_entries": bottleneck_entries,
        "violating_entries": violating_entries,
        "bandwidth": bandwidth,
        "max_traffic": max_traffic,
        "incident_edges": incident_edges,
        "candidate_distances": candidate_distances,
        "migration_costs": migration_costs,
        "split_delta_cache": {},
        "future_gain_cache": {},
        "predicted_locations": pred_arr,
        "current_future_max": current_future_max,
        "trigger_type": trigger_type,
        "user_lat": user_lat,
        "user_lon": user_lon,
    }


def _score_counterfactual_action(
    dag_info,
    node,
    action,
    assignments,
    candidates,
    servers_info,
    cf_ctx,
    lambda_migration,
    lambda_split,
):
    if is_external_node(node) or int(action) == 0:
        return None
    current_server = assignments[node]
    target_server = action_to_server(action, candidates, current_server)
    if target_server == current_server:
        return None

    migration_cost_ms = cf_ctx["migration_costs"].get(node, 0.0)
    node_props = dag_info["nodes"].get(node, {})
    nonlinear_migration_cost_ms = calculate_nonlinear_migration_cost_ms(
        migration_cost_ms,
        node_props.get("image_mb", 0.0),
        node_props.get("state_mb", 0.0),
    )

    split_key = (node, int(action))
    if split_key in cf_ctx["split_delta_cache"]:
        split_delta_ms = cf_ctx["split_delta_cache"][split_key]
    else:
        split_delta_ms = 0.0
        for src, dst, traffic in cf_ctx["incident_edges"].get(node, []):
            split_delta_ms += _edge_split_delta_ms(
                src, dst, traffic, assignments, node, target_server, servers_info
            )
        cf_ctx["split_delta_cache"][split_key] = split_delta_ms
    split_cost_ms = max(0.0, split_delta_ms)
    topology_gain_ms = max(0.0, -split_delta_ms)

    sla_gain_ms = 0.0
    future_gain_ms = 0.0
    non_entry_distance_only = False

    trial_assignments = dict(assignments)
    trial_assignments[node] = target_server
    new_internal_path_ms = float(
        compute_internal_critical_path_ms(dag_info, trial_assignments, servers_info)
    )
    new_max_entry_dist, new_access_latency_ms = _entry_access_profile(
        trial_assignments,
        dag_info,
        cf_ctx["user_lat"],
        cf_ctx["user_lon"],
        servers_info,
    )
    old_max_entry_dist = cf_ctx["current_max_entry_dist"]
    old_access_latency_ms = cf_ctx["current_access_latency_ms"]
    old_internal_path_ms = cf_ctx["current_internal_path_ms"]

    if use_reward_v2_internal_path():
        sla_gain_ms = sla_penalty_gain_ms(
            old_max_entry_dist,
            new_max_entry_dist,
            old_access_latency_ms,
            new_access_latency_ms,
            old_internal_path_ms,
            new_internal_path_ms,
            use_e2e_qos=True,
        )
    elif node in cf_ctx["entry_nodes"] and (
        node in cf_ctx["violating_entries"]
        or node in cf_ctx["bottleneck_entries"]
        or node == cf_ctx["primary_entry"]
    ):
        target_dist = cf_ctx["candidate_distances"].get(int(action))
        if target_dist is not None:
            other_entry_max = max(
                [
                    dist for entry, dist in cf_ctx["entry_distances"].items()
                    if entry != node
                ] or [0.0]
            )
            new_max = max(other_entry_max, target_dist)
            old_max = cf_ctx["current_max_entry_dist"]
            sla_gain_ms = sla_penalty_gain_ms(old_max, new_max)
            sla_gain_ms += (
                max(0.0, old_max - new_max)
                * SLA_PENALTY_PER_KM_MS
                * 0.25
            )
            primary_old = cf_ctx["entry_distances"].get(node, 0.0)
            primary_penalty_old = calculate_sla_penalty_ms(primary_old)
            primary_penalty_new = calculate_sla_penalty_ms(target_dist)
            sla_gain_ms += (
                max(0.0, primary_penalty_old - primary_penalty_new)
                * 0.5
            )
            sla_gain_ms += (
                max(0.0, primary_old - target_dist)
                * SLA_PENALTY_PER_KM_MS
                * 0.1
            )

            pred_arr = cf_ctx["predicted_locations"]
            if pred_arr is not None and pred_arr.size > 0 and cf_ctx["entry_nodes"]:
                future_key = (node, int(action))
                if future_key in cf_ctx["future_gain_cache"]:
                    future_gain_ms = cf_ctx["future_gain_cache"][future_key]
                else:
                    future_excess = []
                    for lat, lon in pred_arr:
                        dists = []
                        for entry in cf_ctx["entry_nodes"]:
                            server_id = target_server if entry == node else assignments[entry]
                            dists.append(
                                _server_distance_to_user(
                                    server_id, float(lat), float(lon), servers_info
                                )
                            )
                        future_excess.append(
                            max(dists) - DISTANCE_THRESHOLD_KM if dists else 0.0
                        )
                    new_future = float(np.mean(np.maximum(0.0, np.asarray(future_excess))))
                    future_gain_ms = future_mean_excess_penalty_gain_ms(
                        cf_ctx["current_future_max"],
                        new_future,
                    )
                    if _node_state_mb(dag_info, node) > 0.0:
                        future_gain_ms *= STATEFUL_FUTURE_GAIN_DISCOUNT
                    cf_ctx["future_gain_cache"][future_key] = future_gain_ms
    else:
        target_dist = cf_ctx["candidate_distances"].get(int(action))
        current_dist = _server_distance_to_user(
            current_server,
            cf_ctx["user_lat"],
            cf_ctx["user_lon"],
            servers_info,
        )
        if target_dist is not None and target_dist < current_dist:
            non_entry_distance_only = True

    effective_sla_gain_ms = sla_gain_ms + 0.5 * future_gain_ms
    score = (
        effective_sla_gain_ms * CF_SLA_WEIGHT
        - (nonlinear_migration_cost_ms / _cf_cost_scale_ms()) * float(lambda_migration)
    )
    return {
        "score": float(score),
        "sla_gain_ms": float(sla_gain_ms),
        "effective_sla_gain_ms": float(effective_sla_gain_ms),
        "future_gain_ms": float(future_gain_ms),
        "topology_gain_ms": float(topology_gain_ms),
        "migration_cost_ms": float(migration_cost_ms),
        "nonlinear_migration_cost_ms": float(nonlinear_migration_cost_ms),
        "split_cost_ms": float(split_cost_ms),
        "non_entry_distance_only": bool(non_entry_distance_only),
        "is_heavy_proactive": bool(_is_heavy_for_proactive(dag_info, node)),
        "is_lightweight_entry_rescue": bool(_is_lightweight_entry_rescue(dag_info, node, cf_ctx)),
    }


def _apply_proactive_distance_bias(
    masked_logits,
    sorted_nodes,
    node_to_idx,
    action_masks,
    assignments,
    candidates,
    user_lat,
    user_lon,
    servers_info,
    dag_info,
    predicted_locations,
    lambda_migration,
    lambda_split,
    *,
    bias_scale=0.15,
    max_bias=0.25,
):
    """Add a light logit prior for actions with positive counterfactual value."""
    if _bypass_rule_based_guards():
        return masked_logits, 0, {"0": len(sorted_nodes)}, _passthrough_guard_stats()

    biased_logits = masked_logits.clone()
    positive_bias_count = 0
    best_action_counts = defaultdict(int)
    score_sum = 0.0
    sla_improving_count = 0
    cost_guard_blocked_count = 0
    non_entry_distance_only_blocked_count = 0
    cf_ctx = _build_counterfactual_context(
        dag_info,
        assignments,
        candidates,
        user_lat,
        user_lon,
        servers_info,
        predicted_locations=predicted_locations,
        trigger_type=TRIGGER_PROACTIVE,
    )

    for node in sorted_nodes:
        if is_external_node(node):
            continue
        node_idx = node_to_idx[node]
        best_action = 0
        best_score = -float("inf")
        for action in range(1, MARL_ACTION_DIM):
            if not bool(action_masks[node_idx, action].item()):
                continue
            scored = _score_counterfactual_action(
                dag_info,
                node,
                action,
                assignments,
                candidates,
                servers_info,
                cf_ctx,
                lambda_migration,
                lambda_split,
            )
            if scored is None:
                continue
            score_sum += scored["score"]
            if scored["sla_gain_ms"] > 0.0:
                sla_improving_count += 1
            lightweight_rescue = (
                scored["is_lightweight_entry_rescue"]
                and scored["sla_gain_ms"] >= LIGHTWEIGHT_ENTRY_MIN_SLA_GAIN_MS
                and scored["score"] > LIGHTWEIGHT_ENTRY_SCORE_FLOOR
            )
            if scored["non_entry_distance_only"] and scored["score"] <= CF_SCORE_EPS:
                non_entry_distance_only_blocked_count += 1
            if scored["score"] <= CF_SCORE_EPS and not lightweight_rescue:
                cost_guard_blocked_count += 1
                continue
            bias_source = max(scored["score"], LIGHTWEIGHT_ENTRY_MIN_BIAS) if lightweight_rescue else scored["score"]
            bias = min(max_bias, bias_scale * bias_source)
            biased_logits[node_idx, action] = biased_logits[node_idx, action] + float(bias)
            positive_bias_count += 1
            if scored["score"] > best_score:
                best_score = scored["score"]
                best_action = action
        best_action_counts[str(best_action)] += 1

    return biased_logits, positive_bias_count, dict(best_action_counts), {
        "counterfactual_score_sum": float(score_sum),
        "sla_improving_action_count": int(sla_improving_count),
        "cost_guard_blocked_count": int(cost_guard_blocked_count),
        "non_entry_distance_only_blocked_count": int(non_entry_distance_only_blocked_count),
    }


def _apply_proactive_size_guard(
    actions,
    sorted_nodes,
    assignments,
    candidates,
    user_lat,
    user_lon,
    servers_info,
    dag_info,
    predicted_locations,
    lambda_migration,
    lambda_split,
):
    """Apply proactive migration budget and cost-benefit filtering."""
    if _bypass_rule_based_guards():
        return list(actions), 0, _passthrough_guard_stats()

    guarded_actions = list(actions)
    clipped = 0
    score_sum = 0.0
    sla_improving_count = 0
    cost_guard_blocked_count = 0
    non_entry_distance_only_blocked_count = 0
    accepted = []
    cf_ctx = _build_counterfactual_context(
        dag_info,
        assignments,
        candidates,
        user_lat,
        user_lon,
        servers_info,
        predicted_locations=predicted_locations,
        trigger_type=TRIGGER_PROACTIVE,
    )
    for idx, (node, action) in enumerate(zip(sorted_nodes, actions)):
        if is_external_node(node) or int(action) == 0:
            continue
        item = _score_counterfactual_action(
            dag_info,
            node,
            action,
            assignments,
            candidates,
            servers_info,
            cf_ctx,
            lambda_migration,
            lambda_split,
        )
        if item is None:
            continue
        score_sum += item["score"]
        if item["sla_gain_ms"] > 0.0:
            sla_improving_count += 1
        if item["non_entry_distance_only"] and item["score"] <= CF_SCORE_EPS:
            non_entry_distance_only_blocked_count += 1
        migration_cost = max(float(item.get("nonlinear_migration_cost_ms", 0.0)), 1e-6)
        gain_cost_ratio = float(item.get("effective_sla_gain_ms", 0.0)) / migration_cost
        high_cost_low_gain = (
            item["nonlinear_migration_cost_ms"] > HIGH_COST_MIGRATION_MS
            and item["effective_sla_gain_ms"] < HIGH_COST_MIN_SLA_GAIN_MS
        )
        low_value_migration = (
            gain_cost_ratio < MIN_SLA_GAIN_COST_RATIO
            and item["effective_sla_gain_ms"] < HIGH_COST_MIN_SLA_GAIN_MS
        )
        if (
            item["score"] <= CF_SCORE_EPS
            or item["sla_gain_ms"] <= 0.0
            or high_cost_low_gain
            or low_value_migration
        ):
            guarded_actions[idx] = 0
            clipped += 1
            cost_guard_blocked_count += 1
            continue
        accepted.append((item["score"], idx, migration_cost))
    keep_indices = set()
    used_budget_ms = 0.0
    cap_by_count = not _unlimited_migration_cap(PROACTIVE_MAX_MIGRATIONS_PER_DECISION)
    for _, idx, migration_cost in sorted(accepted, reverse=True):
        if cap_by_count and len(keep_indices) >= PROACTIVE_MAX_MIGRATIONS_PER_DECISION:
            break
        if used_budget_ms + migration_cost > PROACTIVE_MIGRATION_BUDGET_MS:
            continue
        keep_indices.add(idx)
        used_budget_ms += migration_cost
    for idx, action in enumerate(list(guarded_actions)):
        if int(action) != 0 and idx not in keep_indices:
            guarded_actions[idx] = 0
            clipped += 1
            cost_guard_blocked_count += 1
    return guarded_actions, clipped, {
        "counterfactual_score_sum": float(score_sum),
        "sla_improving_action_count": int(sla_improving_count),
        "cost_guard_blocked_count": int(cost_guard_blocked_count),
        "non_entry_distance_only_blocked_count": int(non_entry_distance_only_blocked_count),
    }


def _clip_reactive_actions(
    actions,
    sorted_nodes,
    assignments,
    candidates,
    user_lat,
    user_lon,
    servers_info,
    dag_info,
    lambda_migration,
    lambda_split,
    node_to_idx=None,
    action_masks=None,
    *,
    max_migrations=None,
    size_guard_enabled=False,
):
    """Keep only reactive migrations with positive counterfactual value."""
    if _bypass_rule_based_guards():
        return list(actions), 0, _passthrough_guard_stats()

    controlled_count = sum(1 for node in sorted_nodes if not is_external_node(node))
    cf_ctx = _build_counterfactual_context(
        dag_info,
        assignments,
        candidates,
        user_lat,
        user_lon,
        servers_info,
        predicted_locations=None,
        trigger_type=TRIGGER_REACTIVE,
    )
    if max_migrations is None:
        max_migrations = _reactive_migration_cap(
            controlled_count,
            len(cf_ctx["violating_entries"]),
        )
    sorted_node_to_action_idx = {node: idx for idx, node in enumerate(sorted_nodes)}
    scored = []
    score_sum = 0.0
    sla_improving_count = 0
    cost_guard_blocked_count = 0
    non_entry_distance_only_blocked_count = 0
    for idx, (node, action) in enumerate(zip(sorted_nodes, actions)):
        if is_external_node(node) or int(action) == 0:
            continue
        item = _score_counterfactual_action(
            dag_info,
            node,
            action,
            assignments,
            candidates,
            servers_info,
            cf_ctx,
            lambda_migration,
            lambda_split,
        )
        if item is None:
            continue
        score_sum += item["score"]
        if item["sla_gain_ms"] > 0.0:
            sla_improving_count += 1
        if item["non_entry_distance_only"] and item["score"] <= CF_SCORE_EPS:
            non_entry_distance_only_blocked_count += 1
        if _cf_migrate_action_allowed(
            item, node, cf_ctx, gate_mode=_cf_gate_mode(), action=int(action)
        ):
            scored.append((item["score"], idx))
        else:
            cost_guard_blocked_count += 1

    selected = {
        idx: (float(score), int(actions[idx]))
        for score, idx in sorted(scored, reverse=True)[:max_migrations]
    }

    if action_masks is not None and node_to_idx is not None:
        entry_candidates = []
        forced_entry_indices = set()
        for node in sorted(
            cf_ctx["violating_entries"],
            key=lambda n: cf_ctx["entry_distances"].get(n, 0.0),
            reverse=True,
        ):
            if node not in node_to_idx or node not in sorted_node_to_action_idx:
                continue
            node_idx = node_to_idx[node]
            action_idx = sorted_node_to_action_idx[node]
            if bool(action_masks[node_idx, 1].item()):
                item = _score_counterfactual_action(
                    dag_info,
                    node,
                    1,
                    assignments,
                    candidates,
                    servers_info,
                    cf_ctx,
                    lambda_migration,
                    lambda_split,
                )
                score = float(item["score"]) if item is not None else 0.0
                entry_candidates.append((score, action_idx, 1))
                continue
            best_entry = None
            for action in range(1, MARL_ACTION_DIM):
                if not bool(action_masks[node_idx, action].item()):
                    continue
                item = _score_counterfactual_action(
                    dag_info,
                    node,
                    action,
                    assignments,
                    candidates,
                    servers_info,
                    cf_ctx,
                    lambda_migration,
                    lambda_split,
                )
                if not _cf_migrate_action_allowed(
                    item, node, cf_ctx, gate_mode=_cf_gate_mode(), action=action
                ):
                    continue
                if best_entry is None or item["score"] > best_entry[0]:
                    best_entry = (item["score"], action_idx, action)
            if best_entry is not None:
                entry_candidates.append(best_entry)

        for score, action_idx, action in sorted(entry_candidates, reverse=True):
            selected[action_idx] = (float(score), int(action))
            if sorted_nodes[action_idx] in cf_ctx["violating_entries"]:
                forced_entry_indices.add(action_idx)
    else:
        forced_entry_indices = set()

    if len(selected) > max_migrations:
        forced = [
            (idx, selected[idx])
            for idx in forced_entry_indices
            if idx in selected
        ]
        forced = sorted(forced, key=lambda kv: kv[1][0], reverse=True)[:max_migrations]
        remaining_slots = max_migrations - len(forced)
        forced_ids = {idx for idx, _ in forced}
        optional = [
            (idx, value)
            for idx, value in selected.items()
            if idx not in forced_ids
        ]
        optional = sorted(optional, key=lambda kv: kv[1][0], reverse=True)[:remaining_slots]
        selected = dict(forced + optional)

    clipped = 0
    clipped_actions = [0 for _ in actions]
    for idx, (score, action) in selected.items():
        clipped_actions[idx] = int(action)
    for idx, (node, action) in enumerate(zip(sorted_nodes, actions)):
        if is_external_node(node) or int(action) == 0:
            continue
        if idx not in selected:
            clipped += 1
    return clipped_actions, clipped, {
        "counterfactual_score_sum": float(score_sum),
        "sla_improving_action_count": int(sla_improving_count),
        "cost_guard_blocked_count": int(cost_guard_blocked_count),
        "non_entry_distance_only_blocked_count": int(non_entry_distance_only_blocked_count),
    }


def run_marl_gat_microservice(
    df,
    servers_df,
    predictor=None,
    proactive=False,
    num_epochs=4,
    inference_mode=False,
    checkpoint_path=None,
    save_checkpoint_path=None,
    warmstart_checkpoint_path=None,
    collect_dag_proactive_stats=False,
    max_lambda_migration=None,
    max_lambda_split=None,
):
    """Run CTDE-GAT-MARL under the same trigger/reward protocol as baselines."""
    servers_info = build_servers_info(servers_df)
    use_proactive = proactive and predictor is not None
    if max_lambda_migration is None:
        # Linear SLA penalty lowers benefit scale, so Proactive needs a stronger
        # cost exchange rate to avoid expensive early migrations.
        max_lambda_migration = 0.50 if use_proactive else 0.35
    if max_lambda_split is None:
        max_lambda_split = 0.04 if use_proactive else 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    guard_mode = "soft-reward (v2 bypass)" if _bypass_rule_based_guards() else "rule-based guards"
    p1_mode = _use_p1_coordination()
    soft_cf_mode = _use_soft_cf_bias()
    sla_gate_mode = _use_sla_violation_gate()
    cf_sla_gain_gate_mode = _use_cf_sla_gain_gate()
    curriculum_on = curriculum_enabled()
    gamma_on = internal_gamma_curriculum_enabled()
    total_cost_train = train_total_cost_enabled()
    cf_train_floor = os.environ.get("MARL_CF_TRAIN_SCORE_FLOOR", "").strip()
    print(
        f"  Device: {device}  |  Proactive: {use_proactive}  |  Model: CTDE-GAT-MARL  |  "
        f"Lambda(CF): migration={max_lambda_migration:.3f}, split={max_lambda_split:.3f}  |  "
        f"Guards: {guard_mode}  |  P1: {'entry-first + max-1' if p1_mode else 'off'}  |  "
        f"P2: curriculum={'on' if curriculum_on else 'off'} soft_cf={'on' if soft_cf_mode else 'off'}  |  "
        f"P3: L_internal gamma={'on' if gamma_on else 'off'}  |  "
        f"A: train_total_cost={'on' if total_cost_train else 'off'}  |  "
        f"E: cf_physical_mig={'on' if _cf_physical_migration_cost_enabled() else 'off'}  |  "
        f"A': reactive_dense_bonus={'on' if reactive_dense_bonus_enabled() else 'off'}  |  "
        f"C': cf_train_floor={cf_train_floor or 'off'}  |  "
        f"B1: SLA gate={'on' if sla_gate_mode else 'off'}  |  "
        f"B1.1: CF gate={'on' if cf_sla_gain_gate_mode else 'off'}"
        + (f" mode={_cf_gate_mode()}" if cf_sla_gain_gate_mode else "")
    )

    hidden_dim = 64
    encoder = GraphEncoder(
        node_feat_dim=MARL_NODE_FEAT_DIM,
        trigger_dim=MARL_TRIGGER_CONTEXT_DIM,
        candidate_dim=MARL_CANDIDATE_FEATURE_DIM,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
    ).to(device)
    actor = SharedNodeActor(
        embedding_dim=hidden_dim,
        trigger_dim=MARL_TRIGGER_CONTEXT_DIM,
    ).to(device)
    critic = CentralCritic(
        embedding_dim=hidden_dim,
        trigger_dim=MARL_TRIGGER_CONTEXT_DIM,
        candidate_dim=MARL_CANDIDATE_FEATURE_DIM,
    ).to(device)
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(actor.parameters()) + list(critic.parameters()),
        lr=3e-4,
    )

    if inference_mode:
        if checkpoint_path is None:
            raise ValueError("inference_mode=True 但未提供 checkpoint_path")
        _load_marl_checkpoint(checkpoint_path, encoder, actor, critic, device)
        num_epochs = 1
    else:
        warmstart = warmstart_checkpoint_path or os.environ.get("MARL_WARMSTART_CHECKPOINT")
        if warmstart and os.path.isfile(warmstart):
            _load_marl_checkpoint(warmstart, encoder, actor, critic, device, strict=False)

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
    epoch_stats_list = []

    timestamps = sorted(df["date_time"].unique())
    df_grouped = df.groupby("date_time")
    global_step = 0

    for epoch in range(num_epochs):
        is_eval_epoch = inference_mode or (epoch == num_epochs - 1)
        taxi_dag_type = {}
        taxi_dag_assignments = {}
        taxi_last = {}
        epoch_loss_history = []
        epoch_entropy_history = []

        total_decision_time = 0.0
        decision_count = 0
        total_migrations = 0
        total_violations = 0
        primary_entry_violations = 0
        max_entry_violations = 0
        severe_sla_violations = 0
        total_sla_excess_distance_km = 0.0
        sla_excess_distance_history = []
        proactive_decisions = 0
        total_reward_sum = 0.0
        total_access_latency = 0.0
        total_communication_cost = 0.0
        total_internal_path_ms = 0.0
        total_migration_cost = 0.0
        migration_decision_count = 0
        total_cost_ms_sum = 0.0
        total_sla_penalty_ms = 0.0
        total_tearing_penalty_ms = 0.0
        total_future_penalty_ms = 0.0
        local_migration_cost_sum = 0.0
        edge_split_cost_sum = 0.0
        dense_distance_bonus_sum = 0.0
        entry_sla_bonus_sum = 0.0
        proactive_logit_bias_count = 0
        proactive_best_bias_action_counts = {str(i): 0 for i in range(MARL_ACTION_DIM)}
        reactive_action_clipped_count = 0
        counterfactual_score_sum = 0.0
        entry_node_migration_count = 0
        sla_improving_action_count = 0
        cost_guard_blocked_count = 0
        non_entry_distance_only_blocked_count = 0
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
        dag_complexity_stats = defaultdict(
            lambda: {"decision_count": 0, "total_cost_ms_sum": 0.0, "migrations": 0}
        )
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
        mask_only_stay_agent_count = 0
        agents_with_migrate_option_count = 0
        legal_migrate_slots_sum = 0
        decisions_with_migrate_option_count = 0
        epsilon_greedy_samples = 0
        argmax_samples = 0
        p1_multi_migration_suppressed_count = 0
        soft_cf_bias_count = 0
        soft_cf_neg_bias_count = 0
        sla_gate_no_violation_decisions = 0
        sla_gate_violation_decisions = 0
        sla_gate_forced_stay_agents = 0
        cf_gate_blocked_slots = 0
        cf_gate_improving_slots = 0
        cf_gate_forced_stay_agents = 0
        cf_gate_entry_rescue_slots = 0
        decisions_with_cf_improve_option_count = 0
        epoch_curriculum = apply_curriculum_for_epoch(
            epoch, num_epochs, is_eval_epoch=is_eval_epoch
        )
        if epoch_curriculum and epoch_curriculum.get("train_total_cost_mode"):
            print(
                f"  [GAT-MARL ep {epoch + 1}] Direction A: "
                f"train=-total_cost_ms, gamma=1.0 fixed, lambda_train=0",
                flush=True,
            )
        elif epoch_curriculum:
            print(
                f"  [GAT-MARL curriculum ep {epoch + 1}] "
                f"S={epoch_curriculum['objective_scale_ms']:.0f} "
                f"alpha_x={epoch_curriculum['sla_alpha_mult']:.2f} "
                f"beta_x={epoch_curriculum['sla_beta_mult']:.2f} "
                f"lambda_x={epoch_curriculum['migration_lambda_mult']:.2f} "
                f"gamma={epoch_curriculum.get('internal_path_gamma', 1.0):.2f}",
                flush=True,
            )

        global _cf_gate_use_train_score_floor
        _cf_gate_use_train_score_floor = (
            not inference_mode and not is_eval_epoch and bool(cf_train_floor)
        )

        if is_eval_epoch:
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
        if total_cost_train:
            train_lm, train_ls = 0.0, 0.0
        else:
            train_lm, train_ls = epoch_lm, epoch_ls

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

                sla_metrics = calculate_entry_sla_metrics(
                    entry_nodes,
                    taxi_dag_assignments[taxi_id],
                    current_lat,
                    current_lon,
                    servers_info,
                )
                if sla_metrics["primary_entry_violation"]:
                    primary_entry_violations += 1
                if sla_metrics["max_entry_violation"]:
                    max_entry_violations += 1
                    total_violations += 1
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
                    dag_type=dag_type,
                    p1_entry_first=p1_mode,
                )
                sorted_nodes = topological_sort(dag_info)
                controlled_nodes = [node for node in sorted_nodes if not is_external_node(node)]
                node_to_idx = {name: i for i, name in enumerate(state["node_names"])}
                actions = []
                lm, ls = epoch_lm, epoch_ls

                t0 = time.perf_counter()
                with torch.no_grad():
                    tensors = _state_to_tensors(state, device)
                    if sla_gate_mode:
                        sla_state = dag_current_sla_violation(
                            dag_info,
                            taxi_dag_assignments[taxi_id],
                            current_lat,
                            current_lon,
                            servers_info,
                        )
                        if sla_state["violated"]:
                            sla_gate_violation_decisions += 1
                        else:
                            sla_gate_no_violation_decisions += 1
                            tensors["action_masks"], forced = _apply_sla_violation_action_gate(
                                tensors["action_masks"]
                            )
                            sla_gate_forced_stay_agents += forced
                    if cf_sla_gain_gate_mode:
                        tensors["action_masks"], cf_gate_stats = _apply_cf_sla_gain_action_gate(
                            tensors["action_masks"],
                            sorted_nodes,
                            node_to_idx,
                            dag_info,
                            taxi_dag_assignments[taxi_id],
                            candidates,
                            current_lat,
                            current_lon,
                            servers_info,
                            trigger_type,
                            lm,
                            ls,
                        )
                        cf_gate_blocked_slots += cf_gate_stats["cf_gate_blocked_slots"]
                        cf_gate_improving_slots += cf_gate_stats["cf_gate_improving_slots"]
                        cf_gate_forced_stay_agents += cf_gate_stats["cf_gate_forced_stay_agents"]
                        cf_gate_entry_rescue_slots += cf_gate_stats.get("cf_gate_entry_rescue_slots", 0)
                        if cf_gate_stats["cf_gate_improving_slots"] > 0:
                            decisions_with_cf_improve_option_count += 1
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
                    if soft_cf_mode:
                        masked_logits, bias_count, best_bias_actions, cf_info = _apply_soft_counterfactual_bias(
                            masked_logits,
                            sorted_nodes,
                            node_to_idx,
                            tensors["action_masks"],
                            taxi_dag_assignments[taxi_id],
                            candidates,
                            current_lat,
                            current_lon,
                            servers_info,
                            dag_info,
                            predicted_locations,
                            lm,
                            ls,
                            trigger_type,
                        )
                        proactive_logit_bias_count += bias_count
                        soft_cf_bias_count += bias_count
                        soft_cf_neg_bias_count += cf_info.get("soft_cf_neg_bias_count", 0)
                        counterfactual_score_sum += cf_info["counterfactual_score_sum"]
                        sla_improving_action_count += cf_info["sla_improving_action_count"]
                        cost_guard_blocked_count += cf_info["cost_guard_blocked_count"]
                        non_entry_distance_only_blocked_count += cf_info["non_entry_distance_only_blocked_count"]
                        for key, value in best_bias_actions.items():
                            proactive_best_bias_action_counts[key] = (
                                proactive_best_bias_action_counts.get(key, 0) + value
                            )
                    elif use_proactive and trigger_type == TRIGGER_PROACTIVE:
                        masked_logits, bias_count, best_bias_actions, cf_info = _apply_proactive_distance_bias(
                            masked_logits,
                            sorted_nodes,
                            node_to_idx,
                            tensors["action_masks"],
                            taxi_dag_assignments[taxi_id],
                            candidates,
                            current_lat,
                            current_lon,
                            servers_info,
                            dag_info,
                            predicted_locations,
                            lm,
                            ls,
                        )
                        proactive_logit_bias_count += bias_count
                        counterfactual_score_sum += cf_info["counterfactual_score_sum"]
                        sla_improving_action_count += cf_info["sla_improving_action_count"]
                        cost_guard_blocked_count += cf_info["cost_guard_blocked_count"]
                        non_entry_distance_only_blocked_count += cf_info["non_entry_distance_only_blocked_count"]
                        for key, value in best_bias_actions.items():
                            proactive_best_bias_action_counts[key] = (
                                proactive_best_bias_action_counts.get(key, 0) + value
                            )
                    invalid_action_masked_count += masked_count
                    action_mask_fallback_count += fallback_count
                    probs = F.softmax(masked_logits, dim=-1)
                    decision_had_migrate_option = False

                    for ms_node in sorted_nodes:
                        node_idx = node_to_idx[ms_node]
                        if is_external_node(ms_node):
                            action = 0
                        elif (not is_eval_epoch) and random.random() < epsilon:
                            legal = torch.nonzero(
                                tensors["action_masks"][node_idx], as_tuple=False
                            ).flatten()
                            action = int(legal[torch.randint(0, len(legal), (1,), device=device)].item())
                            epsilon_greedy_samples += 1
                        else:
                            action = int(torch.argmax(probs[node_idx]).item())
                            argmax_samples += 1
                        if not is_external_node(ms_node):
                            legal = torch.nonzero(
                                tensors["action_masks"][node_idx], as_tuple=False
                            ).flatten()
                            legal_actions = [int(x.item()) for x in legal]
                            if len(legal_actions) == 1 and legal_actions[0] == 0:
                                mask_only_stay_agent_count += 1
                            migrate_legal = [a for a in legal_actions if a > 0]
                            if migrate_legal:
                                agents_with_migrate_option_count += 1
                                legal_migrate_slots_sum += len(migrate_legal)
                                decision_had_migrate_option = True
                        actions.append(action)
                    if decision_had_migrate_option:
                        decisions_with_migrate_option_count += 1

                    if use_proactive and trigger_type == TRIGGER_PROACTIVE:
                        actions, clipped_count, cf_info = _apply_proactive_size_guard(
                            actions,
                            sorted_nodes,
                            taxi_dag_assignments[taxi_id],
                            candidates,
                            current_lat,
                            current_lon,
                            servers_info,
                            dag_info,
                            predicted_locations,
                            lm,
                            ls,
                        )
                        reactive_action_clipped_count += clipped_count
                        counterfactual_score_sum += cf_info["counterfactual_score_sum"]
                        sla_improving_action_count += cf_info["sla_improving_action_count"]
                        cost_guard_blocked_count += cf_info["cost_guard_blocked_count"]
                        non_entry_distance_only_blocked_count += cf_info["non_entry_distance_only_blocked_count"]
                    elif trigger_type != TRIGGER_PROACTIVE:
                        actions, clipped_count, cf_info = _clip_reactive_actions(
                            actions,
                            sorted_nodes,
                            taxi_dag_assignments[taxi_id],
                            candidates,
                            current_lat,
                            current_lon,
                            servers_info,
                            dag_info,
                            lm,
                            ls,
                            node_to_idx,
                            tensors["action_masks"],
                            size_guard_enabled=use_proactive,
                        )
                        reactive_action_clipped_count += clipped_count
                        counterfactual_score_sum += cf_info["counterfactual_score_sum"]
                        sla_improving_action_count += cf_info["sla_improving_action_count"]
                        cost_guard_blocked_count += cf_info["cost_guard_blocked_count"]
                        non_entry_distance_only_blocked_count += cf_info["non_entry_distance_only_blocked_count"]

                    if p1_mode:
                        entry_nodes = get_service_entry_nodes(dag_info)
                        actions, suppressed = _apply_p1_max_one_migration(
                            actions,
                            sorted_nodes,
                            node_to_idx,
                            probs,
                            entry_nodes,
                        )
                        p1_multi_migration_suppressed_count += suppressed

                    for ms_node, action in zip(sorted_nodes, actions):
                        current_server = taxi_dag_assignments[taxi_id][ms_node]
                        if is_external_node(ms_node):
                            target_server = current_server
                        else:
                            target_server = action_to_server(action, candidates, current_server)
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

                lambda_migration_history.append(lm)
                lambda_split_history.append(ls)
                use_dense_bonus = reactive_dense_bonus_enabled()
                shared_reward, agent_rewards, details = calculate_marl_rewards(
                    taxi_id,
                    dag_info,
                    taxi_dag_assignments[taxi_id],
                    old_assignments,
                    (current_lat, current_lon),
                    servers_info,
                    predicted_locations=predicted_locations,
                    trigger_type=trigger_type,
                    lambda_migration=train_lm,
                    lambda_split=train_ls,
                    dense_distance_bonus=use_dense_bonus,
                )
                del agent_rewards
                total_reward_sum += shared_reward
                reward_history.append(shared_reward)
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
                local_migration_cost_sum += details.get("local_migration_cost_sum", 0.0)
                edge_split_cost_sum += details.get("edge_split_cost_sum", 0.0)
                dense_distance_bonus_sum += details.get("dense_distance_bonus_sum", 0.0)
                entry_sla_bonus_sum += details.get("entry_sla_bonus_sum", 0.0)

                nodes_migrated = sum(
                    1 for n in sorted_nodes
                    if old_assignments[n] != taxi_dag_assignments[taxi_id][n]
                )
                controlled_nodes_migrated = sum(
                    1 for n in controlled_nodes
                    if old_assignments[n] != taxi_dag_assignments[taxi_id][n]
                )
                entry_node_migration_count += sum(
                    1 for n in entry_nodes
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
                        epoch_loss_history.append(info["loss"])
                        entropy_history.append(info["entropy"])
                        epoch_entropy_history.append(info["entropy"])
                        epsilon = max(epsilon_min, epsilon * epsilon_decay)
                    global_step += 1
                epsilon_history.append(epsilon)
                touch_taxi_last(taxi_last, taxi_id, row, current_lon, current_lat, ts)
            pbar.update(1)
        pbar.close()

        epoch_counters = {
            "decision_count": decision_count,
            "total_migrations": total_migrations,
            "stay_action_count": stay_action_count,
            "agent_decision_count": agent_decision_count,
            "candidate_action_counts": candidate_action_counts,
            "cost_guard_blocked_count": cost_guard_blocked_count,
            "reactive_action_clipped_count": reactive_action_clipped_count,
            "epsilon_greedy_samples": epsilon_greedy_samples,
            "argmax_samples": argmax_samples,
            "mask_only_stay_agent_count": mask_only_stay_agent_count,
            "agents_with_migrate_option_count": agents_with_migrate_option_count,
            "decisions_with_migrate_option_count": decisions_with_migrate_option_count,
            "legal_migrate_slots_sum": legal_migrate_slots_sum,
            "total_cost_ms_sum": total_cost_ms_sum,
            "severe_sla_violations": severe_sla_violations,
            "sla_excess_distance_history": sla_excess_distance_history,
            "p1_multi_migration_suppressed_count": p1_multi_migration_suppressed_count,
            "soft_cf_bias_count": soft_cf_bias_count,
            "soft_cf_neg_bias_count": soft_cf_neg_bias_count,
            "sla_gate_no_violation_decisions": sla_gate_no_violation_decisions,
            "sla_gate_violation_decisions": sla_gate_violation_decisions,
            "sla_gate_forced_stay_agents": sla_gate_forced_stay_agents,
            "cf_gate_blocked_slots": cf_gate_blocked_slots,
            "cf_gate_improving_slots": cf_gate_improving_slots,
            "cf_gate_forced_stay_agents": cf_gate_forced_stay_agents,
            "decisions_with_cf_improve_option_count": decisions_with_cf_improve_option_count,
            "reward_curriculum": epoch_curriculum,
        }
        epoch_stats_list.append(
            _snapshot_epoch_stats(
                epoch,
                is_eval_epoch,
                epsilon,
                epoch_counters,
                loss_history=epoch_loss_history,
                entropy_history=epoch_entropy_history,
            )
        )
        mig = epoch_counters["total_migrations"]
        stay_r = (
            epoch_counters["stay_action_count"] / epoch_counters["agent_decision_count"]
            if epoch_counters["agent_decision_count"] > 0 else 0.0
        )
        acts = epoch_counters["candidate_action_counts"]
        print(
            f"  [GAT-MARL epoch {epoch + 1}/{num_epochs}"
            f"{' EVAL' if is_eval_epoch else ''}] "
            f"migrations={mig} stay_ratio={stay_r:.4f} "
            f"actions={{0:{acts.get('0', 0)}, 1:{acts.get('1', 0)}, "
            f"2:{acts.get('2', 0)}, 3:{acts.get('3', 0)}}} "
            f"eps_greedy={epsilon_greedy_samples} "
            f"migrate_opt_decisions={decisions_with_migrate_option_count}/{decision_count}"
            + (f" p1_suppressed={p1_multi_migration_suppressed_count}" if p1_mode else "")
            + (f" soft_cf={soft_cf_bias_count}" if soft_cf_mode else "")
            + (
                f" sla_gate={sla_gate_no_violation_decisions}/{decision_count}"
                if sla_gate_mode else ""
            )
            + (
                f" cf_gate={cf_gate_blocked_slots}/{cf_gate_improving_slots}"
                if cf_sla_gain_gate_mode else ""
            ),
            flush=True,
        )

        # 按 epoch 保存检查点，避免一次崩溃导致全部丢失
        if (not inference_mode) and save_checkpoint_path and epoch < num_epochs - 1:
            epoch_ckpt_path = save_checkpoint_path.replace(".pth", f"_epoch_{epoch}.pth")
            _save_marl_checkpoint(epoch_ckpt_path, encoder, actor, critic)

    if (not inference_mode) and save_checkpoint_path:
        _save_marl_checkpoint(save_checkpoint_path, encoder, actor, critic)

    final_counters = _new_epoch_counters()
    final_counters.update({
        "total_decision_time": total_decision_time,
        "decision_count": decision_count,
        "total_migrations": total_migrations,
        "total_violations": total_violations,
        "primary_entry_violations": primary_entry_violations,
        "max_entry_violations": max_entry_violations,
        "severe_sla_violations": severe_sla_violations,
        "total_sla_excess_distance_km": total_sla_excess_distance_km,
        "sla_excess_distance_history": sla_excess_distance_history,
        "proactive_decisions": proactive_decisions,
        "total_reward_sum": total_reward_sum,
        "total_access_latency": total_access_latency,
        "total_communication_cost": total_communication_cost,
        "total_internal_path_ms": total_internal_path_ms,
        "total_migration_cost": total_migration_cost,
        "migration_decision_count": migration_decision_count,
        "total_cost_ms_sum": total_cost_ms_sum,
        "total_sla_penalty_ms": total_sla_penalty_ms,
        "total_tearing_penalty_ms": total_tearing_penalty_ms,
        "total_future_penalty_ms": total_future_penalty_ms,
        "local_migration_cost_sum": local_migration_cost_sum,
        "edge_split_cost_sum": edge_split_cost_sum,
        "dense_distance_bonus_sum": dense_distance_bonus_sum,
        "entry_sla_bonus_sum": entry_sla_bonus_sum,
        "proactive_logit_bias_count": proactive_logit_bias_count,
        "proactive_best_bias_action_counts": proactive_best_bias_action_counts,
        "reactive_action_clipped_count": reactive_action_clipped_count,
        "counterfactual_score_sum": counterfactual_score_sum,
        "entry_node_migration_count": entry_node_migration_count,
        "sla_improving_action_count": sla_improving_action_count,
        "cost_guard_blocked_count": cost_guard_blocked_count,
        "non_entry_distance_only_blocked_count": non_entry_distance_only_blocked_count,
        "invalid_action_masked_count": invalid_action_masked_count,
        "action_mask_fallback_count": action_mask_fallback_count,
        "agent_decision_count": agent_decision_count,
        "controlled_agent_decision_count": controlled_agent_decision_count,
        "pinned_agent_decision_count": pinned_agent_decision_count,
        "stay_action_count": stay_action_count,
        "all_agents_migrated_decisions": all_agents_migrated_decisions,
        "controlled_migrations": controlled_migrations,
        "controlled_all_agents_migrated_decisions": controlled_all_agents_migrated_decisions,
        "candidate_action_counts": candidate_action_counts,
        "joint_action_distribution": joint_action_distribution,
        "dag_complexity_stats": dag_complexity_stats,
        "dag_type_stats": dag_type_stats,
        "dag_proactive_stats": dag_proactive_stats or defaultdict(
            lambda: {"proactive_decisions": 0, "migrated_nodes": 0}
        ),
        "mask_only_stay_agent_count": mask_only_stay_agent_count,
        "agents_with_migrate_option_count": agents_with_migrate_option_count,
        "legal_migrate_slots_sum": legal_migrate_slots_sum,
        "decisions_with_migrate_option_count": decisions_with_migrate_option_count,
        "epsilon_greedy_samples": epsilon_greedy_samples,
        "argmax_samples": argmax_samples,
        "p1_multi_migration_suppressed_count": p1_multi_migration_suppressed_count,
        "soft_cf_bias_count": soft_cf_bias_count,
        "soft_cf_neg_bias_count": soft_cf_neg_bias_count,
        "sla_gate_no_violation_decisions": sla_gate_no_violation_decisions,
        "sla_gate_violation_decisions": sla_gate_violation_decisions,
        "sla_gate_forced_stay_agents": sla_gate_forced_stay_agents,
        "cf_gate_blocked_slots": cf_gate_blocked_slots,
        "cf_gate_improving_slots": cf_gate_improving_slots,
        "cf_gate_forced_stay_agents": cf_gate_forced_stay_agents,
        "decisions_with_cf_improve_option_count": decisions_with_cf_improve_option_count,
    })
    reset_curriculum()
    return _build_marl_return_dict(
        final_counters,
        loss_history=loss_history,
        reward_history=reward_history,
        epsilon_history=epsilon_history,
        entropy_history=entropy_history,
        lambda_migration_history=lambda_migration_history,
        lambda_split_history=lambda_split_history,
        epoch_stats=epoch_stats_list,
        collect_dag_proactive_stats=collect_dag_proactive_stats,
    )
