# MARL 阶段五：策略坍缩与过度迁移最终修复

## 一、当前暴露的问题

最新中等规模 8 epoch 实验说明，单纯增加训练轮数没有解决核心问题：

- Proactive：训练和推理都出现 `stay_action_ratio = 1.0`、`total_migrations = 0`，说明策略坍缩为全 STAY。
- Reactive：仍出现 `controlled_all_agents_migrated_ratio = 1.0`，说明策略在触发后倾向把所有可控微服务一起迁移。
- `dense_distance_bonus_sum` 很大，但没有带来 Proactive 迁移，说明它作为“动作后奖励”不能直接解决探索阶段看不到迁移动作的问题。
- 继续提高 lambda 不是稳健方案，之前已经出现过从“全搬”直接翻到“全不搬”的敏感振荡。

结论：这次必须同时修三类问题：Proactive 的动作前引导、Reactive 的 joint action 约束、以及 dense reward 的尺度失控。

## 二、本次已实施的修复

### 1. Proactive：降低默认 lambda，避免成本项压死探索

目标文件：`algorithms/marl_gat.py`

```python
if max_lambda_migration is None:
    # Proactive 由动作前 distance bias 引导迁移，lambda 只负责轻度成本约束。
    max_lambda_migration = 0.05 if use_proactive else 0.3
if max_lambda_split is None:
    max_lambda_split = 0.02 if use_proactive else 0.1
```

Proactive 的默认成本权重从偏大的 `0.5 / 0.2` 改为 `0.05 / 0.02`。迁移动作由候选距离 bias 负责引导，lambda 只保留轻度成本约束。

### 2. Proactive：加入动作前 counterfactual distance logit bias

目标文件：`algorithms/marl_gat.py`

核心思想：在 Actor softmax 前，对每个候选动作做反事实距离评估。如果候选服务器比当前服务器更接近用户，就给该候选动作增加一个小的 logit bias。

```python
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
    *,
    bias_scale=3.0,
    max_bias=2.5,
):
    biased_logits = masked_logits.clone()
    positive_bias_count = 0
    best_action_counts = defaultdict(int)

    for node in sorted_nodes:
        if is_external_node(node):
            continue
        node_idx = node_to_idx[node]
        current_server = assignments[node]
        current_dist = _server_distance_to_user(current_server, user_lat, user_lon, servers_info)
        best_action = 0
        best_improvement = 0.0
        for action in range(1, MARL_ACTION_DIM):
            if not bool(action_masks[node_idx, action].item()):
                continue
            target_server = action_to_server(action, candidates, current_server)
            target_dist = _server_distance_to_user(target_server, user_lat, user_lon, servers_info)
            improvement = max(0.0, current_dist - target_dist)
            if improvement <= 1e-6:
                continue
            bias = min(max_bias, bias_scale * improvement / max(DISTANCE_THRESHOLD_KM, 1e-6))
            biased_logits[node_idx, action] = biased_logits[node_idx, action] + float(bias)
            positive_bias_count += 1
            if improvement > best_improvement:
                best_improvement = improvement
                best_action = action
        best_action_counts[str(best_action)] += 1

    return biased_logits, positive_bias_count, dict(best_action_counts)
```

这解决的是 Proactive “奖励存在但动作采样阶段仍全 STAY”的问题。它不是替代 Actor，而是在合法 action mask 之后、softmax 之前提供物理先验。

### 3. Dense distance bonus：从毫秒级巨额奖励改为 reward 同尺度

目标文件：`core/marl_reward.py`

```python
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
    ...
    if trigger_type == TRIGGER_PROACTIVE:
        risk_factor = 1.0 + risk_ratio
        bonus_value = min(
            proactive_bonus_max,
            distance_reduction_km * proactive_bonus_per_km * risk_factor,
        )
    else:
        bonus_value = 0.0
```

原来的 `1000.0 / km` 会让 dense bonus 达到几十万级别，和 log-scaled shared reward 不在同一尺度。本次改为单节点最多 `6.0` reward units，避免训练不稳定。

### 4. Reactive：加入基于距离收益的 joint action top-k 裁剪

目标文件：`algorithms/marl_gat.py`

核心思想：Reactive 不能只靠 lambda 抑制全员迁移。Actor 先给出动作，然后只保留本 step 中“距离收益最高”的前 `2` 个迁移动作，其余迁移动作强制改为 STAY。

```python
def _clip_reactive_actions(
    actions,
    sorted_nodes,
    assignments,
    candidates,
    user_lat,
    user_lon,
    servers_info,
    *,
    max_migrations=2,
):
    scored = []
    for idx, (node, action) in enumerate(zip(sorted_nodes, actions)):
        if is_external_node(node) or int(action) == 0:
            continue
        current_server = assignments[node]
        target_server = action_to_server(action, candidates, current_server)
        current_dist = _server_distance_to_user(current_server, user_lat, user_lon, servers_info)
        target_dist = _server_distance_to_user(target_server, user_lat, user_lon, servers_info)
        improvement = current_dist - target_dist
        if improvement > 1e-6:
            scored.append((improvement, idx))

    keep = {idx for _, idx in sorted(scored, reverse=True)[:max_migrations]}
    clipped = 0
    clipped_actions = list(actions)
    for idx, (node, action) in enumerate(zip(sorted_nodes, actions)):
        if is_external_node(node) or int(action) == 0:
            continue
        if idx not in keep:
            clipped_actions[idx] = 0
            clipped += 1
    return clipped_actions, clipped
```

这会直接打断 Reactive 的 `controlled_all_agents_migrated_ratio = 1.0` 问题，同时保留最有物理收益的迁移动作。

### 5. 新增诊断指标

目标文件：`algorithms/marl_gat.py`、`run_medium_validation_cov50.py`、`run_full_pipeline_cov50.py`

新增结果字段：

- `proactive_logit_bias_count`：Proactive 中被正向 bias 的候选动作次数。
- `proactive_best_bias_action_counts`：反事实距离最优候选动作分布。
- `reactive_action_clipped_count`：Reactive 被 top-k 约束裁剪掉的迁移动作次数。

这些指标必须和原有 `controlled_migrations`、`controlled_all_agents_migrated_ratio`、`stay_action_ratio` 一起看。

## 三、已有阶段四修复仍保留

以下修复仍是必要基础，不应回退：

- `USER / UNKNOWN / UNAVAILABLE` 作为 external nodes：保留图编码，但不参与迁移动作。
- `get_service_entry_nodes(...)` 用于 SLA / trigger，避免 external nodes 污染入口判断。
- `GraphEncoder(node_feat_dim=14)` 保留 `is_external` 特征。
- shared reward 和 local reward 只对 deployable nodes 计算迁移成本。
- 并发带宽竞争惩罚保留，用来惩罚多个节点同时迁往同一目标服务器。
- `controlled_*` 指标作为主要迁移行为判断依据。

## 四、验证要求

1. 先运行编译检查，确认 `marl_gat.py`、`marl_reward.py`、中等规模脚本、全流程脚本无语法错误。
2. 跑小样本 Proactive smoke：
   - 预期 `proactive_logit_bias_count > 0`
   - 预期 `total_migrations > 0`
   - 预期 `stay_action_ratio < 1.0`
3. 跑小样本 Reactive smoke：
   - 预期 `reactive_action_clipped_count > 0`
   - 预期 `controlled_all_agents_migrated_ratio < 1.0`
4. 再跑中等规模验证，并删除旧的 `marl_gat_*.pth`，因为 `node_feat_dim=14` 与旧 checkpoint 不兼容。

## 五、禁止回退事项

- 不要再把 Proactive 默认 lambda 提回 `0.5 / 0.2`。
- 不要恢复 `proactive_bonus_per_km=1000.0` 这类毫秒级 dense bonus。
- 不要只靠增加 epoch 解决全 STAY。
- 不要只靠提高 lambda 解决 Reactive 全员迁移。
- 不要让 external nodes 参与迁移、迁移成本或 deployable entry SLA。