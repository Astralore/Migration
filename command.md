# 第八阶段：修复 Hybrid SAC 的 NEAREST 坍缩与 SA 引导断裂

## 一、当前全量实验结论

最新全量实验目录：`experiments/full_pipeline_20260512_111740_cov50_stage7_full`。

实验已经正常完成，`exit_code=0`，不是训练中断，也不是旧权重污染。本轮权重与结果均保存在独立目录。

第七阶段的 `SA_STAY` 动态 mask 实际已经生效：

- Proactive 训练评估：`eval_follow_sa_stay = 0`
- Proactive 推理：`eval_follow_sa_stay = 0`
- Reactive 训练/推理：`eval_follow_sa_stay = 0`

因此，当前问题不是“FOLLOW_SA but SA_STAY 仍然很高”，而是相反：`FOLLOW_SA` 被过度削弱，Hybrid SAC 坍缩到 `NEAREST` 动作。

关键证据：

- Proactive 训练评估动作分布：`[STAY=28777, FOLLOW_SA=0, NEAREST=82664]`
- Proactive 推理动作分布：`[STAY=7958, FOLLOW_SA=0, NEAREST=37228]`
- Proactive 推理中 Hybrid SAC 迁移 `4820`，违规 `10041`，平均成本 `23239.26`
- 同一阶段 DQN 违规 `5542`，平均成本 `20445.27`
- 同一阶段 SA 违规 `5734`，平均成本 `23181.02`

结论：第七阶段修复成功剥离了无效 `FOLLOW_SA`，但引入了新的动作坍缩：Actor 几乎不再使用 SA 建议迁移，转而大量选择 `NEAREST`，导致过激迁移、DAG 多节点迁移和违规上升。

## 二、根因判断

### 1. BC 监督样本被错误限制

当前代码中 `has_bc_target = used_bc_target`，只有当训练阶段通过概率强制执行 `FOLLOW_SA` 时，replay transition 才带 BC loss。

这会导致一个严重问题：Actor 只在“被强制模仿”的少量样本上学习 `FOLLOW_SA`，而不是在所有“SA 确实建议迁移”的样本上学习 `FOLLOW_SA`。随着 `bc_prob_schedule` 衰减，BC 样本迅速减少，Actor 很容易把 `FOLLOW_SA` 丢掉。

应改为：

- `bc_target_action = FOLLOW_SA` 当且仅当 `sa_proposed_server != current_node_server`
- `has_bc_target = (bc_target_action == ACTION_FOLLOW_SA)`
- `used_bc_target` 只表示本次行为动作是否被 BC 强制，不应决定是否写入监督目标

### 2. `NEAREST` 没有动作级约束

当前 `NEAREST` 只要存在候选服务器就合法，且在 `SA_STAY` 时 `FOLLOW_SA` 被 mask 后，动作空间经常变成 `[STAY, NEAREST]`。如果 critic 早期高估 `NEAREST`，Actor 会快速偏向最近服务器。

这在 Proactive 推理中已经出现：

- `FOLLOW_SA=0`
- 所有迁移都由 `NEAREST` 导致
- 多个 DAG 的 proactive 平均迁移节点数达到 `3-5`

这说明 `NEAREST` 已从“兜底候选动作”变成了主策略。

### 3. Q-Filter 对 SA 引导仍偏激进

Proactive 训练中：

- `q_filter_checked = 24448`
- `q_filter_blocked = 6248`

Q-Filter 已经阻断了约 25.6% 的 BC 目标。考虑到 critic 在早期和 sparse DAG reward 下仍有噪声，这会进一步削弱 `FOLLOW_SA` 学习。

### 4. `STAY` 监督不足

第七阶段为避免过度模仿 SA_STAY，屏蔽了 `FOLLOW_SA` when SA_STAY，这是正确的。但当前没有补充 `STAY` 的轻量监督，导致 Reactive 训练/推理中也几乎没有 `STAY`：

- Reactive 训练评估：`[STAY=0, FOLLOW_SA=6209, NEAREST=61748]`
- Reactive 推理：`[STAY=0, FOLLOW_SA=1184, NEAREST=26645]`

这说明 actor 对“什么情况下不迁移”仍然没有学好。

## 三、必须修改的代码方案

目标文件：`algorithms/hybrid_sac.py`

### 任务 1：修复 replay 中 BC target 的写入条件

当前训练节点 transition 中不要再用 `used_bc_target` 作为 `has_bc_target`。

应调整为：

```python
has_bc_target = (bc_target_action == ACTION_FOLLOW_SA)
```

并在 transition 中写入：

```python
"used_bc_action": used_bc_target,
"has_bc_target": has_bc_target,
"bc_target_action": bc_target_action,
```

说明：

- `used_bc_action` 只用于日志统计：实际动作是否由 BC 强制产生
- `has_bc_target` 用于 loss：只要 SA 建议迁移，就给 actor 一个 `FOLLOW_SA` 监督信号
- 不要对 `SA_STAY` 写入 `FOLLOW_SA` 监督

预期效果：

- `FOLLOW_SA` 不应再长期为 0
- Proactive 推理中 `FOLLOW_SA` 应恢复到非零比例
- `NEAREST` 占比应下降

### 任务 2：增加轻量 STAY 稳定监督，但不能恢复 FOLLOW_SA 过模仿

当前 `SA_STAY` 时已经把 `FOLLOW_SA` mask 掉，这是正确的，必须保留。

建议增加一个可选的轻量 `STAY` 监督：

- 当 `sa_proposed_server == current_node_server`
- 且 `trigger_type == TRIGGER_REACTIVE` 或 risk_ratio 较低
- 且当前 action mask 中 `ACTION_STAY` 合法

则可设置：

```python
has_stay_bc_target = True
bc_target_action = ACTION_STAY
```

但 STAY BC 权重要显著低于 FOLLOW_SA，例如：

- `follow_sa_bc_scale = 10.0`
- `stay_bc_scale = 1.0` 或 `2.0`

避免重新形成“永远不迁移”的坍缩。

实现约束：

- 如果 `optimize_sac` 继续沿用当前逐 transition 累加 loss 的写法，可以在单条 transition 内按 `bc_target_action` 选择对应 scale。
- 如果后续将 `optimize_sac` 改成 batch 张量化实现，不能用单个 `if/else` 给整个 batch 套同一个 BC 权重。必须构造逐样本权重张量，例如使用 `torch.where(bc_targets == ACTION_FOLLOW_SA, follow_sa_bc_scale, stay_bc_scale)`，再与 per-sample cross entropy 相乘。
- Cross entropy 必须使用 `reduction="none"` 得到逐样本 loss，再乘以逐样本 scale，最后再求均值或求和。
- 这样才能保证同一个 batch 内 `FOLLOW_SA` 与 `STAY` 两类监督各自使用正确权重，同时避免低效或错误的 Python batch 级分支。

### 任务 3：给 NEAREST 增加动作级诊断与软约束

不要直接硬禁用 `NEAREST`，否则会丢失兜底能力。应先加软约束和诊断。

在 `optimize_sac` 中增加可选 actor regularization：

```python
nearest_prob = action_probs[ACTION_NEAREST]
nearest_reg_loss = nearest_reg_weight * nearest_prob
```

建议初始：

- Proactive：`nearest_reg_weight = 0.02 ~ 0.05`
- Reactive：`nearest_reg_weight = 0.01 ~ 0.03`

仅在以下条件施加：

- `FOLLOW_SA` 当前合法
- `sa_proposed_server != current_node_server`
- 即 SA 已经给出迁移建议时，不鼓励 Actor 无脑绕过 SA 直接选 NEAREST

不要在 `FOLLOW_SA` 被 mask 的 SA_STAY 场景下惩罚 NEAREST，否则 `[STAY, NEAREST]` 的兜底选择会被破坏。

实现约束：

- `nearest_prob` 必须来自 Actor 当前前向传播得到的 `action_probs[ACTION_NEAREST]`，并保留计算图。
- 计算 `nearest_reg_loss` 时严禁对 `action_probs` 或 `nearest_prob` 使用 `.detach()`。
- 该正则项必须加入 `actor_loss`，并通过 actor 的反向传播真实更新 Actor 参数。
- 只允许在日志统计或 Q-Filter 比较中使用 detached 概率；用于正则训练的概率不能 detached。

### 任务 4：推迟或收紧 Q-Filter

当前 Q-Filter 在第 3 个训练 epoch 就会启动，且已经阻断大量 BC。

应调整为更保守：

```python
q_filter_enabled = (
    epoch >= 3
    and 0.0 < current_bc_loss_weight < 0.8
    and len(memory) >= 5 * batch_size
)
```

并将 `q_filter_margin` 从 `0.25` 提高到 `0.5` 或 `1.0`。

含义：只有 critic 明显认为 BC 目标差时才阻断，否则继续保留 SA 引导。

必须继续记录：

- `q_filter_checked`
- `q_filter_passed`
- `q_filter_blocked`
- `q_filter_blocked_ratio`

若 `q_filter_blocked_ratio > 0.2` 且 `FOLLOW_SA` 低于 5%，说明 Q-Filter 仍过强，需要继续推迟或关闭。

### 任务 5：增加动作概率与 Q 值诊断

当前只有最终动作计数，不足以判断是 actor logits 坍缩还是 critic Q 高估。

需要增加以下统计，至少在 eval/inference 返回：

- `eval_action_prob_sums`
- `eval_action_prob_means`
- `eval_q_sums`
- `eval_q_means`
- `eval_sa_migrate_action_counts`
- `eval_sa_stay_action_counts`

其中：

- `eval_sa_migrate_action_counts`：只统计 `sa_proposed_server != current_node_server` 时的动作分布
- `eval_sa_stay_action_counts`：只统计 `sa_proposed_server == current_node_server` 时的动作分布

这两个指标是判断修复是否成功的关键：

- SA 建议迁移时，`FOLLOW_SA` 应恢复为主要候选之一
- SA 建议不迁移时，`FOLLOW_SA` 必须保持 0
- `NEAREST` 不应在两类场景中同时占绝对多数

## 四、建议的验证标准

先做小规模 smoke，再做中等规模验证，最后再全量。

### Smoke 验证

使用 cov50 数据抽取 3-6 辆车，训练 `num_epochs=6`。

必须检查：

- `eval_follow_sa_stay == 0`
- `eval_action_counts` 中 `FOLLOW_SA > 0`
- `eval_sa_migrate_action_counts` 中 `FOLLOW_SA > 0`
- `NEAREST` 不再超过 80%
- `q_filter_blocked_ratio` 不超过 20%

### 中等规模验证

复用 cov50 validation 脚本，结果单独保存。

重点看 Proactive 推理：

- Hybrid SAC 违规不应显著高于 SA/DQN
- Hybrid SAC 平均成本不应高于 SA 10% 以上
- `FOLLOW_SA` 必须非零
- `FOLLOW_SA but SA_STAY` 必须继续为 0
- DAG proactive 平均迁移节点数不应普遍达到 `3-5`

### 全量验证

只有中等规模通过后再运行全量。不得覆盖旧实验目录和旧权重。

## 五、禁止事项

- 不要取消 `SA_STAY` 动态 mask。
- 不要让 `FOLLOW_SA` 在 `sa_proposed_server == current_node_server` 时重新合法。
- 不要简单通过硬禁用 `NEAREST` 解决坍缩。
- 不要让 Q-Filter 在 warmup 早期启用。
- 不要复用上一轮 checkpoint 做本轮验证。