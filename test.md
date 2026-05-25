# GAT-MARL 奖励函数与约束软化重构记录

本文档覆盖记录本轮针对 GAT-MARL 微服务迁移算法的三阶段重构内容。目标是将原先由大量硬编码规则控制的策略空间，收敛为更清晰的核心优化目标：**SLA 服务质量** 与 **迁移代价**。

本轮没有运行新的中等规模实验，只完成代码重构、语法检查和 lint 检查。后续需要通过中等规模实验验证指标变化。

## 一、重构动机

当前算法相比 SA 的主要短板是：SLA Risk、Severe SLA Violations 和 P95 SLA Excess 不占优。之前为避免过迁移加入了大量 hard guard，例如：

- `PROACTIVE_MAX_MIGRATABLE_MB = 100.0`
- heavy proactive 节点直接 block
- stateful 节点 future gain 折扣
- 轻量入口绿色通道固定阈值
- topology / split / future / dense bonus 等多项 reward shaping

这些规则在工程上能防止迁移爆炸，但也会限制强化学习探索空间，使策略更像“规则筛选后的学习”，而不是端到端学习 SLA 与迁移成本之间的权衡。

因此本轮重构遵循三个原则：

1. **主奖励做减法**：只围绕 SLA penalty 和 migration cost 建模。
2. **硬 size guard 软化**：不再用 `size > 100MB` 一刀切，而是用连续非线性迁移成本表达。
3. **训练目标与 action-time score 对齐**：counterfactual score 也改为 SLA gain 与非线性迁移成本之间的权衡。

## 二、阶段一：重构核心奖励函数

修改文件：

- `core/reward.py`
- `core/marl_reward.py`

### 2.1 原奖励结构

原始 shared reward 近似为：

```text
total_cost_ms =
  access_latency_ms
  + migration_delay_ms
  + tearing_delay_ms
  + communication_delay_ms
  + future_delay_ms
  + sla_penalty_ms

reward =
  -log1p(total_cost_ms / REWARD_COST_SCALE_MS)
  + distance_or_recovery_bonus
```

MARL agent reward 又进一步叠加：

```text
agent_reward =
  shared_reward
  + dense_distance_bonus
  + entry_sla_bonus
  - lambda_migration * local_migration_cost
  - lambda_split * local_edge_split_cost
```

问题是目标项过多，训练信号不够聚焦。

### 2.2 新 shared reward

本轮将训练 reward 的核心目标改为：

```text
reward_objective_ms =
  severity_aware_sla_penalty_ms
  + CORE_MIGRATION_REWARD_WEIGHT * nonlinear_migration_cost_ms

reward =
  -log1p(reward_objective_ms / REWARD_COST_SCALE_MS)
```

也就是说，训练主目标只保留：

- SLA 服务质量代价
- 非线性迁移代价

`access_latency_ms`、`tearing_delay_ms`、`communication_delay_ms`、`future_delay_ms` 仍然保留在 `details` 中作为诊断指标，但不再直接进入训练主奖励。

### 2.3 SLA penalty 改为 severity-aware

新增常量：

```python
SLA_QUADRATIC_PENALTY_PER_KM2_MS = 80.0
```

新的 SLA penalty：

```text
sla_penalty =
  SLA_BASE_PENALTY_MS
  + excess_km * SLA_PENALTY_PER_KM_MS
  + excess_km^2 * SLA_QUADRATIC_PENALTY_PER_KM2_MS
```

目的：

- 普通轻微超距仍然是线性惩罚；
- 严重超距会受到更高惩罚；
- 让 reward 更直接关注 `Severe SLA Violations` 和 `P95 SLA Excess`。

### 2.4 迁移成本改为 size/state-aware 非线性成本

新增常量：

```python
CORE_MIGRATION_REWARD_WEIGHT = 0.5
MIGRATION_SIZE_REF_MB = 100.0
MIGRATION_SIZE_ALPHA = 0.75
MIGRATION_SIZE_POWER = 3.0
MIGRATION_STATE_ALPHA = 0.5
NONLINEAR_MIGRATION_COST_CLIP_MS = 300000.0
```

新增函数：

```python
migration_size_state_multiplier(image_mb, state_mb)
calculate_nonlinear_migration_cost_ms(raw_migration_ms, image_mb, state_mb)
```

非线性迁移成本形式：

```text
nonlinear_migration_cost =
  raw_migration_cost
  * (
      1
      + alpha * (transfer_mb / 100)^power
      + beta * log(1 + state_mb / 100)
    )
```

目的：

- 重节点不再被 hard block；
- 重节点会自然产生更高迁移成本；
- stateful 节点也会通过连续惩罚体现额外风险；
- 80MB、120MB、300MB 节点之间不再是一刀切差异，而是连续差异。
- 非线性迁移成本会被截断到 `300000ms`，避免极端重节点被过早封顶成“保护伞”，同时依靠 `log1p` 控制训练 reward 数值范围。

### 2.5 MARL local reward 简化

`core/marl_reward.py` 中做了以下调整：

- `_local_migration_costs()` 改为返回非线性迁移成本。
- `dense_distance_bonus` 默认关闭。
- `entry_sla_bonuses` 改为空字典，即不再通过局部手写 bonus 特批入口节点。
- agent reward 简化为：

```text
agent_reward =
  shared_reward
  - lambda_migration * nonlinear_local_migration_cost
```

`lambda_split` 参数暂时保留在接口和日志中，避免破坏调用链，但不再进入 local reward 主项。

## 三、阶段二：软化 size guard 与绿色通道

修改文件：

- `algorithms/marl_gat.py`

### 3.1 移除 hard size block

原逻辑中存在多处重节点硬过滤：

```text
if is_heavy_proactive and sla_gain_ms < PROACTIVE_HEAVY_SLA_GAIN_FLOOR_MS:
    block
```

本轮已取消这类直接 block：

- Proactive logit bias 中不再因为 `is_heavy_proactive` 直接跳过动作。
- Proactive post-action guard 不再专门 block heavy node。
- Reactive clip 中不再使用 `size_guard_enabled + is_heavy_proactive` 直接过滤动作。

新的逻辑是：

```text
重节点是否值得迁移，由 nonlinear_migration_cost 决定，而不是 hard if 决定。
```

### 3.2 轻量入口 rescue 不再使用 100MB 硬阈值

原逻辑：

```python
node in entry_nodes
and transfer_mb <= PROACTIVE_MAX_MIGRATABLE_MB
and state_mb <= 0
```

新逻辑：

```python
node in entry_nodes
and state_mb <= 0
```

说明：

- 不再用 `transfer_mb <= 100MB` 判定能否进入入口 rescue。
- size 的影响交给非线性迁移成本。
- stateful 仍然保留为较强约束，因为迁移状态数据通常风险更高。

### 3.3 保留的 hard constraint

以下约束仍然保留，因为它们属于物理/合法性约束，而不是策略偏好：

- external node 不可迁移；
- non-deployable node 不可迁移；
- invalid action mask；
- candidate server 合法性；
- action=0 表示 stay。

这些约束不应交给 DRL 学习，否则只会浪费探索。

## 四、阶段三：对齐 counterfactual score 与新 reward

修改文件：

- `algorithms/marl_gat.py`

### 4.1 原 counterfactual score

原打分结构为：

```text
score =
  sla_gain_ms * CF_SLA_WEIGHT
  + future_gain_ms * CF_FUTURE_WEIGHT
  + topology_gain_ms * CF_TOPOLOGY_WEIGHT
  - migration_cost_ms * lambda_migration
  - split_cost_ms * lambda_split
```

这个 score 和新 reward 目标不完全一致，仍然包含 topology/split 等弱项。

### 4.2 新 counterfactual score

本轮改为：

```text
effective_sla_gain_ms =
  sla_gain_ms + 0.5 * future_gain_ms

score =
  effective_sla_gain_ms * CF_SLA_WEIGHT
  - nonlinear_migration_cost_ms * lambda_migration / CF_COST_SCALE_MS
```

即 action-time score 也只关注：

- 当前/预测 SLA 收益；
- 非线性迁移成本。

`topology_gain_ms`、`split_cost_ms`、`migration_cost_ms` 仍保留在返回字典中用于诊断，但不再作为主打分项。

### 4.3 为什么仍保留 future gain

虽然主目标是 SLA + migration，但 Proactive 需要提前感知未来 SLA 风险。因此 `future_gain_ms` 没有作为独立第三目标，而是折算进入 `effective_sla_gain_ms`：

```text
future_gain 是未来 SLA gain，不是额外目标。
```

这样保持 Proactive 特性，同时避免 reward 项继续膨胀。

## 五、本轮实际改动文件

### 5.1 `core/reward.py`

完成内容：

- 新增 severity-aware SLA quadratic penalty。
- 新增 size/state-aware migration multiplier。
- 新增 nonlinear migration cost。
- shared reward 改用 `reward_objective_ms`。
- 保留原始 `total_cost_ms`、`migration_cost`、`tearing_penalty`、`future_penalty` 等诊断字段。

### 5.2 `core/marl_reward.py`

完成内容：

- local migration cost 改为 nonlinear migration cost。
- 默认关闭 dense distance bonus。
- entry SLA bonus 置零。
- local reward 中移除 split cost 主惩罚。

### 5.3 `algorithms/marl_gat.py`

完成内容：

- counterfactual score 使用 nonlinear migration cost。
- 移除 proactive / reactive 中对 heavy node 的硬 block。
- 入口 rescue 不再使用 `transfer_mb <= 100MB` 的硬阈值。
- counterfactual score 与新 reward 对齐为 SLA gain vs nonlinear migration cost。

## 六、当前仍保留的启发式

虽然本轮已经软化了 size guard，但仍保留少量启发式：

- `LIGHTWEIGHT_ENTRY_SCORE_FLOOR`
- `LIGHTWEIGHT_ENTRY_MIN_SLA_GAIN_MS`
- `LIGHTWEIGHT_ENTRY_MIN_BIAS`
- `CF_SLA_ENTRY_SCORE_FLOOR`
- Reactive 中 violating/bottleneck entry fallback

这些仍然是策略辅助项。后续如果新 reward 表现稳定，可以继续减少这些特批规则。

## 六点五、迁移成本压低补丁

在 `reward_refactor_v1` 后续实验中，GAT-MARL Proactive 的 Severe SLA 与 P95 指标接近或优于 SA，但迁移成本仍明显偏高。因此进一步加入 Proactive action-time 成本控制。

随后根据实验发现，成本瓶颈不是单步迁移数量，而是重型节点的非线性成本汇率仍然偏低。因此进一步提高物理软墙：

```python
MIGRATION_SIZE_POWER = 3.0
NONLINEAR_MIGRATION_COST_CLIP_MS = 300000.0
CORE_MIGRATION_REWARD_WEIGHT = 0.5
max_lambda_migration = 0.40  # Proactive
```

目的：

- 让 300MB/500MB 以上节点的迁移成本呈更强非线性增长；
- 避免 `50000ms` clip 过早封顶导致超大节点仍可能越过 counterfactual score；
- 让 critic 在训练阶段更明确地感知物理迁移成本；
- 保持 `CF_SLA_WEIGHT = 0.003` 不变，避免一次性削弱 SLA 修复能力。

新增常量：

```python
PROACTIVE_MIGRATION_BUDGET_MS = 8000.0
MIN_SLA_GAIN_COST_RATIO = 0.15
HIGH_COST_MIGRATION_MS = 5000.0
HIGH_COST_MIN_SLA_GAIN_MS = 3000.0
```

修改位置：

- `algorithms/marl_gat.py::_apply_proactive_size_guard`

新逻辑：

1. 对 actor 已选择的 Proactive 迁移动作重新计算 counterfactual score。
2. 过滤掉以下动作：
   - `score <= 0`
   - 没有正向 SLA gain
   - `effective_sla_gain / nonlinear_migration_cost < 0.25` 且收益不足 3000ms
   - `nonlinear_migration_cost > 5000ms` 但收益不足 3000ms
3. 对通过筛选的动作按 score 排序。
4. 按 score 从高到低保留动作，只要累计非线性迁移成本不超过 `8000ms`。
5. 超出单次 Proactive 成本预算的动作置为 STAY。

目的：

- 不恢复 `size > 100MB` 的硬拦截；
- 保留高价值迁移探索空间；
- 限制单步迁移成本，而不是限制固定迁移数量；
- 抑制低收益、高成本的 Proactive 迁移。

## 七、预期影响

正向预期：

- GAT-MARL 的探索空间变大，不再被 `size > 100MB` 直接截断。
- 重节点仍会因为非线性迁移成本而不容易被选择。
- SLA severe / P95 指标应该更受训练目标影响。
- 训练目标、局部 reward、counterfactual score 更一致。

潜在风险：

- 短期内迁移次数可能上升。
- 之前调好的绿色通道参数可能不再最优。
- `lambda_migration` 可能需要重新调参，因为 local migration cost 已经变成 nonlinear cost。
- 中等规模实验结果可能和之前不可直接横向比较，需要作为新 reward 体系下的新 baseline。

## 八、验证情况

已完成：

```text
python -m py_compile core/reward.py core/marl_reward.py algorithms/marl_gat.py
```

结果：

```text
Syntax OK
```

Lint 检查：

```text
No linter errors found.
```

尚未完成：

- 未运行新的中等规模验证。
- 未重新调 `lambda_migration`。
- 未重新评估绿色通道参数。

## 九、建议下一步实验

建议下一步先跑一轮新的中等规模实验，实验标识可使用：

```text
medium_validation_reward_refactor_v1
```

重点观察：

- `SLA Risk Count`
- `Severe SLA Violations`
- `Avg SLA Excess`
- `P95 SLA Excess`
- `Migrations`
- `Avg Migration Cost`
- `Avg Total System Cost`
- `Avg Migrated Agents / Decision`
- `Stay Action Ratio`

判断重点：

1. SLA severe 和 P95 是否改善；
2. 迁移是否出现爆炸；
3. GAT-MARL 相比 SA 的差距是否缩小；
4. 相比 DQN / Nearest 是否继续保持迁移成本优势。

## 十、当前阶段结论

本轮重构完成了从“硬规则护栏 + 多项 reward shaping”到“核心目标建模 + 非线性成本软约束”的第一版转换。

新的核心逻辑是：

```text
训练奖励：SLA penalty + nonlinear migration cost
动作打分：SLA gain - nonlinear migration cost
重节点约束：连续成本惩罚，而不是 hard block
绿色通道：入口 SLA 辅助，而不是 size 一刀切特批
```

这更符合端到端强化学习的思想，也更容易在论文中解释为：

> 在 SLA 服务质量与迁移代价之间进行可学习的连续权衡，而不是依赖大量人工硬规则筛选动作。
