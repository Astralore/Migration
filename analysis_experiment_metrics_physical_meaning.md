# 微服务迁移实验指标计算与物理意义分析

## 1. 分析目的

本文档用于明确当前微服务迁移实验中各项对比指标的计算方式、实际物理意义，以及它们是否适合作为最终实验对比指标。

当前实验主要比较以下算法：

- SA
- Nearest
- DQN
- GAT-MARL

实验结果表中的核心指标包括：

- `Migrations`
- `Violations`
- `Proactive Decisions`
- `Avg Decision Time (ms)`
- `Avg Access Latency (ms)`
- `Avg Total System Cost (ms)`

此外，GAT-MARL 还额外输出了多智能体行为诊断指标，例如 `controlled_migrations`、`stay_action_ratio`、`controlled_all_agents_migrated_ratio`、`cost_by_dag_type` 等。

## 2. 主表指标说明

### 2.1 Migrations

**计算方式**

`Migrations` 表示一次决策前后，DAG 中节点部署位置发生变化的节点数量累计值。

也就是：

```python
nodes_migrated = sum(
    1 for node in sorted_nodes
    if old_assignments[node] != new_assignments[node]
)
```

**物理意义**

该指标表示微服务实例迁移次数。迁移次数越多，说明算法越激进，可能更快降低用户到服务的距离，但也会带来更高的镜像传输、状态迁移和服务扰动成本。

**合理性判断**

该指标总体合理。当前代码已完成口径统一：

- SA / Nearest / DQN 已从 `topological_sort(dag_info)` 改为 `get_deployable_nodes(dag_info)`。
- GAT-MARL 原本已区分可控 agent / external context node。
- 因此，后续新实验中的主表 `Migrations` 已统一为 deployable service migrations，不再统计 `USER`、`UNKNOWN`、`UNAVAILABLE` 等 external/context 节点。
- 旧实验结果不会自动重算；若用于论文正式对比，应重新运行或明确标注历史口径。

## 3. SLA 违规指标

### 3.1 Violations

**当前含义**

`Violations` 表示 SLA 违规次数。每次触发迁移决策时，如果服务入口距离用户超过阈值，则计一次违规。

当前系统中 SLA 距离阈值来自：

```python
DISTANCE_THRESHOLD_KM
```

GAT-MARL 当前已经改为 `max-entry` 口径：

```python
max_entry_violation = max(entry_distances) > DISTANCE_THRESHOLD_KM
```

即只要任意 service entry 超过阈值，就认为当前 DAG 存在 SLA 违规。

**物理意义**

该指标表示用户请求入口服务距离移动用户过远，可能导致访问延迟超过 SLA 要求。

对于多入口微服务 DAG，端到端体验通常受最差入口或关键入口短板影响，因此 `max-entry` 口径比单一 primary gateway 更严格，也更符合多入口 DAG 的短板效应。

**合理性判断与已对齐实现**

当前代码已将 SA / Nearest / DQN 的主表 `total_violations` 对齐为 `max-entry` 口径：只要任意 service entry 超过 `DISTANCE_THRESHOLD_KM`，该 DAG 决策即计为一次 SLA violation。同时额外保留 `primary_entry_violations`，用于和旧 primary gateway 口径做附表对照。

**建议**

最终实验报告应同时输出：

- `primary_entry_violations`
- `max_entry_violations`

并建议以 `max_entry_violations` 作为主 SLA 指标。

## 4. Proactive Decisions

**计算方式**

`Proactive Decisions` 表示触发类型为 `TRIGGER_PROACTIVE` 的决策次数。

```python
if trigger_type == TRIGGER_PROACTIVE:
    proactive_decisions += 1
```

**物理意义**

该指标表示算法在尚未真正违规之前，基于未来轨迹预测提前进行迁移决策的次数。

**合理性判断**

该指标合理。它可以说明预测模块是否真正参与了迁移决策。

但该指标不能单独代表算法效果，需要与以下指标联合分析：

- `Violations`
- `Migrations`
- `Avg Total Cost`
- `Future Penalty`

如果 proactive decisions 很多但 violations 没有降低，说明预测触发或迁移动作质量存在问题。

## 5. Avg Decision Time 与 Avg Access Latency

### 5.1 Avg Decision Time (ms)

**计算方式**

实验报告中的 `Avg Decision Time (ms)` 表示算法决策耗时：

```python
avg_decision_time_ms = total_decision_time / decision_count * 1000
```

**物理意义**

它表示算法每次触发迁移决策所需的计算时间，而不是用户访问延迟。

**合理性判断**

该指标作为工程运行开销是合理的。它可以反映：

- SA 搜索开销
- DQN 推理开销
- GAT-MARL 图编码和多智能体动作选择开销
- 反事实评分带来的额外计算开销

但它不应被解释为网络访问延迟，也不应作为唯一核心性能指标。

### 5.2 Avg Access Latency (ms)

**计算方式**

```python
avg_access_latency_ms = total_access_latency / decision_count
```

其中单次 `access_latency_ms` 来自：

```python
access_latency_ms = distance_km / 200.0 + 2.0
```

**物理意义**

它表示用户到 service entry 的真实接入延迟估计，不包含迁移传输、SLA penalty 或算法计算耗时。当前全量实验中该项约为 2-3ms，符合边缘服务器距离几十公里、光纤传播速度约 `200 km/ms` 的物理尺度。

## 6. Avg Total System Cost (ms)

**计算方式**

实验报告中的 `Avg Total System Cost (ms)` 由以下函数计算：

```python
avg_total_cost_ms = total_cost_ms_sum / decision_count
```

其中 `total_cost_ms` 来自 `core/reward.py::calculate_microservice_reward(...)`：

```python
total_cost_ms = (
    access_latency_ms
    + migration_delay_ms
    + tearing_delay_ms
    + comm_delay_ms
    + future_delay_ms
    + sla_penalty_ms
)
```

**物理意义**

该指标表示单次触发迁移决策的综合物理代价，包含：

- 用户到服务入口的访问延迟
- 微服务镜像和状态迁移时间
- DAG 拆分后的跨服务器通信代价
- 迁移过程中的 tearing / 同步代价
- 未来预测风险代价
- SLA 违规惩罚

**合理性判断**

该指标适合作为综合优化目标，但需要明确：它不是纯用户访问延迟，而是一个“物理代价 + 惩罚项”的综合成本。

因此，在论文或报告中更推荐称为：

- `Avg Total System Cost`
- `Avg Penalty Cost`
- `Avg System Cost`

不建议将其解释为单纯的访问时延。

## 7. 成本分解指标

### 7.1 total_access_latency

表示用户到 service entry 的访问延迟估计。

当前计算公式来自 `core/physics_utils.py`：

```python
access_latency_ms = distance_km / 200.0 + 2.0
```

其中 `200.0 km/ms` 是光纤有效传播速度量级，`2.0ms` 是基础路由/协议开销。因此只要用户到入口服务的距离在几十公里范围内，该项通常只有约 2ms 量级。

该指标物理意义清晰，适合作为访问性能分解项。

### 7.2 total_migration_cost

表示微服务迁移传输成本。当前计算考虑：

- `image_mb`
- `state_mb`
- 网络带宽
- 同一目标服务器上的并发迁移带宽竞争
- Reactive 模式下的额外迁移放大系数

该指标合理，是微服务迁移问题中的关键成本项。

### 7.3 total_tearing_penalty_ms

表示 DAG 被拆分到不同服务器后，跨服务器边产生的数据撕裂或同步代价。

该指标适合微服务 DAG 迁移问题，能够体现迁移导致服务拓扑被切分的副作用。

### 7.4 total_communication_cost

表示跨服务器 RPC 或服务调用通信延迟。

它和以下因素有关：

- 边 traffic
- 源服务与目标服务所在服务器距离
- 基础路由延迟

当前该项偏低的主要原因是：

- `RPC_SIZE_MB = 0.005`，单次 RPC 载荷被建模为很小的控制/服务调用消息。
- 通信延迟使用边流量归一化权重 `traffic / max_traffic`，不是直接把业务吞吐量全部折算为传输时间。
- 若一条边两端服务部署在同一服务器，通信成本为 0。
- 跨服务器时的传播延迟仍按 `distance_km / 200.0 + 2.0` 计算，因此通常只有少量 ms。

因此，该指标合理，但在当前参数下更像是“跨服务器调用延迟扰动”，不是大体量数据传输成本。大体量传输主要体现在 `migration_cost` 和 `tearing_penalty` 中。

### 7.5 total_future_penalty_ms

表示预测轨迹中未来 service entry 超过阈值的风险代价。

当前该项偏低的原因是：

- 它只在 proactive 且存在 `predicted_locations` 时计算；Reactive 中通常为 0。
- 只计算未来距离超过 `FUTURE_DIST_THRESHOLD` 的超额部分。
- 超额距离只按传播延迟折算：`excess / FIBER_SPEED_KM_MS`，没有叠加固定 `SLA_PENALTY_MS`。
- 未来步长还经过 `FUTURE_DECAY` 加权平均，进一步平滑了数值。

因此，该指标合理，但当前更适合作为辅助风险项，而不是主导成本项。

### 7.6 total_sla_penalty_ms

表示 SLA 违规惩罚总和。

当前每次 SLA 违规会增加固定惩罚：

```python
SLA_PENALTY_MS
```

该指标和 `Violations` 高度相关。报告中如果同时比较 `Violations` 和 `Avg Total Cost`，必须说明 `Avg Total Cost` 已包含 SLA penalty，因此二者不是完全独立指标。

## 8. GAT-MARL 诊断指标

### 8.1 controlled_migrations

只统计可控微服务节点迁移数，不包括 external/context 节点。

该指标比 `total_migrations` 更适合分析 GAT-MARL 行为。

### 8.2 controlled_all_agents_migrated_ratio

表示一次决策中所有可控节点都发生迁移的比例。

该指标用于检测多智能体策略是否出现“全员迁移”坍缩。

该指标非常重要，之前 Reactive 过度迁移问题就是通过该指标发现的。

### 8.3 stay_action_ratio

表示所有 agent 动作中 `STAY` 的比例。

该指标用于检测策略是否出现“全不迁移”坍缩。

高 STAY 不一定坏。如果 SLA 已经满足，高 STAY 代表策略保守且节省迁移成本；但如果 violations 很高且 STAY 也很高，则说明策略迁移不足。

### 8.4 candidate_action_counts

统计 `[STAY, CANDIDATE_1, CANDIDATE_2, CANDIDATE_3]` 的动作分布。

该指标用于判断策略是否退化为：

- 全 STAY
- 全迁移
- 单一 nearest candidate
- 只使用某一个候选服务器

该指标合理，适合分析 Proactive logit bias 是否过强。

### 8.5 proactive_logit_bias_count

表示 Proactive 模式下，被反事实评分正向引导的动作数量。

该指标用于确认 proactive guidance 是否实际生效。

### 8.6 proactive_best_bias_action_counts

表示反事实评分中最优候选动作的分布。

如果该指标全部集中在 `CANDIDATE_1`，说明策略可能退化为 nearest heuristic。

当前全量结果中该指标并未完全压向 `CANDIDATE_1`，说明当前 bias 没有完全替代 Actor。

### 8.7 reactive_action_clipped_count

表示 Reactive 模式下被 top-k / cost guard 裁剪掉的迁移动作数量。

该指标用于衡量防止过度迁移的约束是否生效。

### 8.8 entry_node_migration_count

表示迁移动作中 service entry 节点迁移数量。

该指标用于判断迁移是否真正作用在 SLA 关键入口上。

### 8.9 sla_improving_action_count

表示反事实评分中能够改善 SLA 的动作数量。

该指标是内部诊断指标，适合用于解释策略行为，但不建议作为最终算法优劣的主指标。

### 8.10 cost_guard_blocked_count

表示被成本护栏拦截的动作数量。

该指标用于衡量系统避免高成本迁移的力度。

### 8.11 cost_by_dag_type

按 DAG 类型统计：

- 决策次数
- 迁移次数
- 平均成本
- 全员迁移比例

该指标非常重要，适合用于分析问题是否集中在某些 DAG 类型上，例如：

- `Diamond_DAG_1`
- `FanIn_Aggregator_2`
- `FanOut_Broadcaster_2`
- `Compute_Heavy_DAG`

## 9. 当前指标体系的主要问题

当前最大问题曾经是：

**不同算法的 `Violations` 统计口径尚未完全统一。**

已完成的修正：

- GAT-MARL 已使用 `max_entry_violations` 作为 `total_violations`。
- SA / Nearest / DQN 已同步输出 `primary_entry_violations` 与 `max_entry_violations`。
- SA / Nearest / DQN 的 `total_violations` 已改为 `max_entry_violations`。
- 基线迁移数量统计也改为只统计可部署微服务节点，避免 external/context 节点干扰 `Migrations`。
- 中等规模与全量实验报告脚本已新增 `cost_decomposition_inference.png`，用于展示推理阶段成本分解堆叠图。
- 成本分解主图已收敛为 SLA penalty、Migration、Tearing、Communication 四项；Access/Future 因量级过小，保留在 `results.json` 和附表中。

因此，后续新生成的 `results.json` 和 `result.md` 中，主表 `Violations` 已具备公平对比口径；旧实验结果若未重新运行，则仍需在论文中注明其历史口径或重新生成。

## 10. 成本分解堆叠图

论文中不建议只报告 `Avg Total Cost`，因为它把多个物理来源合并成一个总数，无法解释算法到底是在减少 SLA 惩罚、迁移代价，还是通信/撕裂代价。

当前中等规模和全量实验脚本已新增推理阶段成本分解堆叠图：

```text
cost_decomposition_inference.png
```

图中每个柱子表示某个算法的单次决策平均成本，计算方式为：

```python
avg_component_cost = total_component_cost / decision_count
```

当前论文主图堆叠项进一步收敛为两个主导成本：

- `total_sla_penalty_ms`
- `total_migration_cost`

原因是全量实验中 `total_communication_cost`、`total_access_latency` 和 `total_future_penalty_ms` 的单次决策均值通常只有 0-3ms 量级，`total_tearing_penalty_ms` 通常约几十到一百 ms；而 `SLA penalty` 与 `Migration` 通常达到数千到数万 ms。若把这些小量级项放入主堆叠图，图例会变复杂，但视觉上几乎不可见，反而降低论文图表可读性。

论文正文建议重点解释：

- SLA penalty：服务质量不足导致的惩罚，是算法是否真正改善 SLA 的直接体现。
- Migration cost：微服务迁移带来的传输与状态搬迁代价。

`Tearing`、`Communication`、`Access` 和 `Future` 仍保留在 `results.json` 和指标附表中，不作为主图堆叠项。若需要专门讨论小量级开销，可以另做一个 appendix 图或表格，而不建议放在主图中。

## 11. 建议最终指标体系

正式报告建议主表使用：

| 指标 | 建议口径 | 说明 |
|---|---|---|
| Deployable Migrations | 只统计可部署服务节点 | 避免 external nodes 干扰 |
| Max-Entry SLA Violations | 所有 service entry 中任意一个超阈值即违规 | 符合多入口 DAG 短板效应 |
| Avg Access Latency (ms) | `total_access_latency / decision_count` | 真实接入延迟估计 |
| Avg Total System Cost (ms) | `total_cost_ms_sum / decision_count` | 综合物理代价与惩罚 |
| Avg Decision Time (ms) | 决策计算耗时 | 工程运行开销 |
| Proactive Decisions | proactive trigger 次数 | 预测机制参与度 |

附表建议保留：

- `Primary-Entry Violations`
- `Max-Entry Violations`
- `Migration Cost`
- `SLA Penalty`
- `Tearing Cost`
- `Communication Cost`
- `Controlled All-Migrated Ratio`
- `Stay Ratio`
- `Cost by DAG Type`

## 12. 总结

当前实验指标体系总体合理，能够覆盖：

- SLA 服务质量
- 迁移成本
- 通信和 DAG 拆分代价
- 算法决策耗时
- 多智能体策略行为

当前代码已统一新实验的 SLA violation 和 migration 统计口径；若沿用旧实验结果，则需要重新运行或在论文中标注旧结果可能存在历史统计口径差异。

最推荐的最终主指标是：

- `Deployable Migrations`
- `Max-Entry SLA Violations`
- `Avg Access Latency (ms)`
- `Avg Total System Cost (ms)`
- `Avg Decision Time (ms)`
- `Proactive Decisions`

其中，`Max-Entry SLA Violations` 应作为 SLA 质量的主指标，因为它最符合多入口微服务 DAG 的短板效应。
