# 边缘微服务迁移：DAG、调用关系、奖励与指标设计全面总结

**版本**：2026-05-27  
**代码真源**：`core/microservice_dags.py`、`core/dag_utils.py`、`core/reward.py`、`core/marl_reward.py`、`core/context.py`、`core/physics_utils.py`、各 `algorithms/*.py`、`run_medium_validation_cov50.py`  
**关联文档**：[Reward_v2奖励函数设计总结.md](./Reward_v2奖励函数设计总结.md)、[边缘环境微服务迁移算法全面总结_20260521.md](./边缘环境微服务迁移算法全面总结_20260521.md)

---

## 1. 文档目的与范围

本文档对当前代码库中**与算法决策直接相关**的设计做统一说明，覆盖：

1. **DAG 数据模型**及其在仿真中的生命周期  
2. **服务调用边（edges）**如何进入成本、状态与优化  
3. **奖励函数 v1 / v2** 与 **物理总成本** 的区分  
4. **SLA 违约判定、惩罚项**及实验指标口径  
5. **各算法**（Nearest / SA / DQN / GAT-MARL）对上述抽象的使用差异  

阅读后应能回答：「一次迁移决策里，DAG 如何表示、入口 SLA 如何算、训练 reward 与报表 `total_cost_ms` 为何不同、报表里的 Violations 指什么。」

---

## 2. 系统抽象：从轨迹到 DAG 放置

```text
出租车 GPS 轨迹 (taxi_id, lat, lon, time)
        │
        ▼
每车绑定一种 DAG 类型 (概率抽样) + 全节点 server 放置
        │
        ▼
每个时间点：用户位置 vs 服务入口 → 是否触发迁移 (REACTIVE / PROACTIVE)
        │
        ▼
算法输出新的 per-node server 分配
        │
        ▼
calculate_microservice_reward() → reward (训练) + details (物理分解)
        │
        ▼
仿真循环聚合 → total_cost_ms_sum、total_violations、…
```

**关键分离**：

| 概念 | 用途 |
|------|------|
| `total_cost_ms` | 实验报告、SA 优化目标；真实毫秒量纲 |
| `reward` / `reward_objective_ms` | RL 训练信号；v1 为 log 压缩，v2 为 SLA+非线性迁移的线性负值 |
| `total_violations` | 离散 SLA **风险事件**计数（max-entry），**不是** `sla_penalty_ms` 的累加 |

---

## 3. 微服务 DAG 数据模型

### 3.1 模板库

`core/microservice_dags.py` 定义 `MICROSERVICE_DAGS`：18 种从生产调用图提取的模板，按类型命名，例如：

| 类型前缀 | 拓扑含义（示意） |
|----------|------------------|
| `FanIn_Aggregator_*` | 多上游 → 单聚合有状态服务 |
| `FanOut_Broadcaster_*` | 单广播源 → 多下游有状态服务 |
| `Diamond_DAG_*` | 菱形依赖（分叉再汇合） |
| `Pipeline_Chain_*` | 链式流水线 |
| `Data_Heavy_DAG_*` / `Compute_Heavy_DAG_*` | 数据/计算偏重子图 |

每个模板结构：

```python
{
    "probability": float,           # 新车分配 DAG 时的权重
    "nodes": {
        "MS_xxxxx": {
            "image_mb": int,        # 容器镜像，影响迁移传输
            "state_mb": int,        # 状态数据，有状态服务迁移更重
            "is_stateful": bool,
        },
        ...
    },
    "edges": {
        (src_ms, dst_ms): traffic,  # 调用强度（原始 trace 计数尺度）
        ...
    },
}
```

- **`traffic`**：边上的调用量权重，用于归一化通信代价、GAT 邻接权重、节点 traffic 特征。  
- **跨模板共享节点名**（如 `MS_37691`）表示同一微服务在不同业务 DAG 中复用，但每车只实例化一种 DAG，放置独立。

### 3.2 从原始调用图到 DAG（离线）

`core/dag_complexity.py::dagify_component_edges()` 说明离线处理原则：

- 对弱连通子图按确定性节点序定向，**保留前向边**，丢弃回边/环边（记入 `dropped` 供溯源）。  
- 仿真运行时模板已为 DAG；`topological_sort()` 若仍遇环，会把未排序节点**稳定追加**到末尾，保证所有服务仍可被控制。

### 3.3 复杂度分桶（分析用）

`classify_dag_complexity()` 按节点数、边数、平均度分为 `simple` / `medium` / `complex`，用于 `cost_by_dag_complexity` 等实验切片，**不参与**奖励公式本身。

---

## 4. DAG 拓扑与调用关系处理

### 4.1 节点角色

| 角色 | 判定 | 算法行为 |
|------|------|----------|
| **可部署服务** | `get_deployable_nodes()`：非 external | 可迁移、计入 Migrations、参与 MARL 动作 |
| **External 上下文** | `USER` / `UNKNOWN` / `UNAVAILABLE` | 保留拓扑语义，**不可**迁移；MARL 中 reward 等于 shared_reward |
| **服务入口 (service entry)** | `get_service_entry_nodes()` | SLA 接入、触发、访问延迟、惩罚的瓶颈口径 |

External 节点当前模板库中**通常不出现**于 `nodes` 键，但 `get_service_entry_nodes` 已为「外部 → 可部署」边预留逻辑。

### 4.2 服务入口判定（`get_service_entry_nodes`）

优先级：

1. **外部源直接调用的可部署节点**（`is_external_node(src)` 且 `dst` 可部署）  
2. 否则：**无内部入边**的可部署节点（`deployable - deployable_with_internal_incoming`）  
3. 仍为空：拓扑序中**第一个**可部署节点  

**多入口 DAG** 使用 **max-entry** 口径：

- `d_max = max_{e ∈ entry} dist(user, server(assign[e]))`  
- `access_latency_ms = calc_access_latency_ms(d_max)`  
- 任一入口超距或超 QoS 即视为当前 DAG 存在 SLA 风险（与 `calculate_entry_sla_metrics` 一致）

### 4.3 Gateway 与触发

各算法主循环中：

- `gateway = entry_nodes[0]`（**主入口**）  
- `get_trigger_type(..., gateway_server_lat/lon, ...)` 用**主入口**所在服务器做 TTV / Reactive 判断  
- **违规统计**用 `calculate_entry_sla_metrics(entry_nodes, ...)` 的 **max_entry**，可覆盖非 gateway 的其它入口  

即：**触发**偏保守（单 gateway），**报表 Violations** 偏严格（任一入口）。

### 4.4 调用边在成本模型中的用法

对每条边 `(src, dst)`，流量 `t`，当前放置 `assign`：

| 情形 | 代价 |
|------|------|
| `assign[src] == assign[dst]` | 边代价为 0（同机本地调用） |
| 跨服务器 | **Tearing**：`cross_mb = min(t × RPC_SIZE_MB, MAX_TEARING_MB)`，`tearing_ms = cross_mb / EDGE_BACKHAUL_MBPS × 1000` |
| 跨服务器 | **Comm**：`norm_t = t / max_traffic`，`comm_ms = norm_t × (edge_dist_km / FIBER_SPEED_KM_MS + BASE_ROUTER_DELAY_MS)` |

常量（`core/reward.py`）：

| 常量 | 值 | 含义 |
|------|-----|------|
| `RPC_SIZE_MB` | 0.005 | 单次 RPC 折算 payload |
| `MAX_TEARING_MB` | 50 | 单边撕裂流量上限 |
| `EDGE_BACKHAUL_MBPS` | 1000 | 边缘回传带宽 |
| `FIBER_SPEED_KM_MS` | 200 | 传播速度 km/ms |
| `BASE_ROUTER_DELAY_MS` | 2 | 路由/协议固定延迟 |

**设计含义**：算法不能把 DAG 简单「整体搬到最近服务器」；拆分会同时产生撕裂与跨机 RPC 延迟，traffic 越大的边权重越高。

### 4.5 调用边在状态与 MARL 中的用法

**GAT-MARL**（`build_marl_graph_state`）：

- 邻接矩阵：`(i,j)` 权重 = `traffic / max_traffic`，无向对称，对角线为 1  
- 节点特征：累加关联边 traffic、邻居同机比例 `same_ratio`、入度/出度、拓扑序、是否 sink、是否 entry  

**DQN**（`build_node_state`）：

- 单节点 16 维向量；拓扑维含邻居同机比例、是否 entry、DAG 规模  

**SA 邻域**：

- 不直接解析边结构搜索，但 `total_cost_ms` 通过 `calculate_microservice_reward` **隐式**包含边拆分代价  

**MARL 局部分解**（`marl_reward._local_edge_split_costs`）：

- 仅对**本步相对上一步发生变化**的边，计算拆分代价增量，均摊到可部署端点  

---

## 5. 物理层：接入、带宽与未来风险

### 5.1 接入时延

```text
access_latency_ms(d) = d / FIBER_SPEED_KM_MS + BASE_ROUTER_DELAY_MS
                     = d / 200 + 2
```

奖励中使用 **所有 entry 的最大距离** 对应的延迟。

### 5.2 SLA 空间阈值与 QoS

| 符号 | 来源 | 值 |
|------|------|-----|
| `DISTANCE_THRESHOLD_KM` | `core/context.py` | 15.0 km |
| `USER_SLA_TOLERANCE_MS` | `calc_access_latency_ms(15) × 0.99` | 略低于 15 km 对应延迟 |

**违约（布尔）**：`dist > 15 km` **或** `access_latency_ms > USER_SLA_TOLERANCE_MS`（二者 OR，互补）。

### 5.3 迁移有效带宽（风险耦合）

```text
ρ = min(d_max / D_th, 1)
B_eff = B_min + (B_max - B_min) × ρ²     # 50 ~ 500 Mbps
```

SLA 风险越高，模型假设可用迁移带宽越高（紧急场景资源倾斜）；单节点带宽还要除以**同目标机并发迁移数**。

### 5.4 线性迁移时延（进入 `total_cost_ms`）

对每个发生迁移的可部署节点 `i`：

```text
δ_i^raw = (S_i × 8 / B_i) × 1000 + BASE_MIGRATION_OVERHEAD_MS
```

- `S_i = image_mb + state_mb`，`8` 为 MB→Mbit  
- `BASE_MIGRATION_OVERHEAD_MS = 200`  
- `TRIGGER_REACTIVE` 时：`δ_i^raw × 1.5`（`REACTIVE_MIGRATION_MULT`）

### 5.5 未来风险项 `future_delay_ms`

若有 `predicted_locations`（长度 H）：

- 对每个预测点，算所有 entry 到用户的距离，取 **max**  
- `excess_h = max(0, d_h - FUTURE_DIST_THRESHOLD)`，`FUTURE_DIST_THRESHOLD = 15 km`  
- 权重 `w_h = FUTURE_DECAY^h`（`FUTURE_DECAY = 0.9`），超额按传播速度折算后加权平均  

该项进入 **`total_cost_ms`**；Proactive **触发**则走 TTV 逻辑（见下节），与该项计算独立。

---

## 6. SLA 惩罚设计

### 6.1 超额量定义

```text
E_d = max(0, d_max - D_th)                                    # km
E_q = max(0, L_acc - L_qos) × FIBER_SPEED_KM_MS               # 折算为 km 等价
```

### 6.2 Reward v1（默认 `REWARD_SCHEME=v1`）

无违规时 `P_SLA = 0`。否则：

```text
E_eq = E_d + E_q
P_SLA = SLA_BASE_PENALTY_MS
      + E_eq × SLA_PENALTY_PER_KM_MS
      + E_eq² × SLA_QUADRATIC_PENALTY_PER_KM2_MS
```

默认：`2000 + 500·E_eq + 80·E_eq²`（ms）。

**特点**：含固定底座 2000 ms；距离与 QoS 超额**合并**为 `E_eq` 后惩罚。

### 6.3 Reward v2（`REWARD_SCHEME=v2`）

无违规时 `P_SLA = 0`。否则：

```text
P_SLA = α · E_d² + β · E_q²
```

默认 `α = β = 80` ms/km²；**无** base、**无**线性项；距离与 QoS **分开**二次。

环境变量：`REWARD_V2_SLA_ALPHA_MS_PER_KM2`、`REWARD_V2_QOS_BETA_MS_PER_KM2`。

### 6.4 改善量（Guard / Counterfactual）

| 函数 | 用途 |
|------|------|
| `sla_penalty_gain_ms(old_d, new_d, ...)` | `max(0, P_old - P_new)`，按当前 scheme 计算 |
| `future_mean_excess_penalty_gain_ms` | 预测平均超额距离的 penalty 下降；v2 为二次差分 |

### 6.5 非线性迁移成本（训练目标的一部分）

在 `δ^raw` 之上：

**v1**：

```text
mult = 1 + 0.75·(S/100)³ + 0.5·log1p(state/100)
C_i^nl = min(δ_i^raw · mult, 300000)
J 中权重：CORE_MIGRATION_REWARD_WEIGHT = 0.75
```

**v2**：

```text
M_exp = exp(min(S/τ, c_max))     # τ=100 MB, c_max=8
C_i^nl = min(λ · δ_i^raw · M_exp, 300000)    # λ=0.75
```

---

## 7. 奖励函数：训练目标 vs 评测总成本

### 7.1 物理总成本（所有算法报表一致）

```text
total_cost_ms =
    access_latency_ms
  + migration_delay_ms          # 线性 δ^raw 之和
  + tearing_delay_ms
  + comm_delay_ms
  + future_delay_ms
  + sla_penalty_ms
```

SA 显式最小化 `details["total_cost_ms"]`（`algorithms/sa.py::_sa_total_cost_ms`）。

### 7.2 训练用目标 `reward_objective_ms`

| Scheme | `reward_objective_ms` |
|--------|------------------------|
| v1 | `P_SLA + 0.75 × Σ C_i^nl` |
| v2 | `P_SLA + Σ C_i^nl` |

### 7.3 标量 `reward`

| Scheme | 公式 |
|--------|------|
| v1 | `reward = -log1p(reward_objective_ms / 1000) + reward_bonus` |
| v2 | `reward = -max(reward_objective_ms, 0) / S + reward_bonus`，`S = REWARD_V2_OBJECTIVE_SCALE_MS`（默认 10000） |

当前 `REWARD_RECOVERY_BONUS_MAX = 0`、`REWARD_DISTANCE_BONUS_WEIGHT = 0`，bonus 实际为 0。

**重要**：不能把 `reward` 当作 `-total_cost_ms`；v2 训练甚至**不把** access/tearing/comm/future 放进 `J`，但仍写入 `details` 供报表。

### 7.4 MARL 多智能体分解（`core/marl_reward.py`）

```text
shared_reward = calculate_microservice_reward(...).reward

agent_reward_i =
    shared_reward
  + dense_distance_bonus_i      # 默认关闭
  + entry_sla_bonus_i           # 已置 0，SLA 由 shared penalty 表达
  - λ_mig · (local_migration_cost_i / scale)
```

- `λ_mig`：`lambda_schedule_by_epoch` 从 10% 渐增至 `max_migration`（默认 0.15）  
- v2 时 `scale = REWARD_V2_OBJECTIVE_SCALE_MS`，与 counterfactual 对齐  
- `training_reward`：所有 agent reward 的均值  

`local_edge_split_costs` 已实现增量拆分代价，但当前 `lambda_split` 在训练路径中**未减到 agent_reward**（保留在 details 供分析）。

---

## 8. 触发机制：Reactive 与 Proactive

实现：`core/context.py::get_trigger_type`。

### 8.1 Reactive

当前已违规 → 返回 `REACTIVE`：

```text
spatial:  dist(gateway) > 15 km
qos:      calc_access_latency_ms(dist) > USER_SLA_TOLERANCE_MS
```

### 8.2 Proactive（TTV）

1. 若已 Reactive，不再 Proactive。  
2. 对预测轨迹上每个点算到 gateway 的距离 `fd[h]`。  
3. `ttv_s` = 首次满足空间或 QoS 违规的预测时间（步长 `forecast_step_dt_sec`，默认 60s）。  
4. `estimated_migration_time_s`：默认 `estimate_dag_migration_time_s`（保守估计全 DAG 可部署节点迁移时间），失败兜底 120s。  
5. 若 `ttv_s ≤ est_mig_s + max(1, step_dt_s)` → `PROACTIVE`。

**注意**：`PROACTIVE_WARNING_KM = 5` 与 `check_proactive_sla_violation()` 为**旧接口**，主仿真**不**再以其作为 Proactive 依据。

### 8.3 模式语义

| 配置 | 含义 |
|------|------|
| `proactive=True` + predictor | 允许 TTV 触发；**不等于**每次决策都是 Proactive |
| `proactive_decisions` | 计数 `trigger_type == PROACTIVE` 的决策次数 |
| `decision_count` | 所有触发迁移的决策（含 Reactive） |

---

## 9. 各算法对 DAG 与调用的使用

| 算法 | DAG 决策方式 | 优化/学习目标 | 调用关系显式利用 |
|------|--------------|---------------|------------------|
| **Nearest** | 触发后将**全部**可部署节点迁到最近服务器 | 无搜索；成本仅事后统计 | 通过 reward 间接惩罚拆分 |
| **SA** | 单节点换机或全体 colocate 到某候选；模拟退火最小化 `total_cost_ms` | 物理总成本 | 同上 |
| **DQN** | 每节点独立 4 动作（STAY + 3 候选） | `-log1p(J/1000)` 或 v2 线性 reward | 状态含邻居同机比；无联合 DAG 动作 |
| **GAT-MARL** | 图编码 + 每节点 4 动作；CTDE critic | shared_reward + 局部迁移惩罚 schedule | 邻接矩阵 + counterfactual 边拆分增量 |

### 9.1 GAT-MARL 决策护栏（摘要）

环境变量可配（`algorithms/marl_gat.py`）：

| 机制 | 作用 |
|------|------|
| `PROACTIVE_MIGRATION_BUDGET_MS` | 单步 Proactive 迁移代价预算 |
| `PROACTIVE_MAX_MIGRATIONS_PER_DECISION` | 每步最大迁移节点数（≤0 表示仅预算/ROI） |
| Counterfactual scoring | `sla_gain - λ·migration_cost` 过滤低 ROI 动作 |
| `non_entry_distance_only_blocked` | 非入口仅缩短距离、无 SLA 增益时拦截 |

---

## 10. 实验指标定义与口径

聚合逻辑在各 `run_*_microservice*` 与 `run_medium_validation_cov50.py::_summarize_result` 中统一。

### 10.1 核心成本指标

| 指标 | 定义 |
|------|------|
| `total_cost_ms_sum` | 每次触发决策的 `details.total_cost_ms` 累加 |
| `avg_total_cost_ms` | `total_cost_ms_sum / decision_count` |
| `total_access_latency` | Σ `access_latency_ms` |
| `total_migration_cost` | Σ 线性 `migration_delay_ms` |
| `total_communication_cost` | Σ `comm_delay_ms` |
| `total_tearing_penalty_ms` | Σ tearing |
| `total_future_penalty_ms` | Σ future |
| `total_sla_penalty_ms` | Σ `sla_penalty_ms` |
| **Migration Share** | `total_migration_cost / total_cost_ms_sum` |

### 10.2 SLA 与风险指标

| 指标 | 定义 |
|------|------|
| **`total_violations`** | **主表**：每决策若 `max_entry_violation==1` 则 +1（任一 entry 超距或超 QoS） |
| `primary_entry_violations` | 仅 `entry_nodes[0]` 违规次数（诊断） |
| `max_entry_violations` | 与 total_violations 同口径累加 |
| `severe_sla_violations` | `sla_excess_distance_km > 5` 的决策次数 |
| `avg_sla_excess_distance_km` | 每决策 `max(0, d_max-15)` 的平均 |
| `p95_sla_excess_distance_km` | 上述超额距离的 95 分位 |
| `sla_violations`（details 内） | 当前步**违规入口个数**（可 >1） |

**区分**：

- `total_violations`：**事件次数**（0/1 per decision）  
- `total_sla_penalty_ms`：**连续惩罚毫秒**累加，含 v1 base 或 v2 二次项  

### 10.3 迁移与决策指标

| 指标 | 定义 |
|------|------|
| `total_migrations` | 可部署节点发生 server 变化的次数累加 |
| `decision_count` | 触发迁移的决策次数 |
| `proactive_decisions` | 其中 Proactive 触发次数 |
| `migration_decision_count` | `migration_cost > 0` 的决策数 |
| `avg_decision_time_ms` | 算法计算耗时（非接入延迟） |

### 10.4 综合 Score（`evaluation/metrics.py`）

```text
Score = total_migrations + weight × total_violations    # 默认 weight=0.5，越低越好
```

### 10.5 启用 Reward v2

```bash
export REWARD_SCHEME=v2
# 可选：REWARD_V2_OBJECTIVE_SCALE_MS、REWARD_V2_SLA_ALPHA_MS_PER_KM2 等
```

详见 [Reward_v2奖励函数设计总结.md](./Reward_v2奖励函数设计总结.md)。

---

## 11. 数据流总览

```text
 MICROSERVICE_DAGS[dag_type]
        │
        ├─► dag_utils: deployable / entry / topo_sort
        │
        ├─► context: get_trigger_type (gateway + TTV)
        │
        ├─► reward.calculate_microservice_reward
        │         ├─ edges → tearing + comm
        │         ├─ entries → access + P_SLA
        │         ├─ migration → δ^raw + C^nl
        │         └─► reward (train)  vs  total_cost_ms (eval)
        │
        ├─► marl_reward.calculate_marl_rewards (optional local penalty)
        │
        └─► algorithms aggregate → results.json / result.md
```

---

## 12. 关键文件索引

| 主题 | 文件 |
|------|------|
| DAG 模板 | `core/microservice_dags.py` |
| 拓扑工具 | `core/dag_utils.py`、`core/dag_complexity.py` |
| SLA / 触发 | `core/context.py`、`core/physics_utils.py` |
| 奖励与 SLA 惩罚 | `core/reward.py` |
| MARL 分解 | `core/marl_reward.py` |
| GAT 状态 | `core/marl_state_builder.py` |
| DQN 状态 | `core/state_builder.py` |
| SA / Nearest / DQN / MARL | `algorithms/sa.py`、`nearest.py`、`dqn.py`、`marl_gat.py` |
| 指标汇总 | `run_medium_validation_cov50.py`、`run_comparison.py`、`evaluation/metrics.py` |

---

## 13. 一句话归纳

当前系统将**生产调用图抽象为带 traffic 权重的微服务 DAG**，以 **service entry 的 max-distance SLA** 驱动触发与惩罚，以 **跨边 tearing/comm** 约束放置，用 **`total_cost_ms`（物理毫秒）** 做 SA/报表对齐、用 **v1 log 或 v2 线性 SLA+指数迁移目标** 做 RL 训练；实验上必须用 **`total_violations`（离散风险）** 与 **`total_sla_penalty_ms`（连续惩罚）** 分工解读，不可混为一谈。
