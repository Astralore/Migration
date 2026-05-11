# 微服务迁移算法对比实验报告（全量流水线）

**生成时间**：2026-05-09 20:24:31  
**流程**：启动前已删除 `sac_proactive.pth` / `sac_reactive.pth`（若存在）→ **训练**（仅 `train_df`）→ 保存新权重 → **推理**（仅 `test_df`，同一划分）加载新权重评测 Hybrid SAC。

**数据协议（Strategy B）**：数据协议: load_data(active_users_limit=100, min_vehicle_points=100) + default_rng(42) 80/20 by taxi_id; train_taxis=80, test_taxis=20

**工程上下文（与指标相关）**：

- **物理与奖励**：`total_cost_ms`（接入/迁移/tearing/通信/future/SLA）与 `context` 触发解耦（Reactive 空间 + QoS）。
- **Hybrid SAC**：离散 Actor **Logits 动作掩码**（合法 NEAREST、全非法回退 FOLLOW_SA），训练 `num_epochs=6`。
- **性能路径**：`core/geo.py`、`context.py`、`reward.py`、`state_builder.py` **NumPy 向量化**（Haversine 批量、近邻 `argpartition` 等），降低仿真墙钟时间但不改变公式。

**墙钟时间**：训练阶段约 **77790 s**，推理阶段约 **3052 s**。

---

## 一、训练段结果（train_df）

### Proactive（上表）/ Reactive（下表）

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|---------------------|------------------|---------------------|
| SA | 94 | 52118 | 0 | 12.73 | 7.79 |
| DQN | 16814 | 25482 | 0 | 1.11 | 5389.20 |
| Hybrid SAC | 96 | 76282 | 0 | 2.13 | 7.64 |

| Algorithm | Migrations | Violations | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|------------------|---------------------|
| SA | 88 | 53083 | 9.51 | 7.68 |
| DQN | 17104 | 29497 | 1.09 | 4734.98 |
| Hybrid SAC | 15 | 69352 | 2.94 | 3.17 |


---

## 二、测试段推理结果（test_df）

*Hybrid SAC 与 DQN 在测试段加载训练段保存的 checkpoint；SA 无磁盘权重，在测试段按既有脚本逻辑运行。*

### Proactive（上表）/ Reactive（下表）

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|---------------------|------------------|---------------------|
| SA | 23 | 19336 | 0 | 32.17 | 5.68 |
| DQN | 2448 | 11216 | 0 | 3.12 | 1088.78 |
| Hybrid SAC | 37 | 30902 | 0 | 3.03 | 8.50 |

| Algorithm | Migrations | Violations | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|------------------|---------------------|
| SA | 20 | 15896 | 20.77 | 4.97 |
| DQN | 1365 | 14861 | 1.56 | 1199.63 |
| Hybrid SAC | 1 | 32179 | 2.88 | 2.55 |


---

## 三、Hybrid SAC 泛化对比（训练 → 测试）

- **Proactive Violations**（训练段 → 测试段）: 76282 → 30902

- **Proactive Migrations**（训练段 → 测试段）: 96 → 37

- **Reactive Violations**（训练段 → 测试段）: 69352 → 32179

- **Reactive Migrations**（训练段 → 测试段）: 15 → 1


**测试段决策时延（Hybrid SAC）**：

- Proactive：**3.03 ms**（训练段末次 eval 统计：**2.13 ms**）
- Reactive：**2.88 ms**（训练段：**2.94 ms**）

---

## 四、时延对比（测试段 Proactive：SAC vs SA）

- Hybrid SAC: **3.03 ms**；SA: **32.17 ms**；比值 SA/SAC ≈ **10.6x**

## 五、Proactive 按 DAG 自适应迁移统计（SA 与 Hybrid SAC 同口径）

**统一条件**：已启用前瞻（`use_proactive`）；`get_trigger_type(...) == PROACTIVE`；单次决策内在 **同一拓扑序 `sorted_nodes`** 上比较 `previous_assignments` 与决策后节点放置，统计发生变更的节点数 `migrated_nodes_count`；按 **DAG Name**（`dag_type`）聚合 `proactive_decisions` 与 `migrated_nodes`；**Avg = migrated / proactive_decisions**（保留两位小数）。

**开关对齐（仅推理实验）**：Hybrid SAC 仅在 `inference_mode=True` 时分配并写入；SA 仅在 `run_inference_phase` 的 Proactive 分支传入 `collect_dag_proactive_stats=True`。**训练阶段**两种算法均不采集本统计。

### SA（Simulated Annealing）

| DAG Name | Proactive Decisions | Total Migrated Nodes | Avg Nodes per Decision |
|----------|---------------------|----------------------|------------------------|
| *（无样本）* | — | — | — |

### Hybrid SAC

| DAG Name | Proactive Decisions | Total Migrated Nodes | Avg Nodes per Decision |
|----------|---------------------|----------------------|------------------------|
| *（无样本）* | — | — | — |

---

*报告由 `run_comparison.py --pipeline` 自动生成*
