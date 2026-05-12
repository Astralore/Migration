# 全量 cov50 训练与推理实验报告

**生成时间**：2026-05-12 04:33:58  
**输出目录**：`experiments\full_pipeline_20260511_1711_cov50_predictor_v2_full`  
**数据文件**：`data\processed\taxi_cleaned_active100_min100_eps2h_cov50.csv`  
**数据规模**：Top-100 active taxis；train=148153 rows/80 taxis；test=57783 rows/20 taxis  
**切分方式**：greedy_balanced_by_nearest_server_exposure；test taxi ratio=0.200；test row ratio=0.281；test risk ratio=0.275  
**SAC epochs**：Proactive=6，Reactive=2  
**Checkpoint 目录**：`experiments\full_pipeline_20260511_1711_cov50_predictor_v2_full\checkpoints`

## 训练段

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|---------------------|------------------|---------------------|
| SA | 6530 | 15974 | 2381 | 17.25 | 23570.86 |
| DQN | 21746 | 17021 | 2171 | 1.23 | 23117.09 |
| Hybrid SAC | 6740 | 16014 | 2391 | 3.26 | 22840.13 |

| Algorithm | Migrations | Violations | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|------------------|---------------------|
| SA | 6581 | 16038 | 25.56 | 28587.99 |
| DQN | 21715 | 16718 | 1.14 | 28279.17 |
| Hybrid SAC | 6704 | 16104 | 3.25 | 26113.24 |


## 推理段

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|---------------------|------------------|---------------------|
| SA | 1626 | 5736 | 617 | 37.58 | 23067.47 |
| DQN | 3701 | 8553 | 556 | 2.59 | 20530.48 |
| Hybrid SAC | 1543 | 5764 | 585 | 3.42 | 21026.74 |

| Algorithm | Migrations | Violations | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|------------------|---------------------|
| SA | 1573 | 5749 | 26.37 | 26252.00 |
| DQN | 4260 | 7273 | 1.71 | 24215.71 |
| Hybrid SAC | 1359 | 5767 | 3.44 | 22543.57 |


## 说明

- 本轮显式使用 cov50 清洗数据，不读取旧 cleaned CSV。
- DQN / Hybrid SAC checkpoint 均写入本实验目录，推理阶段只读取本轮新训练权重。
- 结果会在每个算法完成后写入 `results.json` 和本报告，便于长任务中断后检查进度。


## DAG Proactive 迁移统计Proactive 按 DAG 自适应迁移统计（SA 与 Hybrid SAC 同口径）

**统一条件**：已启用前瞻（`use_proactive`）；`get_trigger_type(...) == PROACTIVE`；单次决策内在 **同一拓扑序 `sorted_nodes`** 上比较 `previous_assignments` 与决策后节点放置，统计发生变更的节点数 `migrated_nodes_count`；按 **DAG Name**（`dag_type`）聚合 `proactive_decisions` 与 `migrated_nodes`；**Avg = migrated / proactive_decisions**（保留两位小数）。

**开关对齐（仅推理实验）**：Hybrid SAC 仅在 `inference_mode=True` 时分配并写入；SA 仅在 `run_inference_phase` 的 Proactive 分支传入 `collect_dag_proactive_stats=True`。**训练阶段**两种算法均不采集本统计。

### SA（Simulated Annealing）

| DAG Name | Proactive Decisions | Total Migrated Nodes | Avg Nodes per Decision |
|----------|---------------------|----------------------|------------------------|
| Compute_Heavy_DAG | 125 | 0 | 0.00 |
| Data_Heavy_DAG | 27 | 0 | 0.00 |
| Diamond_DAG_1 | 24 | 0 | 0.00 |
| Diamond_DAG_2 | 30 | 0 | 0.00 |
| FanIn_Aggregator_1 | 8 | 0 | 0.00 |
| FanIn_Aggregator_2 | 84 | 77 | 0.92 |
| FanIn_Aggregator_3 | 57 | 34 | 0.60 |
| FanOut_Broadcaster_1 | 11 | 0 | 0.00 |
| FanOut_Broadcaster_2 | 133 | 0 | 0.00 |
| FanOut_Broadcaster_3 | 52 | 0 | 0.00 |
| IoT_Lightweight_DAG | 66 | 0 | 0.00 |

### Hybrid SAC

| DAG Name | Proactive Decisions | Total Migrated Nodes | Avg Nodes per Decision |
|----------|---------------------|----------------------|------------------------|
| Diamond_DAG_1 | 11 | 0 | 0.00 |
| Diamond_DAG_2 | 15 | 0 | 0.00 |
| Diamond_DAG_3 | 91 | 0 | 0.00 |
| FanIn_Aggregator_1 | 71 | 54 | 0.76 |
| FanIn_Aggregator_2 | 12 | 4 | 0.33 |
| FanOut_Broadcaster_1 | 53 | 0 | 0.00 |
| FanOut_Broadcaster_2 | 149 | 0 | 0.00 |
| FanOut_Broadcaster_3 | 88 | 0 | 0.00 |
| IoT_Lightweight_DAG | 95 | 0 | 0.00 |
