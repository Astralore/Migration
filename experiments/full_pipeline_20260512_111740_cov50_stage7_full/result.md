# 全量 cov50 训练与推理实验报告

**生成时间**：2026-05-12 21:09:58  
**输出目录**：`experiments\full_pipeline_20260512_111740_cov50_stage7_full`  
**数据文件**：`data\processed\taxi_cleaned_active100_min100_eps2h_cov50.csv`  
**数据规模**：Top-100 active taxis；train=148153 rows/80 taxis；test=57783 rows/20 taxis  
**切分方式**：greedy_balanced_by_nearest_server_exposure；test taxi ratio=0.200；test row ratio=0.281；test risk ratio=0.275  
**SAC epochs**：Proactive=6，Reactive=2  
**Checkpoint 目录**：`experiments\full_pipeline_20260512_111740_cov50_stage7_full\checkpoints`

## 训练段

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|---------------------|------------------|---------------------|
| SA | 6544 | 15946 | 2405 | 17.58 | 23662.01 |
| DQN | 23114 | 16380 | 2165 | 1.24 | 23554.17 |
| Hybrid SAC | 20112 | 26004 | 1784 | 2.57 | 26999.15 |

| Algorithm | Migrations | Violations | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|------------------|---------------------|
| SA | 6536 | 16001 | 13.58 | 28579.95 |
| DQN | 24299 | 16738 | 0.94 | 28204.47 |
| Hybrid SAC | 10236 | 15945 | 2.62 | 32993.68 |


## 推理段

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|---------------------|------------------|---------------------|
| SA | 1624 | 5734 | 623 | 17.64 | 23181.02 |
| DQN | 1386 | 5542 | 544 | 1.21 | 20445.27 |
| Hybrid SAC | 4820 | 10041 | 492 | 2.56 | 23239.26 |

| Algorithm | Migrations | Violations | Avg Latency (ms) | Avg Total Cost (ms) |
|-----------|------------|------------|------------------|---------------------|
| SA | 1592 | 5756 | 13.16 | 26218.98 |
| DQN | 2392 | 5785 | 0.71 | 23785.65 |
| Hybrid SAC | 1860 | 5724 | 2.76 | 25022.52 |


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
| Compute_Heavy_DAG | 135 | 0 | 0.00 |
| Data_Heavy_DAG | 28 | 0 | 0.00 |
| Diamond_DAG_1 | 28 | 0 | 0.00 |
| Diamond_DAG_2 | 29 | 0 | 0.00 |
| FanIn_Aggregator_1 | 5 | 3 | 0.60 |
| FanIn_Aggregator_2 | 95 | 65 | 0.68 |
| FanIn_Aggregator_3 | 61 | 36 | 0.59 |
| FanOut_Broadcaster_1 | 6 | 0 | 0.00 |
| FanOut_Broadcaster_2 | 134 | 0 | 0.00 |
| FanOut_Broadcaster_3 | 45 | 0 | 0.00 |
| IoT_Lightweight_DAG | 57 | 0 | 0.00 |

### Hybrid SAC

| DAG Name | Proactive Decisions | Total Migrated Nodes | Avg Nodes per Decision |
|----------|---------------------|----------------------|------------------------|
| Diamond_DAG_1 | 10 | 10 | 1.00 |
| Diamond_DAG_2 | 17 | 68 | 4.00 |
| Diamond_DAG_3 | 55 | 220 | 4.00 |
| FanIn_Aggregator_1 | 75 | 370 | 4.93 |
| FanIn_Aggregator_2 | 7 | 35 | 5.00 |
| FanOut_Broadcaster_1 | 68 | 340 | 5.00 |
| FanOut_Broadcaster_2 | 135 | 670 | 4.96 |
| FanOut_Broadcaster_3 | 62 | 248 | 4.00 |
| IoT_Lightweight_DAG | 63 | 189 | 3.00 |
