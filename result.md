# 微服务迁移算法对比实验报告

**运行模式**：推理模式 (Inference)  
**数据范围**：[10000:15000]  
**生成时间**：2026-04-28 11:48:40

---

## 一、Proactive 模式结果

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Score |
|-----------|------------|------------|---------------------|------------------|-------|
| SA | 122 | 1010 | 1979 | 2.55 | 627.0 |
| DQN | 790 | 1409 | 770 | 0.00 | 1494.5 |
| Hybrid SAC | 1658 | 1004 | 461 | 0.77 | 2160.0 |

---

## 二、Reactive 模式结果

| Algorithm | Migrations | Violations | Avg Latency (ms) | Score |
|-----------|------------|------------|------------------|-------|
| SA | 83 | 1359 | 1.60 | 762.5 |
| DQN | 558 | 1277 | 0.00 | 1196.5 |
| Hybrid SAC | 488 | 1006 | 0.71 | 991.0 |

---

## 三、时延对比分析

- **Hybrid SAC 平均决策时延**: 0.77 ms
- **SA 平均决策时延**: 2.55 ms
- **加速比**: SAC 比 SA 快 **3.3x**

## 六、Hybrid SAC 拓扑自适应迁移行为分析 (Adaptive Migration)

**统计口径**：`inference_mode=True` 且触发类型为 **PROACTIVE** 时，对每个 `taxi_id` 的一次完整 DAG 决策，将 `previous_assignments` 与决策后 `current_assignments` 按节点比对得到 `migrated_nodes_count`，再按 **DAG Name**（`dag_type`）聚合。

| DAG Name | Proactive Decisions | Total Migrated Nodes | Avg Nodes per Decision |
|----------|---------------------|----------------------|------------------------|
| Compute_Heavy_DAG | 135 | 44 | 0.33 |
| Data_Heavy_DAG | 80 | 480 | 6.00 |
| Diamond_DAG_2 | 246 | 944 | 3.84 |

## 七、推理阶段算法对比摘要（Proactive 同一段数据）

| Algorithm | Violations | Proactive Decisions | Migrations | Avg Latency (ms) |
|-----------|------------|---------------------|------------|------------------|
| SA | 1010 | 1979 | 122 | 2.55 |
| DQN | 1409 | 770 | 790 | 0.00 |
| Hybrid SAC | 1004 | 461 | 1658 | 0.77 |

- **Hybrid SAC**：第六节给出 **按 DAG 的按需迁移强度**（Avg Nodes per Decision）；
- **SA**：启发式基线；
- **DQN**：当前脚本在推理段 **无磁盘权重**，表中为「该段数据上从头训练」的仿真结果，与 SAC **加载训练权重** 的推理 **不对等**，对比时建议以 **SA vs SAC** 为主或后续对齐 DQN checkpoint。

---

*报告自动生成*


## 八、训练数据集上的推理复测（数据切片 `[0:10000)`）

**设置**：轨迹预测器在 `[0:10000)` 上拟合（与训练流水线一致）；评测数据为 **同一切片** `[0:10000)`，即原训练集上的 **In-distribution** 推理。Hybrid SAC 加载当前 `checkpoints/sac_proactive.pth` / `sac_reactive.pth`。

### 八.1 Proactive 模式

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Score |
|-----------|------------|------------|---------------------|------------------|-------|
| SA | 331 | 359 | 5474 | 2.81 | 510.5 |
| DQN | 1445 | 1331 | 3627 | 0.00 | 2110.5 |
| Hybrid SAC | 3836 | 41 | 2885 | 0.81 | 3856.5 |

### 八.2 Reactive 模式

| Algorithm | Migrations | Violations | Avg Latency (ms) | Score |
|-----------|------------|------------|------------------|-------|
| SA | 318 | 103 | 1.94 | 369.5 |
| DQN | 715 | 911 | 0.00 | 1170.5 |
| Hybrid SAC | 1173 | 65 | 0.85 | 1205.5 |

### 八.3 Proactive 时延（Hybrid SAC vs SA）

- Hybrid SAC: **0.81 ms**；SA: **2.81 ms**；比值 SA/SAC ≈ **3.5x**

### 八.4 Hybrid SAC（Proactive）按 DAG 自适应迁移统计

口径与第六节一致（`inference_mode=True` 且 **PROACTIVE** 触发）。

| DAG Name | Proactive Decisions | Total Migrated Nodes | Avg Nodes per Decision |
|----------|---------------------|----------------------|------------------------|
| Diamond_DAG_2 | 53 | 196 | 3.70 |
| FanIn_Aggregator_1 | 685 | 1470 | 2.15 |
| FanIn_Aggregator_2 | 41 | 205 | 5.00 |
| FanIn_Aggregator_3 | 1149 | 700 | 0.61 |
| FanOut_Broadcaster_2 | 120 | 600 | 5.00 |
| FanOut_Broadcaster_3 | 244 | 64 | 0.26 |
| IoT_Lightweight_DAG | 593 | 261 | 0.44 |

### 八.5 三算法对照（Proactive，同一切片）

| Algorithm | Violations | Proactive Decisions | Migrations | Avg Latency (ms) |
|-----------|------------|---------------------|------------|------------------|
| SA | 359 | 5474 | 331 | 2.81 |
| DQN | 1331 | 3627 | 1445 | 0.00 |
| Hybrid SAC | 41 | 2885 | 3836 | 0.81 |

- **对照**：正文一至七节为测试段 `[10000:15000)`；本节为训练段 **同分布复测**，便于论文区分 **分布内** 与 **分布外（测试段）** 行为。


## 九、按 3000 行切片的推理对比（`[0,15000)`，SA vs Hybrid SAC）

**生成时间**：2026-04-28 12:15:39  
**协议**：轨迹预测器在 **`[0,15000)`** 上拟合 **一次**；共 **5** 组切片，每组 **3000** 行；每组内依次跑 **Proactive** / **Reactive** 的 SA 与 Hybrid SAC（**不含 DQN**）。
Hybrid SAC：`inference_mode=True`，权重 `checkpoints/sac_proactive.pth` / `sac_reactive.pth`。SA：`collect_dag_proactive_stats=True`（Proactive 路径，与主流程推理口径一致）。

| 数据切片 | 模式 | SA M | SA V | SA ProDec | SA Lat(ms) | SAC M | SAC V | SAC ProDec | SAC Lat(ms) |
|----------|------|------|------|-----------|------------|-------|-------|------------|-------------|
| `[0,3000)` | Proactive | 79 | 94 | 1627 | 2.57 | 905 | 14 | 417 | 0.87 |
| `[0,3000)` | Reactive | 82 | 144 | 0 | 1.64 | 337 | 19 | 0 | 0.79 |
| `[3000,6000)` | Proactive | 78 | 26 | 1869 | 2.62 | 367 | 14 | 1549 | 0.52 |
| `[3000,6000)` | Reactive | 98 | 24 | 0 | 1.76 | 320 | 24 | 0 | 0.77 |
| `[6000,9000)` | Proactive | 119 | 37 | 1359 | 2.55 | 591 | 11 | 817 | 0.56 |
| `[6000,9000)` | Reactive | 99 | 19 | 0 | 1.81 | 506 | 20 | 0 | 0.82 |
| `[9000,12000)` | Proactive | 72 | 298 | 1363 | 2.55 | 925 | 296 | 334 | 0.69 |
| `[9000,12000)` | Reactive | 65 | 397 | 0 | 1.65 | 193 | 298 | 0 | 0.64 |
| `[12000,15000)` | Proactive | 65 | 714 | 1083 | 2.51 | 1165 | 710 | 191 | 0.84 |
| `[12000,15000)` | Reactive | 53 | 830 | 0 | 1.60 | 418 | 710 | 0 | 0.80 |

**本附录总墙钟（5 组 × Proactive+Reactive）**：约 **68 s**。

*本段为一次性批处理追加；未修改 `run_comparison.py` 等主流程代码。*
