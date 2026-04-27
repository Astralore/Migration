# 微服务 DAG 更新后的算法对比实验（2026-04-27）

## 实验目的

在更新 `core/microservice_dags.py` 中的微服务 DAG 定义后，使用现有对比管线 `run_comparison.py`，与 **SA（模拟退火）**、**DQN**、**Hybrid SAC（GAT + 离散 SAC）** 在相同数据与配置下进行推理阶段对比，并将结果归档到本文件（不覆盖历史 `result.md` 的用途时，仍以本报告为准）。

## 运行配置

| 项目 | 值 |
|------|-----|
| 脚本 | `run_comparison.py` |
| 模式 | `INFERENCE_MODE = True`（推理） |
| 预测器拟合 | 出租车索引 `[0, 10000)` |
| 评测数据 | `[10000, 15000)`，5000 行 |
| 边缘服务器 | `data/edge_server_locations.csv`（600 台） |
| 预测 horizon | `FORECAST_HORIZON = 15` |

## SAC 权重 — 实验一：默认最新（非 v3.8）

按文件修改时间，`checkpoints` 下最新的一对权重为（二者时间戳一致）：

- **Proactive**：`checkpoints/sac_proactive.pth`（2026-04-27 10:29:25）
- **Reactive**：`checkpoints/sac_reactive.pth`（2026-04-27 10:29:25）

与 `run_comparison.py` 中默认路径一致，未再指向 `_old` 备份文件。

## DAG 与管线兼容性说明

在跑通全量对比前，对 **`Diamond_DAG_2`** 做了一处**拓扑修正**：

- 原边集中同时存在 `("MS_15284", "MS_7401")` 与 `("MS_7401", "MS_15284")`，在 **MS_15284 ↔ MS_7401** 上形成**有向环**，使得所有节点的入度均大于 0，`get_entry_nodes()` 返回空列表。
- 现有 SA / DQN / Hybrid SAC 均在首跳网关上依赖入口节点，会在 `entry_nodes[0]` 处触发 `IndexError`，实验无法完成。
- **处理**：删除回边 `("MS_7401", "MS_15284")`，保留 `MS_15284 → MS_7401 → MS_37027 → MS_28467` 与 `MS_15284 → MS_28467` 的 DAG 结构，入口节点为 `MS_15284`。

若业务上必须保留双向调用统计，需要在 `core/dag_utils.py` 或各算法中增加**环图 / 多入口**策略，而不是仅依赖入度为零的节点。

## 数值结果 — 实验一：默认 SAC 权重

### Proactive 模式（测试集推理）

| 排名 | Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Score* |
|:----:|-----------|------------:|------------:|---------------------:|-----------------:|--------:|
| 1 | Hybrid SAC | 0 | 1987 | 29 | 0.83 | 993.5 |
| 2 | SA | 549 | 1012 | 29 | 4.56 | 1055.0 |
| 3 | DQN | 2054 | 1236 | 28 | 0.00 | 2672.0 |

\* Score = `Migrations + 0.5 * Violations`（与 `run_comparison.py` 中报告一致）。

### Reactive 模式（测试集推理）

| 排名 | Algorithm | Migrations | Violations | Avg Latency (ms) | Score* |
|:----:|-----------|------------:|------------:|-----------------:|--------:|
| 1 | SA | 692 | 1011 | 1.53 | 1197.5 |
| 2 | Hybrid SAC | 2256 | 1004 | 0.78 | 2758.0 |
| 3 | DQN | 3136 | 1247 | 0.00 | 3759.5 |

### 时延（Proactive 下 SAC 相对 SA）

- Hybrid SAC 平均决策时延：**0.83 ms**
- SA 平均决策时延：**4.56 ms**
- 约 **5.5×** 相对 SA 更快（仅统计有决策记录的步）。

## 简要结论

1. **Reactive**：在当前权重与 DAG 下，**Hybrid SAC** 的违规数（1004）与 SA（1011）接近，但迁移次数明显更多，综合 Score 劣于 SA；**DQN** 违规与迁移均最高。
2. **Proactive**：**Hybrid SAC 出现策略退化**：迁移数为 **0**、违规数 **1987**，与 Reactive 同权下的行为不一致；自动化分析脚本亦提示 Proactive 相对 Reactive 违规大幅上升，需单独排查（触发逻辑、评估阶段 argmax、与训练分布是否一致等）。**SA** 与 **DQN** 在 Proactive 下仍给出非零迁移。
3. 本次运行同时生成了 `outputs/` 下的训练曲线与成本、违规对比图（脚本在推理模式下仍会对 DQN 等写入历史曲线文件，若需可一并查看）。

## 复现命令

在项目根目录执行：

```bash
python run_comparison.py
```

（保持 `INFERENCE_MODE = True`，并确保上述 SAC 权重路径存在。）

---

## 追加：SAC v3.8 权重（`v3.8_old`）

**权重文件**

- Proactive：`checkpoints/sac_proactive_v3.8_old.pth`
- Reactive：`checkpoints/sac_reactive_v3.8_old.pth`

**运行时间**：`result.md` 记录为 2026-04-27 11:36:04（推理区间仍为 `[10000:15000)`，其余配置与实验一相同）。

**说明**：`assign_dag_type()` 使用未固定种子，每次完整跑 `run_comparison.py` 时各出租车被分配的 DAG 类型会重新抽样，因此 **SA / DQN 的绝对指标不宜与上一节逐数字对比**；本节价值在于同一轮运行内 **三算法并列**，以及 **v3.8 SAC 与默认 SAC 在策略形态上的差异**（见下表与结论）。

### Proactive 模式（v3.8 SAC）

| 排名 | Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Score* |
|:----:|-----------|------------:|------------:|---------------------:|-----------------:|--------:|
| 1 | SA | 418 | 1009 | 29 | 4.28 | 922.5 |
| 2 | Hybrid SAC | 660 | 1008 | 29 | 2.14 | 1164.0 |
| 3 | DQN | 1528 | 1050 | 5 | 0.00 | 2053.0 |

### Reactive 模式（v3.8 SAC）

| 排名 | Algorithm | Migrations | Violations | Avg Latency (ms) | Score* |
|:----:|-----------|------------:|------------:|-----------------:|--------:|
| 1 | Hybrid SAC | 0 | 1987 | 0.76 | 993.5 |
| 2 | SA | 1093 | 1008 | 2.05 | 1597.0 |
| 3 | DQN | 2475 | 1071 | 0.00 | 3010.5 |

（排名与 `run_comparison.py` 一致，按 Score 升序。此处 Hybrid SAC 因 **0 迁移** 得到较低 Score，但 **违规数 1987 远高于 SA/DQN**，不能解读为 Reactive 下表现更好。）

### v3.8 与默认 SAC 行为对照（仅 Hybrid SAC）

| 模式 | 指标 | 默认权重（`sac_*.pth`） | v3.8（`*_v3.8_old.pth`） |
|------|------|-------------------------|--------------------------|
| Proactive | Migrations | 0 | 660 |
| Proactive | Violations | 1987 | 1008 |
| Reactive | Migrations | 2256 | 0 |
| Reactive | Violations | 1004 | 1987 |

可见：**当前默认权重**在 Reactive 上更可用、在 Proactive 上几乎不迁移；**v3.8 权重**在 Proactive 上明显更积极且违规与 SA 相当，但在 Reactive 上出现与前者镜像的「不迁移 + 高违规」现象。若要在论文或报告中对比两版权重，建议固定随机种子或固定「车 → DAG」映射后只替换 SAC 检查点再跑。

---

*本报告为针对 `microservice_dags` 更新后的对比实验归档；每次执行 `run_comparison.py` 会更新仓库中的 `result.md`（当前 `result.md` 内容为**最后一次**运行，即 v3.8 实验）。默认权重一节的表格仍以本文件第一节为准。*
