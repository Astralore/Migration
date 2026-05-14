# MARL 阶段四：结合 DAG 数据特性的物理语义修复

## 一、本轮实验与 DAG 数据的共同结论

最新中等规模实验目录：`experiments\medium_validation_20260513_203241_cov50_stage8`

本轮阶段三修复后，训练段已经明显缓解“全员迁移”：

- Proactive 训练：`all_agents_migrated_ratio = 2.08%`，`stay_action_ratio = 55.69%`
- Reactive 训练：`all_agents_migrated_ratio = 3.39%`，`stay_action_ratio = 46.36%`

但测试推理仍然存在泛化坍缩：

- Proactive 推理：`all_agents_migrated_ratio = 79.84%`
- Reactive 推理：`all_agents_migrated_ratio = 83.91%`
- Reactive 推理平均成本仍高达 `49769.77ms`

这说明问题不只是 lambda 或 dense reward，而是当前 `MICROSERVICE_DAGS` 的数据语义和算法环境之间存在不一致。

## 二、DAG 数据特性对当前算法的影响

### 1. `USER` / `UNKNOWN` / `UNAVAILABLE` 不是可迁移微服务

`core/microservice_dags.py` 中确实存在外部节点：

- `Data_Heavy_DAG`：`USER`、`UNKNOWN`
- `Compute_Heavy_DAG`：`UNKNOWN`
- `FanOut_Broadcaster_2`：`UNAVAILABLE`
- `Diamond_DAG_1`：`USER`、`UNKNOWN`

当前代码的问题是：

- `initialize_dag_assignment(...)` 会把所有 DAG 节点都部署到边缘服务器。
- `build_marl_graph_state(...)` 默认所有节点 `node_movable=True`。
- `marl_gat.py` 会对 `topological_sort(dag_info)` 的所有节点采样动作。
- `calculate_microservice_reward(...)` 会把所有发生 assignment 变化的节点都计入迁移成本。

因此当前算法确实可能“迁移 USER / UNKNOWN / UNAVAILABLE”。这不是策略问题，而是环境建模漏洞。

### 2. 许多 DAG 本身非常轻量，部分全迁移并非必然错误

例如：

- `FanOut_Broadcaster_2` 全部节点 `state_mb = 0`，大多是 `50MB` 镜像。
- `Compute_Heavy_DAG` 全部节点 `state_mb = 0`，大多是 `200MB` 镜像。

这类 DAG 中，如果只看 SLA 惩罚与单节点迁移成本，全迁移可能确实是短期最优。因此不能把所有全员迁移都视为算法错误。

真正的问题是：当前系统没有模拟“同一时刻多个节点迁往同一目标服务器时的带宽竞争”，导致集体迁移过于便宜。

### 3. 高状态节点的异构性存在，但梯度尺度仍可能不足

数据中已有明显异构：

- `FanIn_Aggregator_2.MS_27421`：`state_mb = 512`
- `FanOut_Broadcaster_3.MS_13448`：`state_mb = 512`
- `Diamond_DAG_1.UNKNOWN`：`state_mb = 512`，但它应被视为外部节点，不应迁移

当前 `core/marl_reward.py` 已按 `image_mb + state_mb` 计算 local migration cost，因此“没有异构迁移成本”这个判断不成立。

但当前 reward 是 log-scaled shared reward + normalized local penalty。仅靠继续把 `lambda_migration` 拉到 `1.0/1.5` 风险很大，之前小样本已出现过从“过度迁移”翻到“全 STAY”的现象。

## 三、对原修改方案的修正

### 原方案 1：Pinned Nodes，方向正确，但必须同时修 reward / entry 语义

只在 action mask 中把外部节点锁为 `STAY` 还不够，因为当前 reward 和 entry 计算仍会把它们当成可部署服务。

必须新增统一节点分类：

```python
EXTERNAL_NODE_NAMES = {"USER", "UNKNOWN", "UNAVAILABLE"}

def is_external_node(node_name):
    return node_name in EXTERNAL_NODE_NAMES

def get_deployable_nodes(dag_info):
    return [n for n in dag_info["nodes"] if not is_external_node(n)]
```

后续所有“迁移控制”和“迁移成本”只作用于 deployable nodes；GAT message passing 仍保留 external nodes。

### 原方案 2：并发带宽竞争，方向正确，但目标文件不是只改 `core/reward.py`

当前 GAT-MARL 训练使用：

- shared reward：`core/reward.py::calculate_microservice_reward(...)`
- local reward：`core/marl_reward.py::calculate_marl_rewards(...)`

因此并发惩罚必须至少进入 `core/marl_reward.py` 的 `_local_migration_costs(...)`，否则 Actor 学不到“同一目标服务器拥塞”的局部代价。

为了评估指标一致，最好也同步修改 `core/reward.py` 中的 total migration cost；否则训练成本与报告成本会不一致。

### 原方案 3：lambda 提到 `1.0/1.5`，不建议作为默认

当前最新实验中 lambda 已经正确生效：

- Proactive：`lambda_migration = 0.15`，`lambda_split = 0.05`
- Reactive：`lambda_migration = 0.20`，`lambda_split = 0.08`

Reactive 推理仍过度迁移，说明确实需要更强约束；但直接改到 `1.0/1.5` 会极大增加全 STAY 风险。

正确做法是先修物理建模漏洞，再做小步 lambda 消融，例如：

- Reactive：`0.20 / 0.08` → `0.25 / 0.10` → `0.30 / 0.12`
- Proactive：先保持 `0.15 / 0.05`

## 四、必须修改的代码方案

### 任务 1：新增 DAG 节点语义工具

目标文件：建议新增 `core/microservice_node_types.py`，或放入 `core/dag_utils.py`。

需要提供：

```python
EXTERNAL_NODE_NAMES = {"USER", "UNKNOWN", "UNAVAILABLE"}

def is_external_node(node_name):
    return node_name in EXTERNAL_NODE_NAMES

def get_deployable_nodes(dag_info):
    return [n for n in dag_info["nodes"] if not is_external_node(n)]

def get_service_entry_nodes(dag_info):
    # 返回真正的可部署入口服务：
    # 1. deployable 子图中没有 deployable 入边的节点
    # 2. 或直接由 USER / UNKNOWN / UNAVAILABLE 指向的 deployable 节点
```

`get_service_entry_nodes(...)` 很关键，因为原 `get_entry_nodes(...)` 会把 `USER/UNKNOWN` 当入口，从而污染 SLA 距离和 proactive trigger。

### 任务 2：Pinned Nodes 只参与图编码，不参与动作控制

目标文件：`core/marl_state_builder.py`、`algorithms/marl_gat.py`

修改要求：

- `build_marl_graph_state(...)` 中 external nodes 的 action mask 强制为 `[STAY=True, 其他=False]`。
- node feature 增加 `is_external` 一维，因此 `GraphEncoder(node_feat_dim=14, ...)`。
- `marl_gat.py` 执行动作时，external nodes 即使 Actor 输出非 STAY，也必须强制保留原 assignment。
- 统计指标新增：
  - `controlled_agents_per_decision`
  - `pinned_agents_per_decision`
  - `controlled_migrations`
  - `controlled_all_agents_migrated_ratio`

注意：原 `avg_agents_per_decision` 包含 external nodes 后会失真，后续判断过度迁移应看 controlled 指标。

### 任务 3：reward / trigger 只对可部署入口服务计算 SLA

目标文件：`core/reward.py`、`core/marl_reward.py`，可能还包括 proactive trigger 前的 gateway 选择逻辑。

修改要求：

- `_entry_access_profile(...)` 不应再用原始 `get_entry_nodes(...)`，应改用 `get_service_entry_nodes(...)`。
- `estimate_dag_migration_time_s(...)` 默认只估计 deployable nodes。
- `calculate_microservice_reward(...)` 的 migration cost 只统计 deployable nodes。
- `calculate_marl_rewards(...)` 的 local migration / split penalty 只对 deployable nodes 产生有效惩罚，external nodes 的 agent reward 只作为图上下文，不应驱动动作学习。

如果某个 DAG 的 service entry 为空，回退到 deployable nodes 中拓扑最靠前的节点，不能回退到 `USER/UNKNOWN`。

### 任务 4：加入并发带宽竞争惩罚

目标文件：`core/marl_reward.py`，建议同步 `core/reward.py`。

对当前 step 中迁移到同一 target server 的 deployable nodes 计数：

```python
target_counts[target_server] += 1
```

迁移成本改为：

```python
effective_node_bandwidth = effective_bandwidth / target_counts[target_server]
cost_ms = ((image_mb + state_mb) * MB_TO_MBIT / effective_node_bandwidth) * 1000 + BASE_MIGRATION_OVERHEAD_MS
```

这比单纯增大 lambda 更合理，因为它只惩罚“同一时刻挤到同一个目标服务器”的集体迁移，而不会无差别压制所有迁移。

### 任务 5：按 DAG 类型输出诊断，不要只看 overall

当前 `cost_by_dag_complexity` 只有 simple / medium，不足以判断问题是否集中在某些 DAG。

需要新增：

- `cost_by_dag_type`
- `migrations_by_dag_type`
- `all_agents_migrated_ratio_by_dag_type`
- `controlled_migrations_by_dag_type`

重点观察：

- `FanOut_Broadcaster_2`
- `Compute_Heavy_DAG`
- `Data_Heavy_DAG`
- `Diamond_DAG_1`

如果全迁移主要集中在无状态轻量 DAG，不应过度惩罚；如果集中在含 512MB stateful 节点 DAG，则说明 local migration cost 仍不足。

## 五、验证顺序

1. 先跑单 DAG smoke：
   - `FanOut_Broadcaster_2`
   - `Compute_Heavy_DAG`
   - `Data_Heavy_DAG`
   - `Diamond_DAG_1`
2. 检查 external nodes：
   - action mask 只能 STAY
   - assignment 不发生变化
   - migration cost 不包含 external nodes
3. 检查并发惩罚：
   - 同一目标服务器迁移 1 个、2 个、5 个节点时，单节点迁移成本应递增
4. 再跑小样本 Proactive / Reactive smoke。
5. 最后再跑中等规模验证，并删除旧 `marl_gat_*.pth`。

## 六、禁止事项

- 不要把 `USER/UNKNOWN/UNAVAILABLE` 从图中删除；它们仍然是拓扑上下文。
- 不要让 external nodes 参与迁移动作、迁移成本或 deployable entry SLA。
- 不要只通过提高 lambda 解决过度迁移；必须先修正节点语义和并发带宽。
- 不要把 `lambda_migration` 默认直接调到 `1.0/1.5`。
- 不要再用包含 external nodes 的 `avg_agents_per_decision` 判断是否全员迁移，必须使用 controlled 指标。