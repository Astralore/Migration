# 第一阶段（基础设施与防泄漏）— 改动摘要与测试结果

**文档用途**：覆盖记录本次在允许范围内对 `algorithms/hybrid_sac.py`、`algorithms/dqn.py`、`run_comparison.py` 的修改要点及本地验证结果。

---

## 一、核心改动一览

| 编号 | 目标 | 要点 |
|------|------|------|
| A1 | Hybrid SAC 评测无状态泄漏 | 最后一轮 eval 开始前清空 `taxi_dag_assignments`、`taxi_dag_type`，每 epoch 仍重置 `taxi_last`，保证从干净 nearest 起步 |
| A2 | DQN 推理与权重 | 增加 `inference_mode`、`checkpoint_path`、`save_checkpoint_path`；推理时加载权重、`epsilon=0`、跳过 `optimize_model` 与目标网络同步 |
| A3 | DQN 决策时延 | `time.perf_counter()` 包住单次触发内的拓扑序决策循环，返回 `total_decision_time`、`avg_decision_time_ms` |
| A4/A5 | SLA 与报告主指标 | Violations 与 `core.context.check_sla_violation` 对齐（空间 **或** QoS，非仅距离）；`result.md` 报告表主列由 Score 改为 **Avg Total Cost (ms)**（按 `decision_count` 均摊接入+通信+迁移总 ms） |

**未改文件**（按约束）：`core/reward.py`、网络结构、非白名单文件（如 `evaluation/metrics.py` 控制台表头仍为 Score）。

---

## 二、关键代码位置

### 1. `algorithms/hybrid_sac.py`

- **状态清空（A1）**：`for epoch in range(num_epochs):` 内，当 `is_eval_epoch` 时 `taxi_dag_assignments.clear()`、`taxi_dag_type.clear()`，再 `taxi_last = {}`。
- **违规统计（A4）**：将 `gateway_dist > 15.0` 改为 `check_sla_violation(current_lat, current_lon, gw_lat, gw_lon)`（主循环与 `evaluate_sac_policy` 两处）。
- **新增导入**：`from core.context import check_sla_violation`（与原有 `get_trigger_type` 等并列）。

### 2. `algorithms/dqn.py`

- **检查点**：`_save_dqn_checkpoint` / `_load_dqn_checkpoint` 保存/加载 `q_network`、`target_network` 的 `state_dict`。
- **推理分支**：`inference_mode` 为真时 `checkpoint_path` 必填、加载后 `epsilon = 0.0`；主循环内 `if not inference_mode:` 再执行 `optimize_model` 与 `target_network` 同步。
- **计时（A3）**：在 `decision_count += 1` 后、节点循环前 `t_decision_start = time.perf_counter()`，循环后累加 `total_decision_time`；返回中增加 `total_decision_time`、`decision_count_for_latency`（与 `decision_count` 一致）、`avg_decision_time_ms`。
- **违规（A4）**：`check_sla_violation` 替代纯距离判断。
- **训练结束保存**：非推理且提供 `save_checkpoint_path` 时写盘。

### 3. `run_comparison.py`

- **常量**：`DQN_CHECKPOINT_PROACTIVE`、`DQN_CHECKPOINT_REACTIVE`（与 SAC 的 `checkpoints/` 路径并列）。
- **训练段**：`run_dqn_microservice_fair(..., save_checkpoint_path=...)` 分别写 proactive / reactive 权重。
- **推理段**：`inference_mode=True` 且 `checkpoint_path=...`。
- **报告**：`_avg_total_cost_ms(res) = (total_access_latency + total_communication_cost + total_migration_cost) / decision_count`；表头与行内 **Score** 替换为 **Avg Total Cost (ms)**；流水线说明更新为测试段同时加载 **Hybrid SAC 与 DQN** 的 checkpoint。

---

## 三、测试项与结果

| 测试 | 命令 / 方法 | 结果 |
|------|-------------|------|
| 语法与导入 | `python -m py_compile algorithms/dqn.py algorithms/hybrid_sac.py run_comparison.py` | 通过（exit 0） |
| 成本辅助函数 | `_avg_total_cost_ms({'decision_count':2, 'total_access_latency':10, 'total_communication_cost':3, 'total_migration_cost':1})` | 得到 `7.0`（(10+3+1)/2） |
| DQN 权重 round-trip | 临时创建 `MicroserviceDQN` 对，`_save_dqn_checkpoint` 后 `_load_dqn_checkpoint` | 无异常，终端打印 `[DQN SAVE]` / `[DQN LOAD]` |

**未在本文档生成时重跑全量** `run_comparison.py` / `--pipeline`；全量实验需数据与较长墙钟时间。推理前需已存在 `checkpoints/dqn_*.pth`（由训练段写出），否则 `inference_mode` 下 DQN 会按设计在缺权重时报错。

---

## 四、使用注意

1. **DQN 推理依赖磁盘权重**：首次仅开推理模式前请先完成训练阶段或手动放置兼容结构的 `.pth`。
2. **控制台排行榜**：若仍显示 Score，来自 `evaluation/metrics.py`（本次未改）；以本仓库生成的 **result.md / 流水线报告** 中的 **Avg Total Cost (ms)** 为准。

---

*本文件由助手根据第一阶段实现覆盖写入，可与 `command.md` 后续阶段对照迭代。*

## 第二阶段（心智模型重构）— 改动与自动化测试结果

### 1. 改动要点（仅 `algorithms/hybrid_sac.py`）

| 项 | 内容 |
|----|------|
| **B2/B3 自回归观测** | 在 `for ms_node in sorted_nodes:` 内，每步用当前 `graph_state` 做 GAT 前向与 Actor 决策，更新 `taxi_dag_assignments` 后立刻 `build_graph_state` 得到下一节点可见的图状态；训练、eval/推理与 `evaluate_sac_policy` 均一致，消除「整 DAG 共用首轮 embedding」的幻觉。 |
| **B5 稀疏奖励** | 废除 `reward / num_nodes`。写入 Replay Buffer 时：`is_last=False` → `reward_i=0.0`；仅拓扑末节点 `is_last=True` → 写入完整标量 `reward`。 |
| **B6 Gamma 掩码** | 在 `optimize_sac` 中：`effective_gamma = gamma if done else 1.0`。`done=False`（DAG 内部步）时 TD 目标为 `r + 1.0 * V(s')`（配合稀疏奖励时通常为 `0 + V(s')`）；`done=True`（末节点）时为纯蒙特卡罗尾部回报 `r`，不对 DAG 内下一步打折。 |

测试期间曾为缩短首轮 `optimize_sac` 触发时间，在**已删除**的临时脚本中设置环境变量 `HYBRID_SAC_BATCH_OVERRIDE=8`（正式仓库代码中**不包含**该覆盖，默认仍为 `batch_size=32`）。`run_comparison.py` 中的 `ACTIVE_USERS_LIMIT`、`num_epochs` 已在验证后恢复为 **100** 与 **6**。

### 2. 代表性调试日志（摘自一次性 Hybrid-only 烟测；埋点已移除）

以下为采集时的原始行（证明稀疏奖励与非 1 的 γ 掩码在优化器中生效）：

```
[MDP buffer] node_idx=4 action=1 reward=0.000000 is_last=False
[MDP buffer] node_idx=0 action=1 reward=0.000000 is_last=False
[MDP buffer] node_idx=1 action=1 reward=0.000000 is_last=False
[MDP buffer] node_idx=2 action=1 reward=0.000000 is_last=False
[MDP buffer] node_idx=3 action=1 reward=-2.036702 is_last=True
[MDP optimize_sac] done=True sparse_terminal reward=-1459.330673 (no bootstrap)
[MDP optimize_sac] done=False effective_gamma=1.0 reward=0.000000
```

解读简述：

- **Buffer**：前序节点 `reward=0`、`is_last=False`；仅末节点出现非零 `reward` 且 `is_last=True`，符合稀疏回报定义。
- **optimize_sac**：对 `done=False` 的样本打印 `effective_gamma=1.0`；对 `done=True` 的样本走稀疏终止分支且无 bootstrap，与 B6 一致。

### 3. 与 `python run_comparison.py` 的关系

全量对比需先完成 SA、DQN 再进入 Hybrid SAC，墙钟较长。第二阶段除上述专用烟测外，亦曾启动全量 `python run_comparison.py`（小批次调试环境下），其终端中同样出现仅含 **`[MDP buffer]`** 的稀疏奖励行（在共享调试计数耗尽前尚未轮到 `optimize_sac` 打印）。最终交付代码已去掉全部 MDP `print` 与 `batch_size` 环境覆盖。

---

*第二阶段文档追加完毕。*

---

## 第三阶段（物理法则重置）— 改动与测试结果

### 1. 改动要点（红线：`core/reward.py`、`core/context.py`、`algorithms/hybrid_sac.py`）

| 编号 | 内容 |
|------|------|
| **C1** | `migration_delay_ms`：传输体积（MB）乘以 `MB_TO_MBIT = 8` 转为比特；在传输时间（ms）基础上叠加每条迁移 `BASE_MIGRATION_OVERHEAD_MS = 200`（容器启动等固定开销）。Reactive 路径仍保留既有倍数系数。 |
| **C2** | 最终回报：`reward = max(-total_cost_ms / 1000.0, -10.0)`；`details["total_cost_ms"]` 仍为**未缩放**的物理总耗时（ms）。 |
| **C2.1** | SAC 温度初值 `alpha_init` 从 `0.05` 调整为 `0.001`（与回报缩小 1000 倍同量级），`target_entropy` 未改。 |
| **C3** | Proactive 条件：以 TTV（到阈值的预测时间，秒）与 `estimated_migration_time_s + 1.0` 比较；`get_trigger_type` 签名未变，无法精算时内部保守兜底 `estimated_migration_time_s = 2.0`。 |
| **C3.2** | 仅在 `run_hybrid_sac_microservice` 与 `evaluate_sac_policy` 主循环中维护 `migration_lock: taxi_id -> unlock_step`；Proactive 触发后 `unlock_step = sim_step + 2`，锁定期内通过 `proactive_enabled=proactive_gate` 禁止新的 Proactive 触发，Reactive 不受影响。 |

### 2. 自动化烟测方法（不修改 `run_comparison.py`；强制走原始清洗管道）

在 PowerShell 中设置 `PYTHONPATH`、小步数调试与数据截断，直接调用 `load_data(..., processed_csv=False, active_users_limit=2, chunk_size=500)` 与 `run_hybrid_sac_microservice(..., proactive=True, num_epochs=1)`。说明：`run_hybrid_sac_microservice` 内部 `use_proactive = proactive and predictor is not None`，本次烟测**未**传入 `predictor`，因此日志中 `Proactive: False`，触发统计以 **REACTIVE** 为主；用于验证 **C2/C2.1（reward 缩放与 α=0.001）** 已足够。若要连带验证 **C3/C3.2** 的 Proactive 路径，需在同一调用中构造并传入与训练一致的 `predictor`。验证时曾临时通过环境变量 `PHASE3_VERIFY=1` 打印 `alpha_init` 与首条缩放后 `reward`；**正式代码中该埋点已删除**。

### 3. 证明性日志摘录（埋点已移除；以下为当时终端输出）

```
[PHASE3 verify] SAC alpha_init=0.001
[PHASE3 verify] scaled_reward=-5.002055 total_cost_ms=5002.06
```

同时段 `HYBRID_SAC_DEBUG` 行显示 `reward(clipped)` 落在 `[-10, 0]`，例如 `reward(clipped)=-5.002055089837122`、`reward(clipped)=-10.0`（与 `max(..., -10)` 一致）。

### 4. 结论

- **Alpha**：SAC 以 `0.001` 初始化，与缩放后的回报尺度匹配。  
- **Reward**：缩放后单步回报在 `[-10, 0]` 内，`details` 中仍保留真实 `total_cost_ms`。  
- **锁步**：主训练/评估循环与 `evaluate_sac_policy` 均包含 Proactive 双步锁定逻辑，避免预测抖动。

---

*第三阶段文档追加完毕。*

---

## 第四阶段（基线校准）— 改动与测试结果

### 1. 改动要点（仅 `algorithms/sa.py`）

| 项 | 内容 |
|----|------|
| **温度与迭代** | `microservice_simulated_annealing` 默认：`temp=3000.0`、`cooling_rate=0.97`、`max_iter=50`。 |
| **埋点计数** | `sa_neighbor_count`：每评估一个邻域候选（生成并成功比对代价）+1；`sa_accept_count`：每次接受当前解 +1；`sa_worse_accept_count`：接受且 `delta > 0`（Metropolis 上浮接受）+1。 |
| **返回指标** | 返回值改为 **仍可二元解包** 的 `MicroserviceSAReturn`（与 `(best_assignments, best_cost)` 兼容）；统计字典挂在 **`result.sa_stats`**，内含 `sa_accept_rate`、`sa_worse_accept_rate`，以及上述计数。接受率定义为 **`accept_count / neighbor_count`**，劣解接受率定义为 **`worse_accept_count / neighbor_count`**；**当 `sa_neighbor_count == 0` 时两者均为 `0.0`**，避免除零。 |

### 2. 烟测方法

在仓库根目录、已设置 `PYTHONPATH` 的前提下，用 `python -c` 对 `MICROSERVICE_DAGS['IoT_Lightweight_DAG']` 与三候选边服务器调用一次 `microservice_simulated_annealing`；另用 `max_iter=0` 验证 **无邻域评估** 时 `sa_stats` 全零且 **无** `ZeroDivisionError`。

### 3. 证明性日志摘录

**正常一次 SA（默认 `max_iter=50`）：**

```
unpack_ok True
sa_stats {'sa_accept_count': 50, 'sa_worse_accept_count': 17, 'sa_neighbor_count': 50, 'sa_accept_rate': 1.0, 'sa_worse_accept_rate': 0.34}
OK
```

（终端中出现的 `[HYBRID_SAC_DBG reward]` 行来自 `calculate_microservice_reward` 的全局调试开关，与 SA 本次改动无关。）

**边界：`max_iter=0`（分母为 0 防护）：**

```
{'sa_accept_count': 0, 'sa_worse_accept_count': 0, 'sa_neighbor_count': 0, 'sa_accept_rate': 0.0, 'sa_worse_accept_rate': 0.0}
```

### 4. 结论

- `microservice_simulated_annealing` 在新默认参数下可运行，且 **`sa_stats` 中含 `sa_accept_rate`**。  
- **`neighbor_count == 0`** 时比率返回 **`0.0`**，无除零异常。  
- 既有 **`best_assignments, best_cost = microservice_simulated_annealing(...)`** 调用无需改签名即可继续使用。

---

*第四阶段文档追加完毕。*
