# 算法重构意见与修改方案（Action Plan）

**版本**：2026-05-27  
**依据文档**：

- [算法设计全面总结_DAG调用关系奖励与指标.md](docs/算法设计全面总结_DAG调用关系奖励与指标.md)（下称 **《全面总结》**）
- [Reward_v2奖励函数设计总结.md](docs/Reward_v2奖励函数设计总结.md)（下称 **《Reward v2》**）

**代码真源**：`core/reward.py`、`core/marl_reward.py`、`core/dag_utils.py`、`algorithms/marl_gat.py`

---

## 0. 与两份设计文档的关系

| 主题 | 《全面总结》已说明 | 当前缺口（本方案要补） |
|------|-------------------|------------------------|
| DAG / entry / max-entry SLA | §3–§4、§6 | 边代价用 `norm_traffic` 归一化，**拓扑撕裂在 ms 量纲上几乎不可见** |
| `comm` vs `tearing` | §4.4、§7.1 | `comm` 对跨机边 **求和** 而非关键路径；与真实并发语义不符 |
| Reward v2 的 `J` | 《Reward v2》§5–§8 | `P_SLA` 仅看 **空口** `L_acc`（`d_max`），**不含** DAG 内 RPC 阻塞 |
| 训练 vs 报表 | §7、`total_cost_ms` | SA 优化全物理成本；GAT 的 `J` 与「拆图痛不痛」未对齐 → **策略可投机** |
| 实验解读 | §10 | `total_violations`（离散）≠ `sla_penalty_ms`（连续）；重构后需新增 `L_internal` 诊断字段 |

**结论**：原 Action Plan 的三步方向与两份文档**不矛盾**，而是把《全面总结》§4.4 的 comm 模型和《Reward v2》的 `E_q` **补全为「拓扑感知」**；但实现上必须**分阶段、防双重计费、校准量级**，不能一次性硬上「百万级惩罚」。

---

## 1. 问题诊断（对照代码与实验）

### 1.1 归一化陷阱（已证实）

`core/reward.py` 当前逻辑：

```text
norm_traffic = traffic / max_traffic
comm_delay_ms = Σ_{跨机边} norm_traffic × (edge_dist_km / v + L0)
```

《全面总结》§4.4 与代码一致。后果：

- `FanOut` 重边 `traffic=25609` → `norm≈0.99`，即使用户距 50 km，单条边 comm 仍约 **2 ms 量级**；
- **tearing**（按 `traffic × RPC_SIZE_MB`）与 **comm**（按归一化权重）**量纲割裂**，拆重型 DAG 的「时间痛」远小于「字节痛」。

与 GAT 特征一致：`build_marl_graph_state` 邻接权重亦为 `traffic/max_traffic`，**眼睛看见权重、神经几乎感觉不到跨机 RPC 时延**。

### 1.2 求和 vs 关键路径（已证实）

跨机边 `comm_delay_ms` 为 **sum**；《全面总结》未声称使用 critical path。对 `FanOut_Broadcaster` 等并发扇出，真实端到端更接近 **max(分支)**，求和会低估并发、高估链式（取决于拓扑）。

### 1.3 Reward v2 与拓扑脱节（已证实）

《Reward v2》：

```text
E_d 来自 d_max；E_q 来自 L_acc = f(d_max) 的超额
J = P_SLA + C_mig^nl；P_SLA 不含 comm / L_internal
```

实验佐证（`medium_validation_20260526_reward_v2_scale10000_v1`，2 epoch）：

| 阶段 | GAT 现象 |
|------|----------|
| Pro 推理 | P95≈5.9 km，总成本≈19k（较 S=5000 明显改善） |
| Re 推理 | P95≈31 km，迁移 16 次但仍差；**训练 Re 好、推理 Re 差** |
| 训练 | `stay_action_ratio` 仍 ≈99.9% |

说明：**仅调 S / future_gain 不够**；Reactive 与拓扑代价脱节可能是主因之一。

### 1.4 原方案文稿需修正处

| 项 | 原稿 | 正确口径（《全面总结》§5.2） |
|----|------|------------------------------|
| `USER_SLA_TOLERANCE_MS` | 误写「15 ms」 | `calc_access_latency_ms(15km)×0.99` ≈ **2.05 ms**（15 km 对应空口约 2 ms + 路由 2 ms 模型） |
| 惩罚爆炸示例 | 576 ms → 「千万级负分」 | 若 `E_q_km_eq = E_q × 200`，则 `β·E_q_km_eq²` 极大，**易导致梯度爆炸或反向坍缩（全不迁）** → 必须 **clip / 分通道 β / 或 ms 域惩罚** |
| `tearing` | 不混入时延 | **同意**；保留在 `total_cost_ms`，不进入 `L_e2e` |

---

## 2. 对三步 Action Plan 的评审意见

### 第一步：废除 `norm_traffic`，引入 `actual_rpc_calls`

| 维度 | 意见 |
|------|------|
| **必要性** | **强烈同意**。与《全面总结》§4.4 设计意图（traffic 表调用强度）一致，当前归一化破坏量纲。 |
| **`SCALING_FACTOR=0.01`** | 方向对，但属 **超参**，需用 cov50 上 SA/Nearest 的 `total_cost_ms` 分布标定；建议可配置 `EDGE_RPC_SCALING`（默认 0.01）。 |
| **`MIN_RPC_CALLS=1`** | 同意，避免零权边在 GAT 上「断开」。 |
| **联动** | `marl_reward._edge_split_cost`、counterfactual 中凡用 `norm_traffic` 处 **同步改**，否则训练/报表分裂。 |

### 第二步：关键路径 `L_internal`，隔离 tearing

| 维度 | 意见 |
|------|------|
| **必要性** | **同意**，但实现为 **v2.1 物理层** 而非直接替换所有 comm 语义。 |
| **算法** | 在 `topological_sort` 边上做 `node_latency[dst] = max(..., node_latency[src] + effective_edge_ms)`；多入口时对每个 entry 源点初始化 0。 |
| **报表** | 建议新增 `details.internal_critical_path_ms`；`comm_delay_ms` 可改为 **等于 `L_internal`**（或 deprecated 字段保留对比一期）。 |
| **注意** | **不要** 在保留 `sum(comm)` 的同时又把 `L_internal` 加进 `total_cost_ms`，否则 **双重计费**。 |

### 第三步：`L_internal` 进入 Reward v2 的 `E_q`

| 维度 | 意见 |
|------|------|
| **必要性** | **同意**，这是让 GAT「边特征 ↔ 惩罚」对齐的核心。 |
| **公式** | 建议采用《Reward v2》符号，明确定义：<br>`L_e2e = L_acc + L_internal`<br>`E_q_ms = max(0, L_e2e - USER_SLA_TOLERANCE_MS)`<br>`E_q = E_q_ms × FIBER_SPEED_KM_MS`（与现 `_sla_excess_km` 一致） |
| **风险** | `L_internal` 上百 ms 时 `β·E_q²` 极大；需 **(a)** 单独 `REWARD_V2_INTERNAL_BETA` 或 **(b)** `E_q_internal` 用 `log1p` / **(c)** `P_SLA` clip；并 **重标定 S**（可能 1e4–1e6 量级试扫）。 |
| **v1** | 若仍跑 v1 基线，应对 `E_eq` 或 comm 做 **同一物理层**，否则 v1/v2 对比不公平。 |

### 与《Reward v2》实验线的关系

| 正在进行 | 建议 |
|----------|------|
| S=10000 快筛、8 epoch softguard | **先完成当前 v2 线**（作为 **v2.0 基线**），再开 **v2.1（拓扑感知）** 分支 |
| Phase B 对齐 / Phase C | v2.1 稳定后再做公平对照；否则分不清是「公式」还是「comm 模型」带来的变化 |

---

## 3. 修订后的分阶段修改方案（推荐执行顺序）

### 阶段 D0：物理层 comm 重构（**不改** `J` / `REWARD_SCHEME`）— **已实现 2026-05-27**

**目标**：只改 `total_cost_ms` 的 comm 语义，SA/报表先对齐真实拓扑时延。

| 任务 | 文件 | 状态 |
|------|------|------|
| D0.1 | `core/reward.py` | ✅ `edge_actual_rpc_calls` / `edge_effective_latency_ms` / `compute_internal_critical_path_ms` / `compute_tearing_delay_ms` |
| D0.2 | `core/reward.py` | ✅ `comm_delay_ms = L_internal`；已移除 `norm_traffic` 求和 |
| D0.3 | `core/marl_reward.py` | ✅ `_edge_split_cost` 共用 `edge_effective_latency_ms` |
| D0.4 | `run_medium_validation_cov50.py` + 四算法 | ✅ 汇总字段 `total_internal_path_ms`；`details.internal_critical_path_ms` |
| D0.5 | 冒烟 | `python scripts/verify_d0_critical_path.py` |

**环境变量**：`EDGE_RPC_SCALING`（默认 0.01）、`MIN_RPC_CALLS`（默认 1.0）。

**验收**：`Data_Heavy` / `FanOut` 拆机后 `comm`（现为 `L_internal`）显著高于 2 ms 量级；SA 总成本排序合理。

---

### 阶段 D1：Reward v2.1（`L_internal` 进入训练 `P_SLA`）— **已实现 2026-05-27**

**目标**：训练 `J` 感知拓扑撕裂；与 D0 共用 `compute_internal_critical_path_ms`。

| 任务 | 文件 | 状态 |
|------|------|------|
| D1.1–D1.3 | `core/reward.py` | ✅ `use_reward_v2_internal_path()`；`L_e2e` 进入 **训练** `E_q`；`reward_objective_ms` 用 `sla_penalty_objective_ms` |
| D1.4 | `algorithms/marl_gat.py` | ✅ CF 重算动作前后 `L_internal`；`_edge_split_delta_ms` 对齐 D0 |
| D1.5 | 脚本 | ✅ `run_reward_v21_medium_test_cov50.py`、`run_reward_v21_softguard_train_cov50.py` |
| D1.6 | 冒烟 | `python scripts/verify_d1_internal_sla.py` |

**环境变量**：

```text
REWARD_V2_USE_INTERNAL_PATH=1   # 默认开启；设 0 复现 v2.0 训练目标
REWARD_SCHEME=v2
```

**计费约定（避免重复）**：

- `total_cost_ms`（SA/报表）：`L_acc + L_internal(comm) + … + sla_penalty`（**E_q 仍仅空口**）
- `reward_objective_ms`（RL）：`P_SLA(E_d, E_q from L_e2e) + C_mig^nl`

**验收**：见 v2.1 中等规模实验；拆机后 `sla_penalty_objective_ms` ≫ `sla_penalty_ms`。

---

### 阶段 D1.5：防爆减震（Clip + Scale）— **已实现 2026-05-27**

| 机制 | 实现 |
|------|------|
| `MAX_SLA_PENALTY_MS` | 默认 `500000`，`calculate_sla_penalty_ms` 统一 `min(P, cap)` |
| `REWARD_V2_OBJECTIVE_SCALE_MS` | 代码默认 `50000`；`run_reward_v21_*` 默认 **`100000`** |
| 目标 | `r = -J/S` 落在约 `[-5, 0]`，避免 v2.1 拆机时梯度爆炸 |

环境变量：`MAX_SLA_PENALTY_MS`、`REWARD_V2_OBJECTIVE_SCALE_MS`。

---

### 阶段 D2：GAT / DQN 特征对齐 — **已实现 2026-05-27**

| 任务 | 文件 | 状态 |
|------|------|------|
| D2.1 | `core/marl_state_builder.py` | ✅ 边权 / 节点 traffic：`log1p(actual_rpc) / max_log_rpc` |
| D2.2 | `core/state_builder.py` | ✅ DQN 16 维与 `build_graph_state` 邻接矩阵同上 |
| D2.3 | `core/reward.py` | ✅ `traffic_log_rpc_feature()`、`dag_max_traffic_log_rpc()` 公共 API |
| D2.4 | 实验 | v2.1 + D1.5 跑 `run_reward_v21_medium_test_cov50.py` |

---

## 4. 核心代码变更清单（实现时对照）

### 4.1 新增常量（`core/reward.py`）

```python
EDGE_RPC_SCALING = float(os.environ.get("EDGE_RPC_SCALING", "0.01"))
MIN_RPC_CALLS = float(os.environ.get("MIN_RPC_CALLS", "1.0"))
REWARD_V2_USE_INTERNAL_PATH = os.environ.get("REWARD_V2_USE_INTERNAL_PATH", "0") in ("1", "true")
```

### 4.2 边时延（替代 norm_traffic）

```python
def _edge_actual_rpc_calls(traffic: float) -> float:
    return max(MIN_RPC_CALLS, float(traffic) * EDGE_RPC_SCALING)

def _edge_effective_latency_ms(edge_dist_km, traffic, cross_machine: bool) -> float:
    if not cross_machine:
        return 0.0
    base_ms = (max(0.0, edge_dist_km) / FIBER_SPEED_KM_MS) + BASE_ROUTER_DELAY_MS
    return base_ms * _edge_actual_rpc_calls(traffic)
```

### 4.3 关键路径（示意）

```python
def _compute_internal_critical_path_ms(dag_info, assignments, servers_info) -> float:
    # topo 序遍历 edges；node_latency[dst] = max(..., src_lat + eff_ms)
    # return max(node_latency[deployable]) or 0.0
```

### 4.4 v2.1 SLA（示意）

```python
L_acc = calc_access_latency_ms(max_entry_dist_km)
L_internal = _compute_internal_critical_path_ms(...)
if is_reward_v2() and REWARD_V2_USE_INTERNAL_PATH:
    L_e2e = L_acc + L_internal
    E_q_ms = max(0.0, L_e2e - USER_SLA_TOLERANCE_MS)
    E_q = E_q_ms * FIBER_SPEED_KM_MS
else:
    # 现有 _sla_excess_km
```

### 4.5 勿双重计入

```text
total_cost_ms = L_acc + migration + tearing + L_internal + future + P_SLA
# 禁止再加 旧版 sum(comm) 与 L_internal 同时存在
```

---

## 5. 实验与脚本规划

| 阶段 | 脚本（新建或沿用） | Stamp 建议 |
|------|-------------------|------------|
| D0 验证 | `run_medium_validation_cov50.py` + env 仅物理层 | `20260527_comm_critical_path_d0_v1` |
| v2.0 基线（已完成/进行中） | `run_reward_v2_medium_test_cov50.py` | `reward_v2_scale10000_v1` |
| v2.0 全量 | `run_reward_v2_softguard_train_cov50.py` | `reward_v2_softguard_scale10000_v1` |
| v2.1 | `run_reward_v21_medium_test_cov50.py`（待建） | `reward_v21_internal_cp_v1` |
| 公平公式 | `run_reward_v2_phaseB_aligned_cov50.py` | 在 **v2.1 稳定后** 再跑 |

**读表规则**（《全面总结》§10）：同时看 `avg_total_cost_ms`、`p95_sla_excess`、`severe_sla_violations`、`migration_share`、**新增** `avg_internal_path_ms`。

---

## 6. 原 Action Plan 推演（修正版）

**场景**：`Data_Heavy` 网关与重库跨机 50 km，边 `traffic=25609`。

| 量 | 估算 |
|----|------|
| `actual_rpc_calls` | max(1, 25609×0.01) ≈ **256** |
| `base_edge_ms` | 50/200 + 2 ≈ **2.25 ms** |
| `effective_edge_ms` | ≈ **576 ms** |
| `L_internal`（单链主导时） | ≈ **576 ms** |
| `L_e2e` | ≈ 2 ms + 576 ms |
| `E_q_ms` | ≈ 574 ms → `E_q` ≈ 1.15×10⁵ km 等价 |
| `β·E_q²`（β=80） | **极大** → 必须配合 **β_internal 下调 / log 惩罚 / clip / 增大 S** |

**预期行为**（在 D1 标定成功后）：

- Critic 对「高 traffic 跨机」给出强负反馈；
- GAT 利用边权特征学会 **高调用边同机或近机**；
- SA 因 `total_cost_ms` 含 `L_internal` 也会拒绝劣质拆分。

---

## 7. 决策摘要

| 决策 | 建议 |
|------|------|
| 是否做三步重构？ | **是**，但拆为 **D0 → D1 → D2**，不要一步改完 |
| 是否放弃 Reward v2.0？ | **否**；v2.0 作对照，v2.1 为拓扑感知演进 |
| 是否立即改 SA？ | D0 后 SA 自动用新 `total_cost_ms`，无需改 SA 逻辑 |
| 最大风险 | 惩罚量级 + train/infer 不一致；用诊断字段与 2 epoch 快筛 |
| 文档 | 《全面总结》§4.4/§7、《Reward v2》§5/§8 在 D1 合并修订 |

---

## 8. 近期实验索引（v2 线，供对照）

| 目录 | 要点 |
|------|------|
| `experiments/medium_validation_20260526_reward_v2_scale5000_v1` | S=5000；Re 推理 0 迁移；P95≈31 km |
| `experiments/medium_validation_20260526_reward_v2_scale10000_v1` | S=10000 + future_gain 修复；Pro P95≈5.9 km；Re P95≈31 km |
| `experiments/medium_validation_20260526_phaseC_softguard_v1` | v1 标杆：GAT Pro 总成本≈18.6k，P95≈4.9 km |
| `experiments/medium_validation_20260526_phaseB2_reward_align_v1` | v1 + MAX=1：GAT Pro 总成本≈15.4k |

---

## 9. 下一步（待你确认后动代码）

1. **实现阶段 D0**（仅 `core/reward.py` + `marl_reward` 边代价对齐 + 冒烟实验）  
2. 评审 D0 的 `avg_internal_path_ms` 与 SA 成本分布  
3. **实现阶段 D1**（`REWARD_V2_USE_INTERNAL_PATH` + counterfactual + 文档）  
4. 2 epoch 快筛 → 8 epoch；再与 Phase C / B2 对比  

如同意，可从 **D0.1–D0.3** 开始提交代码变更。
