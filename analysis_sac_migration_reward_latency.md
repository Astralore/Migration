# `command.md` 修改计划 — 对照当前实现的审查报告（覆盖更新）

**审查基准**：仓库内 **`command.md`（已由本次任务同步修订）** 与 **`core/reward.py`、`core/context.py`、`algorithms/hybrid_sac.py`、`core/geo.py`、`core/state_builder.py`** 及 **`sa.py` / `dqn.py` / `hybrid_sa_dqn.py`** 的调用关系。

---

## 1. 修订概要（相对原 70 行左右草稿）

| 原问题 | 本次在 `command.md` 中的处理 |
|--------|------------------------------|
| Markdown 代码块断裂、阶段编号与实现脱节 | 重写为 **结构化章节 + 合法围栏代码**；增加 **§0 影响面清单**。 |
| `MIN_BW_MBPS` 等常量缺失 | 在 **§2.1** 显式写出 JIT 常量建议值。 |
| 仅 tearing、丢失原 `communication_cost` | **§2.4** 增加 **`comm_delay_ms`**，与现 `norm_traffic * dist` 语义承接。 |
| Future 与旧 `FUTURE_DIST_THRESHOLD` 脱节 | **§2.5** 要求 **超阈值 excess + 衰减 + `/weight_sum`**，与现双重循环语义对齐。 |
| `check_sla_violation` 仍传 `current_dag_reward` | **§0 / §4** 写明 **删参** 及 **`sa`/`dqn`/`hybrid_sac`/`hybrid_sa_dqn`** 全部改调用。 |
| 全非法 mask 无脑 STAY | **§3** 固定 **回退 `action_mask[1]=True`（FOLLOW_SA）**，与 `SACDiscreteActor` **0/1/2** 定义一致。 |
| 阈值多源 | **§1 / §4** 要求 **`SLA_DISTANCE` 单一真源**、`USER_SLA_TOLERANCE_MS` 与 **`calc_access_latency_ms(15km)`** 对齐建议。 |
| Future「BASE 幽灵」 | **§1** 增加 **`propagation_latency_ms`**；**§2.5** 超额 **只用传播 ms**，安全步 **Future=0**。 |
| Reward 截断与 SLA 平原 | **§2.8** 使用 **`REWARD_CLIP_MIN`（如 -10000）< -(SLA_PENALTY + 附加成本)`**，保留违规域内梯度。 |
| Tearing 过小 | **§2.3** **写死** `raw_rpc_count = traffic` 或 **`norm_traffic * max_traffic` 还原**，再 `* RPC_SIZE_MB` + **Cap**。 |
| Reward SLA 与 Context 双条件错位 | **§2.7** 与 **§4**：`sla_penalty_ms` 用 **空间 OR `access_latency_ms>USER`**；多入口时 **Context qos 与 §2.6 同源（建议 max 入口）**。 |
| Actor mask 仅 1D | **§3**：**1D / 2D** 全非法回退，`mask[1]=True` 不得用于 batch。 |

---

## 2. 审查结论：当前方案与代码的一致性

### 2.1 通过项

- **`physics_utils` + `calc_access_latency_ms`**：满足 **无环**（`context`/`reward` 单向依赖），且 **`FIBER_SPEED_KM_MS` 单位 km/ms** 与公式一致。  
- **JIT `risk_ratio**2`**：与现 **`reward.py`** 中 `max_entry_dist` / `SLA_DISTANCE_THRESHOLD` 定义兼容。  
- **迁移按节点累加**、**tearing 用 `min(..., MAX_TEARING_MB)` + 原始 RPC 或还原流量**：与 **DAG 边权量级** 一致，避免 **norm×5KB≈0** 失效。  
- **Access 取多入口 max**：与「木桶网关」直觉及现多 `entry_nodes` 循环一致。  
- **`sla_penalty_ms` 与线性 `calc_access` 分离声明**：避免旧版 **乘子 + 固定罚** 未文档化叠加。  
- **Actor mask + PyTorch bool + 全非法回退 FOLLOW_SA**：与 **`SACDiscreteActor`** 注释及 **SA 草稿** 一致。  
- **向量化落点**：`geo.find_k_nearest_servers`、`context` 预测环、`reward` future、`state_builder` mobility，与热点路径一致。

### 2.2 实施时仍须注意的残留风险

1. **`reactive_violation = spatial or qos`**：若 **`USER_SLA_TOLERANCE_MS = calc_access_latency_ms(15km)`**，则 **`qos` 与 `spatial` 几乎同时真**，`qos` 多为冗余；若容限 **显著大于** 该值，则 **`spatial` 仍保 15 km 硬边界**——文档已要求双条件，实现时 **不要用宽松 ms 替代 km**。  
2. **`get_trigger_type` 去 `current_dag_reward`**：四处调用（`hybrid_sac`×2、`sa`、`dqn`、`hybrid_sa_dqn`）**漏改一处即运行时错误**；建议在 PR 中 **grep 校验**。  
3. **`details` 键名**：`hybrid_sac` 等若依赖 **`details['access_latency']`** 等旧键，需在 **`reward`** 内 **双写**或统一改读取侧，否则训练日志/分解图断裂。  
4. **`comm_delay_ms` + `tearing_delay_ms`**：与旧 **`beta*comm` + tearing** 权重不同；若需保持量级，可暂时引入 **`beta_ms`/`tear_scale`** 并在验收中对比曲线——当前 `command.md` 为 **1:1 相加**，重训后调参可能仍必要。  
5. **`reward` 截断**：已改为 **`REWARD_CLIP_MIN`（如 -10000）** 显式大于 **`SLA_PENALTY_MS`**，避免 **违规+不同附加成本** 全部被钳到同一标量；实施时仍须与 **critic 尺度 / Huber** 联调。

### 2.3 未在 `command.md` 展开但建议后续单开任务

- **`microservice_simulated_annealing`** 内对 **`calculate_microservice_reward`** 的调用：新 ms 标度下 **SA 温度与迭代次数** 是否仍合适，需 **回归实验**。  
- **`build_action_mask` 拓扑规则**：施工图要求 **mask**，具体 **何时禁 STAY / 禁 NEAREST** 仍需设计（可引用现 **`sample_action_with_mask`** 设计意图或 DAG 邻居约束）。

---

## 3. 与历史问题的映射（归档）

- **量纲致命歧义（200 vs 200000）**：已由 **`km/ms` 常量 + 单一公式** 固化。  
- **JIT 方向反了**：已由 **`risk_ratio**2` + MIN/MAX 带宽** 修正。  
- **触发器与 reward 耦合**：已由 **删 `SLA_REWARD_THRESHOLD` 路径 + 物理双条件** 解决方向。  
- **Eval 0 迁移**：由 **mask + FOLLOW_SA 回退 + 训练/推断同路径** 覆盖。

---

## 4. 总评

| 维度 | 结论 |
|------|------|
| 与当前项目对齐度 | **高**：显式列出 **调用链、动作编号、四算法改签名、`comm`/`future` 与旧 reward 对齐策略**。 |
| 可执行性 | **可分工实施**；最大集成风险在 **`get_trigger_type` 签名 + `details` 兼容**。 |
| 建议 | 合并前 **全仓库 grep `current_dag_reward` / `get_trigger_type`**；合并后 **`run_comparison.py` 全量推理烟测**。 |

---

*本文件为 **`command.md` 同步修订版** 的审查报告，并覆盖上一版分析报告正文。*  
*维护：与仓库 `command.md`、`analysis_sac_migration_reward_latency.md` 同次更新。*
