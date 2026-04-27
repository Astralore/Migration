# 微服务迁移：物理毫秒化与 Actor Mask 重构施工图（与当前实现对齐）

> 角色：按本仓库 **现有调用关系**（`sa.py` / `dqn.py` / `hybrid_sac.py` / `hybrid_sa_dqn.py` → `context.get_trigger_type` → `reward.calculate_microservice_reward`）实施；禁止引入 `reward`↔`context` 循环导入。

---

## 0. 影响面清单（实施前必读）

| 模块 | 变更要点 |
|------|----------|
| `core/physics_utils.py` | **新建**：光速常量、`calc_access_latency_ms`；可再放 **仅与物理相关的** 阈值常量（见下）。 |
| `core/reward.py` | `calculate_microservice_reward` 改为全 ms 分量与 `total_cost_ms`；`details` 字段名建议保留或做映射，避免 `hybrid_sac` 等依赖方断裂。 |
| `core/context.py` | `check_sla_violation` **去掉** `current_dag_reward`；`get_trigger_type` **去掉**该形参；`check_proactive_sla_violation` 同步改签名。 |
| `algorithms/hybrid_sac.py` | 动作 **0=STAY / 1=FOLLOW_SA / 2=NEAREST**（与 `SACDiscreteActor` 注释一致）；`sample_action` 与 `get_action_deterministic` **统一走带 mask 的路径**；若尚无 mask 构造逻辑，需在 **主循环内** 或 `state_builder` 中实现 `build_action_mask(...)`。 |
| `algorithms/sa.py` / `dqn.py` / `hybrid_sa_dqn.py` | 所有 `get_trigger_type(..., current_dag_reward, ...)` 改为 **不传 reward**（或传 `None` 若暂做兼容层）。 |
| `core/geo.py` | 可选：`find_k_nearest_servers` 对 **600 台** 服务器做 **numpy 距离矩阵**（与向量化阶段一致）。 |
| `core/state_builder.py` | 批量 haversine / 与预测相关的 **mobility 特征** 向量化。 |

**训练**：奖励尺度与 MDP 变更后须 **废弃旧 checkpoint**，`run_comparison.py` 中 `INFERENCE_MODE=False` 重训。

---

## 阶段一：`core/physics_utils.py`（无环、单位无歧义）

1. **常量**（与当前 `command` 审查结论一致）  
   - `FIBER_SPEED_KM_MS = 200.0`  # 语义：**km/ms**（≈ 200,000 km/s）  
   - `BASE_ROUTER_DELAY_MS = 2.0`  
   - **禁止**再使用「名为 `_KM_S` 却赋 200.0 表示 km/s」的写法。

2. **公共函数**（供 `reward`、`context` **单向** `import`）：

```python
def calc_access_latency_ms(distance_km: float) -> float:
    """一次真实接入：传播 + 固定路由/协议基线 (ms)。用于用户→入口、跨边传播等「新开连接」语义。"""
    return (distance_km / FIBER_SPEED_KM_MS) + BASE_ROUTER_DELAY_MS


def propagation_latency_ms(distance_km: float) -> float:
    """仅传播段 (ms)，不含 BASE_ROUTER_DELAY。distance_km<=0 时须返回 0.0。
    用于 Future 超额距离等「非新开物理端口」的惩罚，避免安全步吃到 2ms 幽灵基线。"""
    if distance_km <= 0.0:
        return 0.0
    return distance_km / FIBER_SPEED_KM_MS
```

**单位**：`FIBER_SPEED_KM_MS` 为 **km/ms** 时，`distance_km / FIBER_SPEED_KM_MS` 已为 **ms**，**禁止**再乘 `1000`（否则与阶段一矛盾）。

3. **可选**：将 `SLA_DISTANCE_THRESHOLD_KM = 15.0` 放在本文件或继续仅在 `context.py` 定义一处，**`reward.py` 中 SLA 判断必须与 `context.DISTANCE_THRESHOLD_KM` 同源**（建议 `context` 从 `physics_utils` 只 import 函数，阈值仍保留在 `context` 并在 `reward` 中 `from core.context import DISTANCE_THRESHOLD_KM` 或反向由 `physics_utils` 导出常量二选一，**全仓库单一真源**）。

---

## 阶段二：`core/reward.py` — 全毫秒 `total_cost_ms`

### 2.1 JIT 带宽（与现实现 `risk_ratio` 定义一致）

- `risk_ratio = min(max_entry_dist_km / SLA_DISTANCE_THRESHOLD, 1.0)`，其中 `max_entry_dist_km` 与当前实现对 **各 entry** 取 max 一致。  
- 常量建议：`MIN_BW_MBPS = 50.0`，`MAX_BW_MBPS = 500.0`（与先前规格一致，可微调）。  
- `effective_bandwidth = MIN_BW_MBPS + (MAX_BW_MBPS - MIN_BW_MBPS) * (risk_ratio ** 2)`  
- **Proactive / Reactive 不对称**：若需保留「Reactive 迁移更痛」，在 **`migration_delay_ms` 上** 乘 `TRIGGER_REACTIVE` 对应系数（从 `core.context` 引入常量），**不要**再依赖 `current_dag_reward` 触发。

### 2.2 多节点迁移（**Per-node 累加**，对齐当前 `for node ... if assignment changed`）

对 **`current_assignments[node] != previous_assignments[node]`** 的每个节点：

`migration_delay_ms += ((image_mb + state_mb) / effective_bandwidth) * 1000.0`

（无状态节点 `state_mb=0` 仍只迁镜像。）

### 2.3 Tearing（**原始 RPC 次数** + **Cap**，避免「归一化 × 5KB」过小）

**暗坑**：若用 `norm_traffic * RPC_SIZE_MB` 且 `norm_traffic∈[0,1]`，则每边等效流量 **亚 MB 级**，除以 Gbps 骨干后延迟 **≈0**，撕裂约束名存实亡。

**定案**（写死，实现勿猜）：

- 常量：`RPC_SIZE_MB = 0.005`，`MAX_TEARING_MB = 50.0`，`EDGE_BACKHAUL_MBPS = 1000.0`  
- 边上 **原始调用次数** 记为 `traffic`（与 `dag_info['edges'][(src,dst)]` 一致，即当前 `reward` 循环里的 `traffic`）。  
- 仅 **`src_server != dst_server`** 的边：  
  `cross_mb = min(traffic * RPC_SIZE_MB, MAX_TEARING_MB)`  
  `tearing_delay_ms += (cross_mb / EDGE_BACKHAUL_MBPS) * 1000.0`  

**若某路径上只剩 `norm_traffic`**：先在同一 DAG 内取 `max_traffic = max(dag_info['edges'].values())`，再 **`raw_rpc_count = norm_traffic * max_traffic`**，然后 **`cross_mb = min(raw_rpc_count * RPC_SIZE_MB, MAX_TEARING_MB)`**。

### 2.4 跨边「传播型」通信延迟（**承接**原 `communication_cost`）

当前实现有独立的 **`norm_traffic * dist_km`** 边项；全 ms 化后建议单独一项，避免信息丢失：

- 对跨服务器边：`comm_delay_ms += norm_traffic * calc_access_latency_ms(edge_dist_km)`  
- 或与 tearing 合并文档化；**`total_cost_ms` 中至少保留一种「边级延迟」**。  

### 2.5 Future（**衰减 + 归一 + 与 `FUTURE_DIST_THRESHOLD` 对齐**；**禁止 BASE 幽灵**）

与现 **`reward.py`** 一致：仅对 **预测距离超过 `FUTURE_DIST_THRESHOLD`（km）** 的 **超额部分** 计惩罚。

**暗坑**：若对 `excess_km` 使用 `calc_access_latency_ms(excess_km)`，则 **`excess_km=0` 时仍有 `BASE_ROUTER_DELAY_MS`**，未来全程安全也会出现 **≈2ms/步** 的虚假 Future 成本，抬高总 cost 基线。

**定案**：超额部分 **只计传播 ms**，**不加** `BASE_ROUTER_DELAY_MS`（用阶段一的 **`propagation_latency_ms`**）：

- `len(future_distances)==0` → `future_delay_ms = 0.0`  
- 否则（`d` 为该步用于判定的距离 km，多入口时取该步 **各 entry 的最大距离** 再与阈值比较）：  
  `excess_km = max(0.0, d - FUTURE_DIST_THRESHOLD)`  
  `raw = sum(propagation_latency_ms(excess_km) * (FUTURE_DECAY ** i) for i, d in enumerate(future_distances))`  
  `weight_sum = sum(FUTURE_DECAY ** i for i in range(len(future_distances)))`  
  `future_delay_ms = raw / weight_sum`  

当所有步均安全时，各步 `excess_km=0` → **`future_delay_ms = 0`**。

### 2.6 多入口 Access（木桶：**max**）

`access_latency_ms = max(calc_access_latency_ms(dist_km) for dist_km in entry_distances_km)`

### 2.7 SLA 显式项（与文档「双重计价」声明一致；**与阶段四双条件对齐**）

**暗坑**：若阶段四已用 **`reactive_violation = spatial or qos`**（空间超阈 **或** 纯接入延迟超 `USER_SLA_TOLERANCE_MS`），而 Reward 仅在 **`max(entry_distances_km) > SLA_DISTANCE_THRESHOLD`** 时扣 **`sla_penalty_ms`**，则会出现 **Context 已判违规、Reward 不扣 5000** 的 **监督信号错位**（模型可钻空子）。

**定案**：`sla_penalty_ms` 的触发条件与 **阶段四** 在语义上 **同构**（空间 **OR** 接入延迟 QoS）：

- **线性**：已含在 **`access_latency_ms`**（§2.6：`max(calc_access_latency_ms(dist_km) for dist_km in entry_distances_km)`）随各入口距离增长。  
- **跳变**：当  
  **`max(entry_distances_km) > SLA_DISTANCE_THRESHOLD`** **OR** **`access_latency_ms > USER_SLA_TOLERANCE_MS`**  
  时，`sla_penalty_ms = SLA_PENALTY_MS`（如 `5000.0`），否则 `0.0`。  

（第二项即「各入口接入延迟取木桶最劣后仍超 QoS 容限」；与 `max(calc_access_latency_ms(d) for d in ...)` 是否 **严格大于** `USER_SLA_TOLERANCE_MS` 与 §2.6 的 **`access_latency_ms`** 为 **同一标量**，实现时 **禁止** 另算一套口径。）

- **多入口与 Context 对齐（建议）**：阶段四若仅对 **单一 gateway 距离** 做 `qos`，而 Reward 用 **多入口 max**，仍存在边缘不一致；建议在 **`get_trigger_type` / `check_sla_violation`** 中对 **所有 entry** 计算 `calc_access_latency_ms` 并取 **max** 再与 **`USER_SLA_TOLERANCE_MS`** 比较，与 §2.6 **同源**。

- **禁止**在 `access_latency_ms` 上再乘旧版 `SLA_VIOLATION_MULTIPLIER`（避免与 `sla_penalty_ms` 三重叠加）。

### 2.8 合并与 Reward 截断（**避免「平原效应」**）

`total_cost_ms = access_latency_ms + migration_delay_ms + tearing_delay_ms + comm_delay_ms + future_delay_ms + sla_penalty_ms`  

**暗坑**：若 `sla_penalty_ms = 5000` 且 **`reward = max(-total_cost_ms, -5000)`**，则 **`total_cost_ms ∈ [5000, 5000+其它)`** 时 reward **全部被钳到 -5000**（违规又重迁 vs 违规少迁 **同分**），Critic 在违规邻域 **梯度平原**。

**定案**：截断下限的绝对值须 **大于** 单次 SLA 跳变 + 典型附加成本上界，留出「违规 + 不同附加代价」的可分梯度空间。建议：

```python
SLA_PENALTY_MS = 5000.0
REWARD_CLIP_MIN = -10000.0  # 可随训练再调，须满足 REWARD_CLIP_MIN < -(SLA_PENALTY_MS + 典型 migration/future 上界)
reward = max(-total_cost_ms, REWARD_CLIP_MIN)
```

---

## 阶段三：`algorithms/hybrid_sac.py` — Logits Mask（PyTorch）

**动作语义**：`0=STAY`，`1=FOLLOW_SA`，`2=NEAREST`（与 `SACDiscreteActor` 类注释一致）。

1. 在 **`forward` 末尾** 或 **独立方法** `forward_masked(..., action_mask)` 中：先算 `logits = self.policy_net(state)`，再应用 mask。  
2. **设备与 dtype**：`action_mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=logits.device)`；若 mask 为 float「1=合法」，先 `> 0` 转 bool。  
3. **全非法保护**（禁止无脑全 STAY；**兼容 1D 推断与 2D Batch 训练**）：  
   - **1D** `mask.shape == (action_dim,)`：`if not mask.any(): mask[1] = True`（**FOLLOW_SA**）。  
   - **2D** `mask.shape == (batch_size, action_dim)`：对 **`~mask.any(dim=-1)`** 的行，将 **`mask[row, 1] = True`**；**禁止**写 `mask[1] = True`（会把 **batch 维整条** 误标为合法或触发维度错误）。  

```python
if action_mask_tensor.dim() == 1:
    if not action_mask_tensor.any():
        action_mask_tensor[1] = True
else:
    all_invalid = ~action_mask_tensor.any(dim=-1)
    action_mask_tensor[all_invalid, 1] = True
```

4. **`logits = logits.masked_fill(~action_mask_tensor, -1e9)`** 或等价写法；**batch 维**时 **`logits`** 与 **`action_mask_tensor`** 均为 **`(batch, action_dim)`**。  
5. **`sample_action` / `get_action_deterministic`**：均改为调用上述逻辑（可复用 `sample_action_with_mask` 并新增 `get_action_deterministic_masked`，避免两套实现）。  
6. **Critic**：训练时对非法动作的 Q 处理与现 **离散 SAC** 实现保持一致（仅合法动作参与 `log_pi` 或文档规定的方式）。

---

## 阶段四：`core/context.py` — 触发器与 Reward 解耦

1. `from core.physics_utils import calc_access_latency_ms`（**不得**反向让 `physics_utils` import `context`）。  
2. **`USER_SLA_TOLERANCE_MS` 标定**（推荐，与空间 SLA 一致）：  
   `USER_SLA_TOLERANCE_MS = calc_access_latency_ms(DISTANCE_THRESHOLD_KM)`  
   或略小（如 `* 0.99`）用于「纯延迟」略早于 15 km 的告警；**须与产品语义一致**。  
3. **Reactive 双条件**（防 SLA 被架空；**与 §2.7 对齐**）：  

单入口（仅 `gateway_node`）时可写：

```python
dist_km = haversine_distance(user_lat, user_lon, gateway_server_lat, gateway_server_lon)
spatial = dist_km > DISTANCE_THRESHOLD_KM
qos = calc_access_latency_ms(dist_km) > USER_SLA_TOLERANCE_MS
reactive_violation = spatial or qos
```

**多入口（建议与 `reward` 一致）**：对每个 entry 算 `dist_e`，令 **`dist_worst_km = max(dist_e)`**，**`spatial = dist_worst_km > DISTANCE_THRESHOLD_KM`**，**`qos = max(calc_access_latency_ms(dist_e) for ...) > USER_SLA_TOLERANCE_MS`**（与 §2.6 **`access_latency_ms`** 定义一致），再 **`reactive_violation = spatial or qos`**。

4. **删除** `current_dag_reward < SLA_REWARD_THRESHOLD` 分支；`check_sla_violation` / `get_trigger_type` / `check_proactive_sla_violation` **删除** `current_dag_reward` 参数，并更新 **全部调用方**（见第 0 节）。  
5. **Proactive 前瞻**：仍使用 `PROACTIVE_WARNING_KM` 与 `predicted_locations` 循环；向量化见阶段五。

---

## 阶段五：向量化（禁止 `lru_cache`）

1. **`core/geo.py`**：`find_k_nearest_servers` 对固定 `servers_df` 可预计算坐标矩阵，单次查询为 **向量距离 → `argpartition`** 取 top-k。  
2. **`core/context.py`**：`get_trigger_type` 内对未来点与网关的 haversine，改为 **`numpy` 广播**（`(H,)` 对常量 gw）。  
3. **`core/reward.py`**：`future_delay_ms` 中入口距离矩阵 **(H, n_entry)** 向量化。  
4. **`core/state_builder.py`**：`_mobility_features` 等对 `predicted_locations` 的循环改为 **numpy `mean(axis=0)`**。

---

## 验收要求（提交审查用）

1. `core/physics_utils.py`：**完整**常量 + `calc_access_latency_ms` 源码。  
2. `core/context.py`：**双条件** `reactive_violation` + **已移除** `current_dag_reward` 的 `check_sla_violation` 签名。  
3. `core/reward.py`：**一行** `total_cost_ms = ...`；**`cross_mb`** 使用 **`traffic` 或 `norm_traffic*max_traffic` 还原**；**`future_delay_ms`** 使用 **`propagation_latency_ms(excess_km)`**（**非** `calc_access_latency_ms(excess_km)`）+ **`/ weight_sum`**；**`reward = max(..., REWARD_CLIP_MIN)`** 且 **`REWARD_CLIP_MIN <= -10000`** 或与 `SLA_PENALTY_MS` 显式留裕度。  
4. `algorithms/hybrid_sac.py`：**Tensor mask** + **全非法回退 FOLLOW_SA（索引 1）**；**1D/2D mask** 分支（`dim==1` vs `all_invalid = ~mask.any(dim=-1); mask[all_invalid,1]=True`）+ `sample_action`/`get_action_deterministic` **均已接入**。  

---

*本文件已根据当前仓库实现（`reward.py` 四分量、`context.py` 触发链、`hybrid_sac.py` Actor 动作定义、多算法调用 `get_trigger_type`）做补充与排版修正。*
