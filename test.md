# 分阶段验证报告（覆盖 `command.md` 阶段一、二、三、四、五）

**依据**：工作区 `command.md`（物理毫秒化 + Context 解耦 + Hybrid SAC Logits Mask + 向量化）。  
**实施日期**：以本文件为准。

---

## 阶段一：`core/physics_utils.py`

| 项 | 结果 |
|----|------|
| 文件已创建 | `core/physics_utils.py` |
| 常量写死 | `FIBER_SPEED_KM_MS = 200.0`（km/ms）、`BASE_ROUTER_DELAY_MS = 2.0` |
| 函数 | `calc_access_latency_ms`、`propagation_latency_ms` |
| 导入环 | `python -c "from core import physics_utils, context, reward"` **通过** |

---

## 阶段二：`core/reward.py`

| 项 | 结果 |
|----|------|
| `calculate_microservice_reward` 已改为 **ms 量纲** `total_cost_ms` | 通过 |
| 烟测 | 使用 `IoT_Lightweight_DAG` + `edge_server_locations.csv` 单次调用 **无异常**，返回 `reward` 与 `details['total_cost_ms']` 一致 |

### `total_cost_ms` 求和代码（供审查）

以下为 `core/reward.py` 中 **合并六项** 的源码片段：

```python
    total_cost_ms = (
        access_latency_ms
        + migration_delay_ms
        + tearing_delay_ms
        + comm_delay_ms
        + future_delay_ms
        + sla_penalty_ms
    )

    reward = max(-total_cost_ms, REWARD_CLIP_MIN)
```

（`REWARD_CLIP_MIN = -10000.0`，`SLA_PENALTY_MS = 5000.0`，与 `command.md` §2.8 一致。）

---

## 阶段四：`core/context.py` + 调用方同步

| 项 | 结果 |
|----|------|
| `check_sla_violation` | 已移除 `current_dag_reward`；**`spatial or qos`**，`qos` 使用 `calc_access_latency_ms(dist_km) > USER_SLA_TOLERANCE_MS` |
| `USER_SLA_TOLERANCE_MS` | `calc_access_latency_ms(DISTANCE_THRESHOLD_KM) * 0.99`（实测约 **2.05425 ms**） |
| `get_trigger_type` / `check_proactive_sla_violation` | 已同步新签名 |
| 调用方已改 | `algorithms/sa.py`、`dqn.py`、`hybrid_sac.py`（2 处）、`hybrid_sa_dqn.py` 中 **`get_trigger_type` 不再传入 `current_dag_reward`**；触发前多余的 **`calculate_microservice_reward`（仅用于取 reward 判触发）** 已删除 |
| 批量导入 | `python -c "import algorithms.sa; import algorithms.dqn; import algorithms.hybrid_sac; import algorithms.hybrid_sa_dqn"` **通过** |

### `get_trigger_type` 函数签名（供审查）

```text
get_trigger_type(
    user_lat, user_lon,
    gateway_server_lat, gateway_server_lon,
    predicted_locations=None,
    proactive_enabled=False,
)
```

（`inspect.signature` 与源码一致。）

---

## 阶段三：`algorithms/hybrid_sac.py` — Actor Logits Mask（PyTorch）

| 项 | 结果 |
|----|------|
| `build_microservice_action_mask` | STAY / FOLLOW_SA 恒为 `True`；无候选时 NEAREST 为 `False` |
| `apply_action_mask_to_logits` | 非法 logits `masked_fill(-1e9)`；**全非法行**强制 `fallback_action_index=1`（FOLLOW_SA），兼容 1D/2D |
| `SACDiscreteActor.forward` | Softmax 前对 `logits_raw` 应用掩码；`action_mask` 经 `as_tensor(..., device=logits.device)` 对齐 device |
| `sample_action` / `get_action_deterministic` | 均经 `forward(..., action_mask)` 接入同一套逻辑 |
| `optimize_sac` replay | 支持 6/7 元组；训练步使用存储的 `mask_cpu`；bootstrap 下一状态 Actor 使用全 1 mask（与 `command.md` 折中一致） |

### 处理 logits 与 1D/2D 全非法保护（核心片段，供审查）

```python
def apply_action_mask_to_logits(
    logits: torch.Tensor,
    action_mask: Optional[torch.Tensor],
    *,
    fallback_action_index: int = ACTION_FOLLOW_SA,
    fill_value: float = -1e9,
) -> torch.Tensor:
    if action_mask is None:
        return logits

    squeeze_out = logits.dim() == 1
    logits_work = logits.unsqueeze(0) if squeeze_out else logits

    m = torch.as_tensor(action_mask, device=logits_work.device, dtype=torch.bool)
    if m.dim() == 1:
        m = m.unsqueeze(0).expand(logits_work.shape[0], -1)
    if m.shape != logits_work.shape:
        raise ValueError(f"action_mask shape {m.shape} != logits {logits_work.shape}")

    m = m.clone()
    all_invalid = ~m.any(dim=-1)
    if all_invalid.any():
        m[all_invalid, fallback_action_index] = True

    out = logits_work.masked_fill(~m, fill_value)
    return out.squeeze(0) if squeeze_out else out
```

```python
    def forward(self, node_embedding, sa_prior, action_mask: Optional[torch.Tensor] = None):
        ...
        logits_raw = self.policy_net(state)
        logits = apply_action_mask_to_logits(logits_raw, action_mask)
        action_probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        ...

    def sample_action(self, node_embedding, sa_prior, action_mask: Optional[torch.Tensor] = None):
        action_probs, log_probs = self.forward(node_embedding, sa_prior, action_mask)
        ...

    def get_action_deterministic(self, node_embedding, sa_prior, action_mask: Optional[torch.Tensor] = None):
        action_probs, _ = self.forward(node_embedding, sa_prior, action_mask)
        return int(action_probs.argmax().item())
```

---

## 阶段五：向量化（`core/geo.py`、`context.py`、`reward.py`、`state_builder.py`）

| 文件 | 向量化落点 | 说明 |
|------|------------|------|
| `core/geo.py` | `haversine_distance` | 支持 NumPy 广播（文档化；公式未改） |
| `core/geo.py` | `find_k_nearest_servers` | 整表 `latitude`/`longitude` 一次 `haversine_distance` → `argpartition` + **稳定** `argsort`（并列距离与旧 `iterrows`+`sort` 顺序一致） |
| `core/context.py` | `_future_distances_km_to_gateway` + `check_proactive_sla_violation` / `get_trigger_type` | 前瞻 `(H,)` 对网关常量广播，替代逐点 `for` + haversine |
| `core/reward.py` | `build_servers_info` | 列 `to_numpy` + `dict(zip(...))`，去掉 `iterrows` |
| `core/reward.py` | `calculate_microservice_reward` | 入口距离向量；`access_latency_ms = calc(max(d))`（`d>=0` 上 `calc` 单调，与 `max(calc(d))` 等价）；边列表批量 haversine + 掩码 `same`；`future_delay_ms` 为 `(H, n_entry)` 距离矩阵 + `np.dot` 权重 |
| `core/state_builder.py` | `_mobility_features` | `np.asarray` + `np.mean(axis=0)` |
| `core/state_builder.py` | `build_graph_state` | entry 最大距离与 SA 距离批量 haversine；`sa_stay` 仍逐节点比较（无距离循环） |

**禁区**：未使用 `lru_cache`；未改 `physics_utils` 公式；`find_k_nearest_servers` 经 **全 k / 多坐标** 下与旧实现逐元素比对（含并列距离的 **stable** 序）。

### 实验与回归（本次已跑）

- **`python _verify_physics_reward_context.py`**：**退出码 0**，与向量化前同一套断言（context / reward / 导入）。
- **`find_k_nearest_servers` 无损校验**：对 `edge_server_locations.csv` 在若干 `(lat,lon)` 与 `k in {1,3,5,len(df)}` 下，与 **逐行 `iterrows` + `sort`** 的暴力结果 **逐元一致**（含等距 tie-break）。
- **微基准（本机）**：`find_k_nearest_servers(..., k=3)` 约 **500 次合计 ~20 ms**（量级供参考，非 CI 门禁）。

---

## 自动化测试执行记录（本次已实际跑通）

**脚本**：仓库根目录 **`_verify_physics_reward_context.py`**（可重复执行：`python _verify_physics_reward_context.py`）。

**环境**：`d:\Migration\Migrate-main`，依赖 **`data/edge_server_locations.csv`**。

**终端原始输出**（控制台编码可能导致个别符号显示为乱码，以 **PASS/FAIL** 为准）：

```
=== 1. physics_utils ===
  [PASS] FIBER_SPEED_KM_MS
  [PASS] BASE_ROUTER_DELAY_MS
  [PASS] calc(0)==2
  [PASS] propagation(0)==0
  [PASS] propagation excess
  calc_access_latency_ms(15km) = 2.075000 ms
=== 2. context ===
  [PASS] USER_SLA matches 0.99*calc(15)
  [PASS] check_sla same point
  [PASS] check_sla far
  [PASS] get_trigger safe no proactive
  [PASS] get_trigger proactive — 'PROACTIVE'
=== 3. reward.calculate_microservice_reward ===
  [PASS] reward == -total_cost or clip
  [PASS] details total_cost_ms
  [PASS] future 0 no preds
  [PASS] migration > 0 after change
=== 4. algorithms 导入 ===
  [PASS] import sa, dqn, hybrid_sac, hybrid_sa_dqn

ALL CHECKS PASSED
```

**退出码**：`0`（全部断言通过）。

### 测试用例摘要

| 编号 | 断言内容 | 结果 |
|------|----------|------|
| T1 | `FIBER_SPEED_KM_MS == 200`、`BASE == 2` | PASS |
| T2 | `calc_access_latency_ms(0)==2`、`propagation(0)==0`、`propagation(20)==20/200` | PASS |
| T3 | `USER_SLA_TOLERANCE_MS` 与 `0.99*calc(15km)` 一致 | PASS |
| T4 | `check_sla_violation`：同点不违规、远距离违规 | PASS |
| T5 | `get_trigger_type`：安全无前瞻为 `None`；前瞻超阈为 `PROACTIVE` | PASS |
| T6 | `reward` 与 `total_cost_ms` 一致（或触及 clip）；`details` 中 `total_cost` 与 `total_cost_ms` 一致；无预测时 `future_penalty==0` | PASS |
| T7 | 修改节点分配后 `migration_cost > 0` | PASS |
| T8 | `algorithms` 四模块可导入 | PASS |

**补充**：`python -m compileall core algorithms -q` **退出码 0**（语法编译检查）。

### `run_comparison.py`（阶段三端到端烟测）

**当时配置（阶段三短时烟测，已结束）**：`INFERENCE_MODE = False`；`TRAIN_END_INDEX = 2500`；Hybrid SAC `num_epochs = 2`。

**当前仓库**：`run_comparison.py` 已恢复 **全量训练/对比默认** — `TRAIN_END_INDEX = 10000`，Proactive/Reactive 下 Hybrid SAC **`num_epochs = 6`**（与 `run_all_algorithms` 内 SAC 一致），`INFERENCE_MODE = False`（全量训练后出表；改为 `True` 时在 `[10000:15000)` 上加载 checkpoint 推理）。

**命令**：`python run_comparison.py`（工作目录 `d:\Migration\Migrate-main`）。

**结果摘要（短时烟测那次）**：

- **退出码**：`0`。
- **日志检索**：输出中 **无** `nan` / `NaN` / `RuntimeError` / `dimension` 等与梯度或维度相关的报错。
- **Hybrid SAC（Reactive）**：`Hybrid SAC done in 46.3s`；权重写入 `checkpoints/sac_reactive.pth`；调试摘要中 `Explore actions (Actor sampling): 12`，`Total global steps: 402`。
- **可视化**：`outputs/hybrid_sac_*_training.png` 等生成成功。

---

## 备注

- `calculate_microservice_reward` 仍保留 **`alpha, beta, gamma, delta` 形参** 以兼容旧调用，当前实现中 **未再参与** `total_cost_ms`（与 `command.md` 全 ms 相加式一致）。  
- `details['state_divisor']` 现为 **JIT 的 `effective_bandwidth`（Mbps）**，便于排查；若下游误当旧「state_divisor」语义，需在文档中说明。
