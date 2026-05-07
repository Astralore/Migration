# 当前训练/推理效果异常的原因分析与修改建议

本文基于当前代码实现、`result.md` 的全量流水线结果，以及上一版 `command.md` 中提出的三个假设，判断哪些问题确实可能导致当前效果偏差，哪些只是指标展示问题，并给出适配现有代码的修改顺序。

---

## 0. 当前结果的主要异常

`result.md` 中最值得关注的不是 Hybrid SAC 是否“会迁移”，而是它在 Proactive 下迁移过度：

| 阶段 | 算法 | Migrations | Violations | 现象 |
|------|------|------------|------------|------|
| 训练 Proactive | SA | 2827 | 18185 | 迁移少，违规高 |
| 训练 Proactive | DQN | 11734 | 16855 | 中等迁移 |
| 训练 Proactive | Hybrid SAC | 33471 | 13523 | 违规最低，但迁移极高，Score 很差 |
| 测试 Proactive | SA | 527 | 8980 | 迁移少 |
| 测试 Proactive | DQN | 3718 | 8967 | 中等迁移 |
| 测试 Proactive | Hybrid SAC | 35114 | 8257 | 违规最低，但迁移远超基线 |

当前的主要问题可概括为：

- **Hybrid SAC 已经不再是“全 0 STAY 死锁”**，但转向了另一个极端：Proactive 下过度迁移。
- **SA 的 Proactive DAG 统计中迁移节点为 0**，说明 SA 在某些 proactive 决策里几乎不改变 DAG 放置，可能是退火搜索太冷、邻域太窄，或 reward 让“迁移”不划算。
- **DQN 的 Avg Latency 为 0.00 ms** 是计时缺失，不是算法效果问题。

---

## 1. 对上一版三条修改建议的判断

| 建议 | 是否可能解释当前效果差 | 是否适合当前实现 | 建议优先级 |
|------|------------------------|------------------|------------|
| `migration_delay_ms` 补 `MB -> Mb` 的 `*8`，并加入迁移固定开销 | **是，尤其能解释 Hybrid SAC 过度迁移** | **适合，但必须作为 reward 版本变更重跑全实验** | 高 |
| SA 初始温度从 `100` 提到 `5000` | **能解释 SA 过冷，但不能解释 Hybrid SAC 过迁移** | **适合，但建议先加接受率日志再标定温度，不建议盲目只改一个常数** | 中 |
| DQN 补内部决策计时 | **不能解释效果差，只解释 0.00 ms 展示异常** | **适合，低风险** | 中低 |

结论：如果目标是改善当前 `result.md` 中 Hybrid SAC 的“高迁移低违规但 Score 很差”，最关键的是 **reward 的迁移成本尺度**，其次才是 SA 的温度标定。DQN 计时只是报告口径修复。

---

## 2. 核心原因一：迁移代价被系统性低估，Hybrid SAC 学会“用迁移换违规”

### 2.1 当前代码

位置：`core/reward.py`，`calculate_microservice_reward(...)`。

当前迁移公式：

```python
delta_ms = ((image_mb + state_mb) / effective_bandwidth) * 1000.0
if trigger_type == TRIGGER_REACTIVE:
    delta_ms *= REACTIVE_MIGRATION_MULT
migration_delay_ms += delta_ms
```

其中：

- `image_mb` / `state_mb`：DAG 节点配置中以 MB 表示的镜像和状态大小。
- `effective_bandwidth`：由 `MIN_BW_MBPS=50`、`MAX_BW_MBPS=500` 插值得到，变量名是 Mbps。

如果这些变量名反映真实单位，则当前公式存在单位不一致：

```text
MB / Mbps = MegaBytes / Megabits_per_second
```

应先将 MB 转成 Mb，即乘 8。

### 2.2 为什么它更容易伤害 Hybrid SAC

SA / DQN / Hybrid SAC 都共享这个 reward，但 Hybrid SAC 的动作结构更容易把该漏洞放大：

- Hybrid SAC 每个触发步先得到一个 SA 提案；
- 然后按 DAG 拓扑序对每个节点做一次离散动作：`STAY / FOLLOW_SA / NEAREST`；
- Proactive 场景下，只要迁移成本偏低，SAC 就很容易学到“主动换位置能减少未来/当前 SLA 违规”的策略；
- 当前 `result.md` 中 Hybrid SAC 的测试 Proactive 结果就是这种形态：`Violations=8257` 最低，但 `Migrations=35114` 极高。

这说明 SAC 不是完全坏掉，而是 reward 中 **迁移惩罚相对 SLA / future penalty 太便宜**。

### 2.3 建议修改

建议在 `core/reward.py` 中显式区分“传输时间”和“固定迁移开销”。

推荐常量：

```python
MB_TO_MBIT = 8.0
BASE_MIGRATION_OVERHEAD_MS = 200.0
```

推荐公式：

```python
transfer_ms = ((image_mb + state_mb) * MB_TO_MBIT / effective_bandwidth) * 1000.0
delta_ms = transfer_ms + BASE_MIGRATION_OVERHEAD_MS
if trigger_type == TRIGGER_REACTIVE:
    delta_ms *= REACTIVE_MIGRATION_MULT
migration_delay_ms += delta_ms
```

### 2.4 是否要直接采用 200 ms

`200.0 ms` 适合作为第一轮工程默认值，但它是超参，不是定律。建议做三组 ablation：

| 实验 | `MB_TO_MBIT` | `BASE_MIGRATION_OVERHEAD_MS` | 目的 |
|------|--------------|------------------------------|------|
| A | 8 | 0 | 只验证单位修正影响 |
| B | 8 | 100 | 轻量固定恢复成本 |
| C | 8 | 200 | 更强抑制 ping-pong |

不建议一开始同时改 `SLA_PENALTY_MS`，否则无法判断到底是迁移项修复还是 SLA 惩罚重标定起作用。

### 2.5 需要同步关注 reward scaling，而不是简单放宽 clip

当前：

```python
REWARD_CLIP_MIN = -10000.0
reward = max(-total_cost_ms, REWARD_CLIP_MIN)
```

迁移成本乘 8 后，`total_cost_ms` 更容易超过 10000。若大量样本都被截断到 `-10000`，SAC 会失去区分“坏”和“极坏”的梯度信号。

但这里不能简单把：

```python
REWARD_CLIP_MIN = -50000.0
```

作为修复。原因是 SAC 的 critic 要拟合折扣累计回报。如果单步 reward 被放宽到几万毫秒量级，`target_value = reward + gamma * V(s')` 很容易进入数万到数十万的负值区间；当前虽然 critic 用了 `smooth_l1_loss`，但过大的 Q 标尺仍会让 critic / actor 更新非常不稳定，甚至出现 NaN 或策略突然坍塌。

因此，**物理成本应保留毫秒单位用于报告，但神经网络训练 reward 应做缩放**。

建议新增：

```python
REWARD_SCALE_MS = 1000.0
REWARD_CLIP_MIN = -10.0
```

并将训练 reward 改为：

```python
reward = max(-total_cost_ms / REWARD_SCALE_MS, REWARD_CLIP_MIN)
```

同时在 `details` 中继续保留真实物理值：

```python
details = {
    "total_cost_ms": total_cost_ms,
    "reward": reward,
    "reward_scale_ms": REWARD_SCALE_MS,
    ...
}
```

这样做的效果：

- `total_cost_ms` 仍是物理毫秒，报告和论文可解释；
- SAC 看到的 reward 落在大约 `[-10, 0]`，critic Q 值尺度更稳定；
- `MB_TO_MBIT=8` 与迁移 overhead 的物理差异仍然保留，只是整体线性缩放。

因此改完迁移成本后必须重新打开：

```powershell
$env:HYBRID_SAC_DEBUG_STEPS="200"
python run_comparison.py
```

重点看：

```text
total_cost_ms=...
reward(clipped)=-10000.0
```

如果采用 reward scaling 后仍大量顶到 `-10.0`，优先调 `REWARD_SCALE_MS`（例如 2000 或 5000），而不是继续把 clip min 拉到巨大负数。

#### 2.5.1 Reward Scaling 后必须同步调整 SAC 的 Alpha（熵坍塌陷阱）

> ⚠️ **架构复核补充**：Reward Scaling 是正确的，但遗漏了 SAC 最核心的特性——**最大熵（Maximum Entropy）**。

**问题分析**：

SAC 的优化目标是 $R + \alpha \mathcal{H}$（奖励 + 熵系数 × 策略熵）。

- 原本 Reward 是 `-5000` 级别，网络为了不扣分会努力寻找最优解
- 现在 Reward 缩小 1000 倍变成 `-5`
- 如果代码里的温度系数 `alpha` 保持不变（当前 `alpha_init=0.05`）
- 那么在网络眼里，"保持动作随机性（高熵）的收益"将瞬间远超"减少物理延迟的收益"
- **结果：模型变成纯粹的随机发生器，到处乱搬家**

**当前代码现状**：

```python
# algorithms/hybrid_sac.py
alpha_init = 0.05
target_entropy = -np.log(1.0 / action_dim) * 0.90  # ≈ 0.99

# 自动 alpha 调整机制
log_alpha = torch.tensor(np.log(alpha_init), ...)
alpha_optimizer = optim.Adam([log_alpha], lr=learning_rate)
```

当前代码 **已有** 自动 alpha 调整（`log_alpha` + `alpha_optimizer`），但自动调整的速度可能跟不上 reward scaling 带来的天平倾斜。

**推荐修正**：

在引入 `REWARD_SCALE_MS` 的同时，必须同步等比例缩小 SAC 的初始 `alpha` 值：

```python
# 原始 reward ~= -5000, alpha_init = 0.05
# 缩放后 reward ~= -5,   alpha_init 应按比例缩小

REWARD_SCALE_MS = 1000.0
alpha_init = 0.05 / (REWARD_SCALE_MS / 1000.0)  # = 0.00005

# 或者更保守的做法：
alpha_init = 0.001  # 让物理奖励重新主导，再由自动调整慢慢找平衡
```

如果使用自动 alpha 调整机制，还需要考虑 `target_entropy` 的设置：

```python
# target_entropy 是策略熵的期望值，与 reward 尺度无关
# 但如果 alpha 被大幅压低，自动调整可能需要更长时间才能稳定
# 因此在 reward scaling 改动后，建议先跑小规模实验观察 alpha 的收敛曲线
```

**验证检查**：实现后打印 `alpha_history`，确认：
```text
Epoch 1: alpha=0.001, entropy=0.95, actor_loss=...
Epoch 3: alpha=0.002, entropy=0.90, ...  # alpha 应该稳定在一个小值
```

如果 `alpha` 快速膨胀（例如 > 0.1），说明熵收益仍然占主导，需要继续调低 `alpha_init` 或 `target_entropy`。

---

## 3. 核心原因二：SA 温度与毫秒级 cost 不匹配，导致基线过冷

### 3.1 当前代码

位置：`algorithms/sa.py`，`microservice_simulated_annealing(...)`。

默认参数：

```python
temp=100.0
cooling_rate=0.95
max_iter=30
```

邻域：

```python
node = random.choice(all_nodes)
other_servers = [s for s in candidate_server_ids if s != old_server]
new_server = random.choice(other_servers)
```

接受准则：

```python
delta = neighbor_cost - current_cost
if delta < 0:
    accept = True
else:
    accept = random.random() < math.exp(-delta / temp)
```

`current_cost = -current_reward`，而 `reward = -total_cost_ms`（带截断），所以 `delta` 是毫秒级 cost 差。

### 3.2 为什么 `temp=100` 很可能过冷

如果某个邻域迁移暂时增加 500 ms：

```text
P = exp(-500 / 100) = 0.0067
```

如果增加 2000 ms：

```text
P = exp(-2000 / 100) ≈ 2e-9
```

也就是说，SA 几乎只接受立即变好的动作，退火变成局部贪心。结合当前邻域只有 `k=3` 最近服务器中的单节点变化，SA 可能很难走出当前放置。

这与 `result.md` 中测试 Proactive DAG 表里 SA 的 `Total Migrated Nodes` 全为 0 是一致的。

### 3.3 建议不要只“拍脑袋改 5000”

`temp=5000` 在量级上合理，但最好增加轻量日志或返回统计后再标定。

建议先让 SA 返回或累计以下指标：

```python
sa_accept_count
sa_worse_accept_count
sa_neighbor_count
```

然后记录：

```text
accept_rate = accept_count / neighbor_count
worse_accept_rate = worse_accept_count / neighbor_count
```

目标不是越高越好。一个可用的经验范围：

- 初期总接受率：20% - 60%
- 更差解接受率：5% - 30%

### 3.4 推荐修改方案

第一轮可改为：

```python
def microservice_simulated_annealing(..., temp=5000.0, cooling_rate=0.97, max_iter=50, ...)
```

原因：

- `temp=5000` 与 `SLA_PENALTY_MS=5000` 同量级；
- `cooling_rate=0.97` 比 0.95 慢一些，避免 30 步内过快冻结；
- `max_iter=50` 对小 DAG 仍可接受，但会增加 SA wall time。

如果担心 SA 太慢，先只改：

```python
temp=3000.0
cooling_rate=0.97
max_iter=30
```

### 3.5 与 Hybrid SAC 的关系

Hybrid SAC 训练时会先调用 SA 得到 `sa_proposal`，Actor 的动作 1 是 `FOLLOW_SA`。因此 SA 不是单纯基线，它也是 SAC 的 teacher / prior。

如果 SA 太冷、经常不迁，SAC 的 `FOLLOW_SA` 先验质量会偏保守；SAC 可能转向动作 2 `NEAREST` 或在 Proactive 下学出更激进的替代策略。改善 SA 不是直接解决 SAC 过迁移，但能让 SAC 的动作空间参考更稳定。

---

## 4. 指标问题：DQN 的 `Avg Latency=0.00 ms` 是埋点缺失

### 4.1 当前代码

位置：`algorithms/dqn.py`。

DQN 每个触发步按拓扑序逐节点决策：

```python
if random.random() < epsilon:
    action = random.randint(0, 3)
else:
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = q_network(state_t)
        action = q_values.argmax().item()
```

返回结果中没有：

```python
total_decision_time
decision_count_for_latency
avg_decision_time_ms
```

所以 `run_comparison.py` 读不到该字段，表格用默认 0，显示为 `0.00 ms`。

### 4.2 建议修改

在 `run_dqn_microservice_fair` 初始化：

```python
total_decision_time = 0.0
decision_count_for_latency = 0
```

在每个触发决策、进入 `for ms_node in sorted_nodes` 前后包住整个 DQN 决策块：

```python
t_start = time.perf_counter()

for ms_node in sorted_nodes:
    ...
    if random.random() < epsilon:
        action = random.randint(0, 3)
    else:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = q_network(state_t)
            action = q_values.argmax().item()
    ...

t_end = time.perf_counter()
total_decision_time += (t_end - t_start)
decision_count_for_latency += 1
```

返回字典增加：

```python
"total_decision_time": total_decision_time,
"decision_count_for_latency": decision_count_for_latency,
"avg_decision_time_ms": (
    total_decision_time / decision_count_for_latency * 1000
    if decision_count_for_latency > 0 else 0
),
```

这不会改变算法效果，只修复报告可信度。

---

## 5. 额外但重要：当前 `Score` 必须从主指标中退场

当前报告使用：

```python
score = total_migrations + 0.5 * total_violations
```

这会让一次迁移的代价等价于两次违规的一半权重关系。与 reward 中 `SLA_PENALTY_MS=5000`、迁移按 ms 物理量计算的训练目标并不一致。

因此出现一种表面矛盾：

- Hybrid SAC 在 reward 中认为：用迁移减少 SLA / future penalty 可能是值得的；
- 报告 Score 却用 `M + 0.5V` 这个自创、无物理单位的式子强行给迁移计数和违规计数加权；
- 只要表格里继续叫它 **Score**，审稿人会自然把它当作最终评价标准，而不是一个辅助统计。

因此这里不能只“额外增加物理指标”，而应当 **替换或降级当前 Score**。

### 5.1 建议替换主 Score

```text
Avg total_cost_ms per decision
Avg migration_cost_ms
Avg sla_penalty_ms
Avg physical cost = total_cost_sum / decision_count
```

现有结果字典已有部分累计字段：

- `total_access_latency`
- `total_communication_cost`
- `total_migration_cost`
- `reward_history`

但缺少 `total_tearing_cost`、`total_sla_penalty_ms`、`total_cost_ms_sum`。建议在各算法中统一累计 `details` 里的这些项，再让 `result.md` 展示。

### 5.2 报告表格建议

将 `run_comparison.py` 的表头从：

```text
| Algorithm | Migrations | Violations | ... | Score |
```

改为：

```text
| Algorithm | Migrations | Violations | Avg Total Cost (ms) | Avg Migration Cost (ms) | SLA Penalty (ms) | Avg Latency (ms) |
```

如果仍想保留迁移/违规组合指标，应改名为 **Legacy Count Score**，并在报告中标注：

```text
Legacy Count Score = Migrations + 0.5 * Violations，仅用于与旧实验对照，不作为主评价。
```

### 5.3 可行的物理同态 Score

如果需要一个单列越小越好的综合数，建议使用与 reward 同源的：

```text
Avg Total Cost (ms) = total_cost_ms_sum / decision_count
```

或者在没有逐步累计 `total_cost_ms_sum` 前，先用近似：

```text
Physical Penalty ~= total_migration_cost + total_communication_cost + total_access_latency + total_sla_penalty_ms
```

但最终应以 `details["total_cost_ms"]` 的累计为准。

---

## 6. 阶段性修复实验原则（旧顺序已废弃）

早期建议曾把 DQN latency 放在第一步，因为它最安全。但经过架构复核，当前的核心风险不是“表格缺秒表”，而是：

1. 裁判口径不一致；
2. 物理成本量纲不一致；
3. 训练/推理状态不公平；
4. SAC 的 DAG 内 MDP 语义不严谨。

因此旧顺序 **“先修 DQN latency → 再修 reward → 最后调 SA”** 已废弃。新的总顺序以第 14 节为准：

```text
先统一物理裁判和主评价指标
→ 再修训练/推理防泄漏
→ 再修 SAC replay / MDP
→ 最后调 SA 与 DQN 展示细节
```

仍然保留的单项实验原则：

- `MB_TO_MBIT=8` 和 `BASE_MIGRATION_OVERHEAD_MS` 应分开做 ablation；
- 改 reward 后必须删除旧 SAC 权重重训；
- 改 trigger / metric 后必须重新生成 `result.md`；
- 改 SAC replay 后必须重新训练 checkpoint；
- DQN latency 可以独立修，但它不应再作为第一阶段主手术。

---

## 7. 当前最终判断

当前 `command.md` 原来的三条判断方向基本正确，但还不够。更新后的判断是：

1. **迁移单位漏 `*8` 很可能是 Hybrid SAC 过度迁移的关键原因之一**，适合修改；但应先做 `*8` ablation，再决定是否加 `100/200 ms` overhead。
2. **触发、reward、报告 Violations 必须使用同一套 SLA 函数**；否则 Proactive 迁移收益被扭曲。
3. **Proactive 不能继续依赖固定距离缓冲**，应改成预测轨迹越界 / TTV。
4. **当前 Score 必须从主指标中退场**，否则与 reward 目标冲突。
5. **Hybrid SAC 的 eval 状态重置和 replay MDP 语义必须修**；其中 replay 不能用 Contextual Bandit 降级作为正式方案。
6. **DQN checkpoint inference 与 latency 都要补**，但它们分别解决公平性和展示，不是 Hybrid 过迁移主因。
7. **SA 温度过低很可能是真的**，但应在物理裁判和 reward 修完后再用接受率统计标定。

---

## 8. 额外逻辑漏洞一：触发条件、reward 违规与报告违规口径不一致

这是比 DQN 计时更重要的指标逻辑问题，可能直接导致 Proactive 决策过多、迁移过多，但 `result.md` 中的 `Violations` 看起来没有同步对应。

### 8.1 当前代码口径

`core/context.py` 中 Reactive 触发条件是：

```python
spatial_violation = dist_km > DISTANCE_THRESHOLD_KM
qos_violation = calc_access_latency_ms(dist_km) > USER_SLA_TOLERANCE_MS
return spatial_violation or qos_violation
```

其中：

```python
DISTANCE_THRESHOLD_KM = 15.0
USER_SLA_TOLERANCE_MS = calc_access_latency_ms(DISTANCE_THRESHOLD_KM) * 0.99
```

由于 `calc_access_latency_ms(d) = d / 200 + 2`，15 km 对应约 `2.075 ms`，乘 `0.99` 后约 `2.054 ms`。因此 QoS 触发大约在：

```text
d / 200 + 2 > 2.054
d > 10.8 km
```

也就是说，**Reactive 触发阈值实际约为 10.8 km，而报告里的 real violation 仍只按 `gateway_dist > 15.0` 计数**。

`sa.py` / `dqn.py` / `hybrid_sac.py` 中报告违规计数都是类似：

```python
if gateway_dist > 15.0:
    total_violations += 1
```

`core/reward.py` 里的 `sla_penalty_ms` 又是：

```python
spatial_violation = max_entry_dist_km > SLA_DISTANCE_THRESHOLD
qos_violation = access_latency_ms > USER_SLA_TOLERANCE_MS
sla_penalty_ms = SLA_PENALTY_MS if (spatial_violation or qos_violation) else 0.0
```

所以当前存在三套相关但不完全一致的口径：

| 位置 | 口径 | 实际效果 |
|------|------|----------|
| `context.py` 触发 | `dist > 15km OR latency > 0.99 * latency(15km)` | 约 10.8 km 就会 Reactive |
| `reward.py` SLA penalty | 同上 | 约 10.8 km 就会扣 5000 |
| `result.md` Violations | 仅 `gateway_dist > 15km` | 只有超过 15 km 才算违规 |

### 8.2 造成的后果

- 算法会在 **10.8-15 km** 区间频繁触发并被 reward 认为已经 SLA penalty；
- 但 `result.md` 里这些状态 **不算 Violations**；
- Proactive 模式还会用 `PROACTIVE_WARNING_KM = 5.0` 提前触发，触发密度更高；
- 这会鼓励算法大量提前迁移，但报告上只能看到 Migrations 激增，Violations 下降幅度未必能解释迁移量。

这非常可能是当前 Hybrid SAC Proactive 迁移过多的原因之一。

### 8.3 修改建议

必须先统一指标口径，再调 reward。

可选方案 A：严格距离 SLA

```python
USER_SLA_TOLERANCE_MS = calc_access_latency_ms(DISTANCE_THRESHOLD_KM)
```

或直接在 trigger / reward / metric 中统一只用 `dist > 15.0` 作为 real SLA。

可选方案 B：保留 QoS SLA，但报告也按 QoS 统计

将三算法中的：

```python
if gateway_dist > 15.0:
    total_violations += 1
```

改为调用同一函数：

```python
from core.context import check_sla_violation

if check_sla_violation(current_lat, current_lon, gw_lat, gw_lon):
    total_violations += 1
```

推荐先采用 **方案 B**，因为它和当前 reward / trigger 更一致。

### 8.4 Proactive 触发不应再使用固定距离缓冲，应改为预测轨迹越界 / TTV

当前：

```python
PROACTIVE_WARNING_KM = 5.0
```

旧建议曾考虑把它改成：

```python
PROACTIVE_WARNING_KM = 12.0  # 或 0.8 * DISTANCE_THRESHOLD_KM
```

但这仍然是 **固定距离代理**，不是严格的前瞻触发。它有两个理论问题：

- 车辆停在缓冲区附近但速度为 0 时，会持续 Proactive 焦虑；
- 高速车辆可能很快越界，固定距离缓冲无法表达“多久后会违规”。

当前系统已经有 `predictor.predict_future(...)`，因此更合理的 Proactive 触发应基于预测轨迹是否会在 horizon 内触发 **同一套 SLA 判定**。

### 8.4.1 推荐触发语义

Reactive：

```python
if check_sla_violation(current_lat, current_lon, gw_lat, gw_lon):
    return TRIGGER_REACTIVE
```

Proactive：

```python
if proactive_enabled and predicted_locations:
    for step, (pred_lat, pred_lon) in enumerate(predicted_locations, start=1):
        if check_sla_violation(pred_lat, pred_lon, gw_lat, gw_lon):
            return TRIGGER_PROACTIVE
```

也就是：**只有预测器明确认为未来某一步会越过真实 SLA，才触发 Proactive**。

### 8.4.2 TTV（Time-to-Violation）必须考虑迁移耗时

如果未来需要区分“马上违规”和“很久以后违规”，可返回或记录：

```python
ttv_steps = first step where check_sla_violation(pred_lat, pred_lon, gw_lat, gw_lon)
```

进一步可转换为时间：

```python
ttv_seconds = ttv_steps * delta_t_future
```

但 Proactive 触发不能只写成静态：

```python
if ttv_steps <= PROACTIVE_TTV_STEPS:
    ...
```

因为微服务迁移本身需要时间。如果预测器显示 1.5 秒后违规，但当前 DAG 迁移预计需要 2.0 秒，那么此时再触发已经太晚，迁移途中仍会产生真实 SLA 违规。

因此真正的触发条件应是：

```text
TTV <= estimated_migration_time + safety_buffer
```

也就是：

```python
if ttv_seconds is not None and ttv_seconds <= estimated_migration_time_s + PROACTIVE_SAFETY_BUFFER_S:
    return TRIGGER_PROACTIVE
```

其中：

```python
PROACTIVE_SAFETY_BUFFER_S = 1.0  # 初始值，可做 ablation
```

`estimated_migration_time_s` 可以先用当前 DAG 的保守估计：

```python
estimated_migration_time_ms = estimate_dag_migration_ms(
    dag_info=dag_info,
    current_assignments=current_assignments,
    candidate_assignments=sa_proposal_or_nearest,
    trigger_type=TRIGGER_PROACTIVE,
)
estimated_migration_time_s = estimated_migration_time_ms / 1000.0
```

短期若不想引入完整候选放置估计，可用 DAG 上界：

```python
estimated_migration_time_ms = sum(
    ((image_mb + state_mb) * MB_TO_MBIT / effective_bandwidth) * 1000.0
    + BASE_MIGRATION_OVERHEAD_MS
    for node in dag_info["nodes"].values()
)
```

注意：这个估计必须和 `core/reward.py` 的迁移成本公式同源，否则 trigger 会再次和 reward 脱节。

这种动态安全线比 `PROACTIVE_WARNING_KM=12.0` 更贴近“按需前瞻”，也能减少因为静态距离缓冲造成的 ping-pong。

#### 8.4.2.1 预测器抖动（Predictor Jitter）与迁移锁定机制

> ⚠️ **架构复核补充**：动态 TTV 触发引入了新的工程陷阱——**预测器抖动**。

**问题场景**：

在真实车辆轨迹中，预测器的输出往往带有抖动（Jitter）：
- 前一秒预测 TTV = 1.5s（触发阈值内，触发了迁移）
- 下一秒车辆稍微减速，预测器改口说 TTV = 2.5s（超出触发阈值）

这种抖动会导致系统在"需要搬"和"不需要搬"之间疯狂横跳。如果每次触发都会中断并重启当前的 DAG 迁移任务，整个系统就会瘫痪。

**当前代码风险点**：

```python
# core/context.py - get_trigger_type()
if ttv_seconds is not None and ttv_seconds <= estimated_migration_time_s + PROACTIVE_SAFETY_BUFFER_S:
    return TRIGGER_PROACTIVE
# 如果下一个时间步 TTV 突然增大，就不再返回 PROACTIVE
# 但此时 DAG 迁移可能已经开始执行了一半
```

**推荐修正**：加入**迁移锁定（Migration Lock）** 或 **迟滞区间（Hysteresis）** 机制。

**方案 A：显式迁移锁定**

```python
# 在 hybrid_sac.py 或 run_comparison.py 的算法调用层维护状态
migration_lock = {}  # taxi_id -> lock_until_timestamp

def should_trigger_proactive(taxi_id, ttv_seconds, current_timestamp):
    # 如果当前 taxi 正在迁移锁定期内，忽略所有触发
    if taxi_id in migration_lock and current_timestamp < migration_lock[taxi_id]:
        return False
    
    if ttv_seconds is not None and ttv_seconds <= estimated_migration_time_s + PROACTIVE_SAFETY_BUFFER_S:
        # 触发迁移，并设置锁定期
        migration_lock[taxi_id] = current_timestamp + estimated_migration_time_s + LOCK_BUFFER_S
        return True
    
    return False

LOCK_BUFFER_S = 2.0  # 迁移完成后额外锁定 2 秒，防止立即反复触发
```

**方案 B：TTV 迟滞区间**

```python
# 使用两个阈值：进入阈值和退出阈值
TTV_ENTER_THRESHOLD_S = estimated_migration_time_s + 1.0   # 触发迁移
TTV_EXIT_THRESHOLD_S = estimated_migration_time_s + 3.0    # 解除迁移状态

# 只有当 TTV 从"安全区"跌入"危险区"时才触发
# 一旦触发，必须等 TTV 回升到"安全区"才能重新判断
```

**验证检查**：在 debug 日志中打印迁移触发事件，确认：
```text
Taxi 42: TTV=1.5s, triggered PROACTIVE, lock_until=t+3.0s
Taxi 42: TTV=2.5s, still in lock period, ignored
Taxi 42: TTV=4.0s, lock expired, normal monitoring resumed
```

如果看到同一辆 taxi 在短时间内反复触发/取消触发，说明迁移锁定机制没有正确工作。

#### 8.4.2.2 致命陷阱：连续物理缓冲与离散采样步长的"错位"

> 🚨 **致命理论陷阱**：上述 `LOCK_BUFFER_S = 2.0s` 的设计在**离散事件仿真（Discrete-Event Simulation）**中毫无意义！

**问题剖析**：

回顾 `core/data_loader.py` 中的真实 GPS 轨迹数据——采样频率通常是 **10 秒、15 秒甚至 30 秒一个点**！

如果给系统加了一个 `2.0` 秒的锁，但仿真环境的主循环是每隔 `15` 秒才 tick（推进一步）一次，那么：

```text
仿真步 T=0:   触发 Proactive，锁定 2.0s → lock_release = T+2s
仿真步 T=15:  此时 lock 已经"默默过期" 13 秒，无法拦截这一步的预测器抖动！
```

这个 `2.0` 秒的锁会在**两次仿真步之间"默默过期"**，根本无法拦截下一次环境 tick 带来的预测器抖动。

**推荐修正**：迁移锁定必须与**环境的数据采样步长（Time Steps）**对齐

**方案 A（推荐）：步数锁定**

更稳健的做法是，一旦触发迁移，强制锁定未来的 `K` 个仿真步（Steps）：

```python
# 使用仿真步数而非绝对秒数
LOCK_STEPS = 2  # 锁定未来 2 个仿真步

migration_lock = {}  # taxi_id -> (lock_release_step, lock_release_timestamp)

def should_trigger_proactive(taxi_id, ttv_seconds, current_step, current_timestamp):
    # 步数锁定检查
    if taxi_id in migration_lock:
        lock_release_step, lock_release_ts = migration_lock[taxi_id]
        if current_step < lock_release_step:
            return False  # 仍在锁定步数内
    
    if ttv_seconds is not None and ttv_seconds <= estimated_migration_time_s + PROACTIVE_SAFETY_BUFFER_S:
        # 触发迁移，设置步数锁定
        migration_lock[taxi_id] = (current_step + LOCK_STEPS, current_timestamp + estimated_migration_time_s)
        return True
    
    return False
```

**方案 B：时间戳对齐**

如果必须使用时间锁定，锁定解除条件必须是：

```python
# 确保 lock_release_timestamp 至少跨越了当前仿真步的时间间隔
LOCK_BUFFER_S = max(2.0, avg_sample_interval_s)  # 例如 max(2.0, 15.0) = 15.0s

# 或者动态计算：
LOCK_BUFFER_S = estimated_migration_time_s + current_sample_interval_s
```

解锁条件：
```python
if current_trajectory_timestamp > lock_release_timestamp:
    # 解锁
```

**验证检查**：实现后打印锁定状态，确认：
```text
Step 100 (T=1500s): Taxi 42 triggered PROACTIVE, lock until step 102
Step 101 (T=1515s): Taxi 42 still locked (step < 102), trigger ignored
Step 102 (T=1530s): Taxi 42 lock expired, monitoring resumed
```

如果锁定在同一仿真步内就过期了，说明锁定机制没有正确对齐仿真步长。

### 8.4.3 对 `future_delay_ms` 的同步要求

`reward.py` 里的 `FUTURE_DIST_THRESHOLD` 当前也是固定距离阈值。若 trigger 改成预测轨迹越界，建议同步把 `future_delay_ms` 改为基于同一 SLA 函数的未来违规程度或 TTV penalty，否则 trigger 与 reward 仍会出现语义割裂。

---

## 9. 额外逻辑漏洞二：Hybrid SAC 训练末 eval 没有重置环境状态

`hybrid_sac.py` 中 `taxi_dag_type` 和 `taxi_dag_assignments` 在 epoch 循环外初始化：

```python
taxi_dag_type = {}
taxi_dag_assignments = {}
...
for epoch in range(num_epochs):
    taxi_last = {}
```

进入最后一轮 eval 时只重置了指标：

```python
if is_eval_epoch:
    total_violations = 0
    total_migrations = 0
    ...
```

但是 **没有重置 `taxi_dag_assignments` / `taxi_dag_type`**。因此训练阶段 `result.md` 中 Hybrid SAC 的 headline 指标来自“最后 eval epoch”，但这个 eval 的初始环境不是从 nearest 初始化开始，而是继承了前 5 个训练 epoch 改过的所有 taxi DAG 放置。

### 9.1 造成的后果

- 训练段 Hybrid SAC 的 eval 指标不等价于“在 train_df 上从统一初始状态评估策略”；
- 它评估的是一个已经被训练过程多次迁移过的环境终态；
- 测试段 inference 是新函数调用，`taxi_dag_assignments` 重新为空，从 nearest 初始化开始；
- 因此 `result.md` 的“训练段 → 测试段”泛化对比混入了 **环境初始状态差异**。

这会污染：

```text
Hybrid SAC 泛化对比（训练 → 测试）
```

尤其是 Proactive / Reactive Migrations 的对比。

### 9.2 修改建议

在进入 eval epoch 时重置仿真环境状态：

```python
if is_eval_epoch:
    taxi_dag_type = {}
    taxi_dag_assignments = {}
    taxi_last = {}
    # 再重置指标
```

更规范的是封装一个 `reset_simulation_state()`，确保 train eval 和 inference 都从相同规则的初始状态开始。

注意：这会改变训练段 Hybrid SAC 的 headline 指标，但更公平。

---

## 10. 额外逻辑漏洞三：Hybrid SAC replay 中 next state / done 语义不严谨

当前 Hybrid SAC 在一次 DAG 决策中，按拓扑序对多个节点依次动作。执行完所有节点后才计算一次总 reward：

```python
reward, details = calculate_microservice_reward(...)
per_node_reward = reward / num_nodes
```

随后对每个节点 transition 存入 replay：

```python
memory.append((
    graph_state,
    node_idx,
    action,
    per_node_reward,
    next_graph_state,
    is_last,
    mask_cpu,
))
```

其中：

- 每个节点共享同一个 `graph_state`；
- 每个节点也共享同一个 `next_graph_state`，它是整张 DAG 所有动作执行完之后的状态；
- 只有最后一个节点 `done=True`；
- `optimize_sac` 中非终止 transition 的 target 又用同一个 `node_idx` 去取 `next_embeddings[node_idx]`。

这并不严格对应拓扑序决策的 MDP。

### 10.1 问题在哪里

如果把 DAG 内节点决策看成一个序列：

```text
node_1 action -> state_2
node_2 action -> state_3
...
node_n action -> final_state -> reward
```

那么第 i 个节点的 next state 应该是“执行第 i 个动作后、准备决策第 i+1 个节点”的状态，并且 next node index 应该是 `node_{i+1}`。

但当前实现对所有非最后节点都给了：

```text
next_graph_state = final_state
next_node_idx = same node_idx
done = False
```

这会让 critic 学到不准确的 bootstrap 目标，可能导致 Q 估计不稳定，从而进一步放大过迁移或策略偏置。

### 10.2 不建议采用 Contextual Bandit 降级方案

旧建议中曾提出把 DAG 内所有节点 transition 都设为 `done=True`，作为 terminal contextual bandit 处理。经过架构复核，不建议采用。

原因：

- 这会让 Critic 的 target 丢弃未来价值项，前序节点无法学习“我放在这里会给后续节点通信/tearing 带来什么影响”；
- DAG 拓扑依赖会被削弱，SAC 退化成逐节点短视贪心；
- 论文审查中容易被指出“声称图拓扑协同，但训练目标是独立 bandit”的理论硬伤。

因此，**不要把 10.2 作为正式修复方案**。它最多只能作为临时 debug 对照实验，用来判断错误 bootstrap 是否导致训练不稳定，不能作为最终实现。

### 10.3 推荐方案 A：完整自回归 MDP（保留当前逐节点 Actor）

完整做法是存储每一步执行后的中间 `graph_state_i`、`next_node_idx`：

```text
(graph_state_i, node_i, action_i, reward_i, graph_state_{i+1}, node_{i+1}, done_i)
```

并在 `optimize_sac` 中用 `next_node_idx` 取下一节点 embedding。

当前实现需要的关键改动：

1. 在 DAG 拓扑序决策时，每执行一个节点动作后，立即构造一个中间 `graph_state_next_i`；
2. replay 中额外保存 `next_node_idx`；
3. `optimize_sac` 对非终止项使用 `next_node_idx` 而不是当前 `node_idx`；
4. reward 可以继续用最终总 reward 的延迟分配，但建议至少只在最后一步给总 reward，或使用 potential-based shaping 分摊，不要简单 `reward / num_nodes` 后又 bootstrap 多步。

#### 10.3.1 信用分配（Credit Assignment）的稀疏奖励修正

> ⚠️ **架构复核补充**：当前代码使用 `per_node_reward = reward / num_nodes` 平均分配，存在严重的因果倒置问题。

**问题示例**：
- Step 1：Actor 把"前端节点"放到了最优位置（完美决策）
- Step 3：Actor 把"数据库节点"放错位置，导致巨大的通信惩罚
- 平均分配后，网络会错误地认为 Step 1 的动作也很糟糕

**推荐修正**：采用 **稀疏奖励（Sparse Reward）** 方案：

```python
for i, (node_idx, action) in enumerate(node_transitions):
    is_last = (i == len(node_transitions) - 1)
    
    # 稀疏奖励：只有最后一个节点获得真实 reward
    if is_last:
        reward_i = reward  # 完整的 total_cost_ms 计算结果
    else:
        reward_i = 0.0     # 中间节点强制为 0
    
    memory.append((
        graph_state_i,      # 当前节点决策前的状态
        node_idx,
        action,
        reward_i,           # 稀疏奖励
        graph_state_next_i, # 当前节点决策后的状态
        next_node_idx,      # 下一个待决策的节点
        is_last,
        mask_cpu,
    ))
```

**理论依据**：

- DAG 内部节点决策之间没有物理时间流逝，因此 `gamma` 在 DAG 内部应视为 1.0
- SAC 的 Critic 网络会通过 Bootstrap（TD 反向传播）正确地将最终惩罚回传给前序节点的决策
- 这种方式让网络自动学习"哪个节点的决策导致了最终的好/坏结果"

**验证检查**：实现后打印 replay 中的 reward 分布，确认：
```text
DAG transition 1: reward_i=0.0, is_last=False
DAG transition 2: reward_i=0.0, is_last=False
DAG transition 3: reward_i=-2345.6, is_last=True
```

#### 10.3.1.1 致命陷阱：自回归 MDP 中的"时间扭曲"与 Gamma 衰减

> 🚨 **致命理论陷阱**：稀疏奖励方案如果不配合 Gamma 掩码，会产生"时间扭曲"效应。

**问题剖析**：

在标准 SAC 算法中，Critic 网络的目标值计算公式是：

$$Q_{target} = r + \gamma \times (1 - \text{done}) \times Q_{next}$$

DAG 内部的拓扑决策是**瞬间完成的（物理时间没有流逝）**。如果在 DAG 内部的 Transition 中使用了环境默认的 $\gamma$（如 $0.99$），灾难就会发生：

**具体示例**（假设 DAG 有 5 个节点，最终真实惩罚是 -5000）：
- 节点 5 看见的 Q 值是 `-5000`
- 节点 4 看见的 Q 值是 `0.99 × -5000 = -4950`
- 节点 3 看见的 Q 值是 `0.99² × -5000 = -4900.5`
- 节点 1 看见的 Q 值是 `0.99⁴ × -5000 ≈ -4802`

网络会产生一种**极度危险的错觉**：
> "处于拓扑前端的节点，犯错的代价比后端节点小"

这不仅违背物理常识，还会**彻底破坏 GAT 对拓扑结构的公平学习**！

**推荐修正**：引入 **Gamma 掩码（Gamma Masking）**

在 `optimize_sac` 计算 Target Q 时，必须区分 DAG 内部 Transition 和跨物理时间步的 Transition：

```python
# algorithms/hybrid_sac.py - optimize_sac()

# 从 replay buffer 取出 transition 时，额外传入 is_intra_dag 标志
# 或者直接利用现有的 done 标志来推断

for transition in batch:
    graph_state, node_idx, action, reward, next_graph_state, done, mask = transition
    
    # Gamma 掩码：DAG 内部 transition (done=False) 使用 gamma=1.0
    # 跨物理时间步的 transition (done=True 或最后节点) 使用原始 gamma
    if done:
        # 最后一个节点，使用原始 gamma（或直接 target = reward）
        effective_gamma = gamma  # 实际上 done=True 时 target_value = reward
    else:
        # DAG 内部节点，强制 gamma=1.0（无时间折扣）
        effective_gamma = 1.0
    
    # 计算 target Q
    with torch.no_grad():
        if done:
            target_value = reward
        else:
            next_value = compute_soft_value(next_graph_state, next_node_idx)
            target_value = reward + effective_gamma * next_value
```

**更简洁的实现方式**：

由于 DAG 内部 `reward_i = 0`（稀疏奖励），且我们希望 `gamma = 1.0`，实际上：

```python
# DAG 内部 transition 的 target:
# target_value = 0 + 1.0 * Q_next = Q_next

# 最后节点 transition 的 target:
# target_value = final_reward (done=True, no next value)
```

这意味着可以直接在代码中写：

```python
effective_gamma = 1.0 if not done else gamma
# 或者更直接：
effective_gamma = 1.0  # DAG 内部全部用 1.0，因为 done=True 时反正不用 gamma
```

**验证检查**：实现后抽样打印 Target Q 值，确认：
```text
DAG node 1: target_Q ≈ -5000 (not -4802)
DAG node 2: target_Q ≈ -5000 (not -4851)
DAG node 5 (last): target_Q = -5000 (reward itself)
```

如果前端节点的 Target Q 明显小于最终惩罚的绝对值，说明 Gamma 掩码没有正确工作。

这里还有一个必须提前规避的工程陷阱：**状态幻觉**。

如果在 DAG 内部循环中虽然记录了 `graph_state_{i+1}`，但底层 `taxi_dag_assignments[taxi_id]` 或观测向量没有实时反映 `action_i` 的落子，那么 `graph_state_{i+1}` 仍然是假的。比如节点 1 已经被动作迁移到基站 B，那么节点 2 决策时看到的输入里，节点 1 的 placement / server 特征必须已经是 B，否则自回归 MDP 只是形式正确、语义错误。

因此 Step 3.2 的前置约束是：

```text
apply action_i 后，必须立即刷新环境状态和 graph_state，再进入 node_{i+1}
```

当前代码中动作执行已经会立即写：

```python
taxi_dag_assignments[taxi_id][ms_node] = target_server
```

但现有训练逻辑只在整张 DAG 决策前构造一次 `graph_state`，整张 DAG 决策后构造一次 `next_graph_state`。完整修复时必须把 `build_graph_state(...)` 移入节点循环，或提供一个轻量的 `update_graph_state_after_action(...)`，确保中间状态真实更新。

示意：

```text
for i, node_i in enumerate(sorted_nodes):
    graph_state_i = state before action_i
    action_i = actor(...)
    apply action_i
    refresh observation so node_i placement is visible
    graph_state_next_i = state after action_i
    next_node_idx = node_{i+1} if exists else None
    done_i = (i == num_nodes - 1)
    reward_i = 0 for i < last, final_reward for last
```

这种方案最符合当前 Actor 结构，因为现有 Actor 本来就是 **逐节点输入、逐节点输出**。

测试要求：实现后抽样打印 replay 中一条 DAG 序列，人工核对：

```text
action_i 将 node_i 放到 server B
graph_state_{i+1} 中 node_i 的 placement / sa_prior / 相关通信特征已反映 server B
node_{i+1} 的 Actor 输入基于更新后的 graph_state_{i+1}
```

如果这个检查不通过，就不能声称已修复自回归 MDP。

### 10.4 推荐方案 B：宏观 DAG Transition（需要网络结构调整）

如果不想维护 DAG 内自回归 MDP，也可以把一次 DAG 决策看成一个宏观动作：

```text
State: 决策前 graph_state
Action: 整个 DAG 的动作向量，例如 [STAY, NEAREST, FOLLOW_SA]
Reward: 总 reward
Next State: 所有节点执行后的 final graph_state
Done: 根据真实 episode 语义确定
```

优点：

- reward 与动作向量一一对应；
- 保留 DAG 全局决策视野；
- 不需要伪造 per-node bootstrap。

缺点：

- 当前 `SACDiscreteActor` 只输出单节点 3 类动作 logits，不能直接表示动作向量；
- 需要改成多头 Actor，或自回归策略网络输出整条动作序列的 log probability；
- critic 也要支持 `(graph_state, action_vector)`，改动明显更大。

因此在当前代码基础上，**优先选择方案 A：完整自回归 MDP**。宏观 Transition 更适合作为下一版架构重构。

---

## 11. 额外逻辑漏洞四：DQN 在“推理阶段”其实仍在测试集上在线训练

`run_comparison.py` 的 `run_inference_phase` 中：

```python
proactive_results["DQN"] = run_dqn_microservice_fair(df, servers_df, predictor=predictor, proactive=True)
```

但 `run_dqn_microservice_fair` 每次调用都会：

```python
q_network = MicroserviceDQN(...)
target_network = MicroserviceDQN(...)
optimizer = optim.Adam(...)
...
loss_val = optimize_model(...)
epsilon *= epsilon_decay
```

也就是说，所谓 inference phase 对 DQN 来说不是“加载训练好的 DQN 在 test_df 上评估”，而是 **在 test_df 上重新初始化并在线学习**。

### 11.1 后果

- Hybrid SAC 是 train_df 训练、保存 checkpoint、test_df 加载评估；
- DQN 是 test_df 上从随机网络开始边跑边学；
- SA 是无训练算法；
- 三者在测试段并不是完全同类的“推理”定义。

这会影响 `result.md` 中测试段 DQN 与 Hybrid SAC 的比较可信度。

### 11.2 修改建议

给 DQN 加与 SAC 类似的模式：

```python
run_dqn_microservice_fair(
    df,
    servers_df,
    predictor=None,
    proactive=False,
    inference_mode=False,
    checkpoint_path=None,
    save_checkpoint_path=None,
)
```

训练阶段保存：

```python
torch.save(q_network.state_dict(), "checkpoints/dqn_proactive.pth")
```

推理阶段加载：

```python
q_network.load_state_dict(torch.load(checkpoint_path, map_location=device))
epsilon = 0.0
skip optimize_model(...)
```

如果短期不想实现 DQN checkpoint，应在 `result.md` 中明确标注：

```text
DQN test results are online-learning results on test_df, not checkpoint inference.
```

---

## 12. 额外逻辑漏洞五：Hybrid SAC 的 eval tie-break 可能引入 FOLLOW_SA 偏置

当前 `get_action_deterministic`：

```python
tie_break = torch.tensor([0.0, 1e-5, 0.5e-5])
scores = logits + tie_break
idx = scores.argmax(dim=-1)
```

这个修复解决了早期 `argmax(softmax)` 恒选 0 的死锁，但也带来一个副作用：当三个 logits 非常接近时，eval 会偏向动作 1，即 `FOLLOW_SA`。

如果 SA proposal 在某些场景中倾向迁移，或者 reward 迁移成本偏低，那么这个 tie-break 会强化迁移。

### 修改建议

tie-break 可以保留，但建议只在 logits 几乎完全相等时生效，而不是无条件加：

```python
if (logits.max(dim=-1).values - logits.min(dim=-1).values) < 1e-7:
    scores = logits + tie_break
else:
    scores = logits
```

或者将 tie-break 偏向改为更保守的顺序：

```python
STAY > FOLLOW_SA > NEAREST
```

但这可能重新带来少迁移倾向。更好的做法是：先修 reward 和触发口径，再判断是否还需要改 tie-break。

---

## 13. 额外逻辑漏洞六：processed CSV 捷径会绕过 `ACTIVE_USERS_LIMIT` 等参数

`core/data_loader.py`：

```python
if processed_csv is not False and processed_csv is not None:
    proc = os.path.abspath(processed_csv)
    if os.path.isfile(proc):
        return _load_from_processed_csv(proc, start_index, end_index, chunk_size)
```

这意味着只要传入的 processed CSV 存在，就不会重新执行：

- `active_users_limit`
- `min_vehicle_points`
- raw 清洗
- Top-N 筛选

当前正式实验使用的 processed 文件本身就是 active100，因此 `result.md` 没问题。但 smoke test 中曾把 `ACTIVE_USERS_LIMIT=5`，实际仍加载 active100，这说明这个捷径很容易造成误判。

### 修改建议

短期：在 `run_comparison.py` 的报告中写出实际加载行数与 taxi 数，不只写配置值。

长期：让 processed 文件名和参数绑定，或在 `_load_from_processed_csv` 后二次校验：

```python
if active_users_limit is not None:
    assert df["taxi_id"].nunique() <= active_users_limit
```

或对 smoke test 强制：

```python
processed_csv=False
```

---

## 14. 更新后的架构级手术顺序

综合最新架构复核意见，优先级再次调整。新的原则是：**先清理评估环境和防泄漏，再修 SAC 的 MDP 表达，随后重置物理 reward 与前瞻触发，最后校准基线**。

原因：如果先调 reward，再重构 SAC replay / MDP，那么前面调好的 reward-策略平衡很可能在 MDP 改动后全部作废。因此 reward 和 trigger 应放在 MDP 结构稳定之后。

| 优先级 | 问题 | 为什么优先 |
|--------|------|------------|
| P0 | Hybrid SAC eval 前重置环境状态 | 先保证训练末 eval 与 test inference 起点公平 |
| P0 | DQN checkpoint inference + 禁止 test 在线训练 | 先堵住测试集泄漏，保证算法对比口径一致 |
| P0 | 统一 SLA 判定函数与 `result.md` 主指标 | 先统一裁判；主表改用 Avg Total Cost，旧 Score 降级 |
| P1 | 修 Hybrid SAC replay 的完整自回归 MDP | 先修策略学习的状态转移语义，避免 reward 调参作废 |
| P1 | 自回归 MDP 状态实时刷新检查 | 防止记录了 `state_{i+1}` 但观测没有反映 `action_i` 的状态幻觉 |
| P1 | **DAG 内稀疏奖励（Sparse Reward）** | 解决信用分配错位，只在最后节点给完整 reward |
| P1 | **DAG 内 Gamma 掩码（Gamma Masking）** | 防止"时间扭曲"，DAG 内部 transition 强制 γ=1.0 |
| P2 | 修 `migration_delay_ms` 的 `MB_TO_MBIT=8` + overhead | MDP 稳定后重置物理成本 |
| P2 | 引入 reward scaling + **同步调整 alpha_init** | 防止 Q target 过大 + 防止熵坍塌 |
| P2 | Proactive 改为动态 TTV：TTV <= 迁移耗时 + buffer | 触发机制必须考虑迁移本身耗时 |
| P2 | **加入迁移锁定（Migration Lock）机制** | 防止预测器抖动导致的迁移死锁 |
| P2 | **迁移锁定使用步数而非绝对秒数** | 防止锁定在离散仿真步之间"默默过期" |
| P3 | SA 温度 + 接受率统计 | 物理成本稳定后再校准基线 |
| P3 | DQN latency 埋点 | 修报告展示 |
| P3 | tie-break 条件化 / processed CSV 参数校验 | 防止后续 smoke 和边界行为误判 |

推荐新的“稳妥手术顺序”：

```text
第一阶段：基础设施与防泄漏（清理考场，不动核心学习逻辑）
  A1. Hybrid SAC 最后一轮 eval 前清空 taxi_dag_assignments / taxi_dag_type / taxi_last。
  A2. DQN 增加 save/load checkpoint；推理阶段 epsilon=0，禁止 optimize_model。
  A3. DQN 增加 time.perf_counter() 延迟埋点。
  A4. 统一 result.md 的主指标：废除主表 Score，改为 Avg Total Cost (ms)。
  A5. 统一 Violations 使用 check_sla_violation 或明确单一 SLA 判定。

第二阶段：心智模型重构（修 MDP，最硬核）
  B1. 不采用 Contextual Bandit 降级。
  B2. 实现 DAG 拓扑序的完整自回归 transition (state_i, action_i, next_state_i, reward_i, done_i)。
  B3. 每次 apply action_i 后实时刷新 taxi_dag_assignments 和 graph_state_i+1。
  B4. 抽样打印 replay，人工核对 state_{i+1} 是否真实反映 action_i。
  B5. [新增] 采用稀疏奖励：中间节点 reward_i=0，只有 last_node 获得完整 reward。
      —— 解决信用分配错位，让 Critic 通过 Bootstrap 正确回传惩罚。
  B6. [致命陷阱] Gamma 掩码（Gamma Masking）：DAG 内部 transition 强制 γ=1.0。
      —— 防止"时间扭曲"：DAG 内部无物理时间流逝，使用 γ=0.99 会导致前端节点惩罚被错误衰减。
      —— 实现：if not done: effective_gamma = 1.0
      —— 验证：抽样打印 Target Q，确认节点 1 的 target_Q ≈ -5000 而非 ≈ -4802。

第三阶段：物理法则重置（修 Reward 与触发）
  C1. migration_delay_ms 加 MB_TO_MBIT=8 和基础迁移 overhead。
  C2. reward 改为 -total_cost_ms / REWARD_SCALE_MS，clip 保持在 [-10, 0]。
  C2.1 [新增] 同步调整 SAC 的 alpha_init：按 REWARD_SCALE_MS 等比例缩小。
       —— 当前 alpha_init=0.05，若 REWARD_SCALE_MS=1000，建议降至 0.001 或更低。
       —— 防止熵坍塌：reward 缩小后熵收益占主导，模型变成随机发生器。
  C3. Proactive 改为动态 TTV：TTV <= estimated_migration_time + safety_buffer。
  C3.1 [新增] 加入迁移锁定（Migration Lock）或迟滞区间（Hysteresis）。
       —— 一旦触发 Proactive 迁移，在 DAG 迁移完成前忽略预测器抖动。
  C3.2 [致命陷阱] 迁移锁定必须使用**步数锁定**而非绝对秒数！
       —— 问题：真实 GPS 采样频率 10~30s/点，2.0s 的锁会在仿真步之间"默默过期"。
       —— 推荐：LOCK_STEPS = 2（锁定未来 2 个仿真步），而非 LOCK_BUFFER_S = 2.0s。
       —— 验证：确认锁定跨越至少一个完整的仿真步，而非在同一步内过期。
  C4. 跑极小规模 smoke，确认：
      - reward 没有大量顶到 clip
      - critic loss 不出现 NaN
      - alpha 没有快速膨胀（应稳定在小值如 0.001~0.01）
      - 同一 taxi 没有在短时间内反复触发/取消触发

第四阶段：基线校准（唤醒 SA）
  D1. 给 SA 增加接受率 / worse_accept_rate 统计。
  D2. 在新物理成本稳定后，再调 temp=3000~5000、cooling_rate、max_iter。
  D3. 视最终迁移倾向，再判断是否需要条件化 SAC tie-break。
```

这些改动中，第二和第三阶段都会改变核心实验结果；每次进入下一阶段的正式实验前都必须删除旧权重并重新运行：

```powershell
python run_comparison.py --pipeline
```