# 🔬 系统审计报告 (System Audit Report)

**项目名称**: Migrate-main — 基于 Trigger-Aware GAT + Discrete SAC 的边缘微服务迁移系统  
**审计日期**: 2026-04-26  
**审计版本**: v3.7 (JIT Migration + 3D Trigger Context)  
**审计模式**: 只读白盒审计 (Read-Only White-Box Audit)

---

## 📋 目录

1. [物理与环境常量 (Environment & Simulation)](#1-物理与环境常量-environment--simulation)
2. [SA 基线算法配置 (Baseline SA Configuration)](#2-sa-基线算法配置-baseline-sa-configuration)
3. [Hybrid SAC 与 GAT 超参数 (RL Hyperparameters)](#3-hybrid-sac-与-gat-超参数-rl-hyperparameters)
4. [奖励函数惩罚系数 (Reward Multipliers)](#4-奖励函数惩罚系数-reward-multipliers)
5. [公平性分析 (Fairness Analysis)](#5-公平性分析-fairness-analysis)

---

## 1. 物理与环境常量 (Environment & Simulation)

### 1.1 轨迹预测视界 (Forecast Horizon)

| 参数 | 值 | 定义位置 |
|------|-----|---------|
| `FORECAST_HORIZON` | **15 步** | `algorithms/sa.py:19`, `algorithms/hybrid_sac.py:44` |

**说明**: SA 和 Hybrid SAC 使用相同的 15 步预测视界，确保公平对比。

### 1.2 SLA 违规距离阈值

| 参数 | 值 | 定义位置 | 用途 |
|------|-----|---------|------|
| `SLA_DISTANCE_THRESHOLD` | **15.0 km** | `core/reward.py:43` | 奖励函数中的 SLA 死亡惩罚阈值 |
| `DISTANCE_THRESHOLD_KM` | **15.0 km** | `core/context.py:10` | REACTIVE 触发器判断阈值 |
| `SLA_DISTANCE_THRESHOLD_KM` | **15.0 km** | `core/state_builder.py:34` | risk_ratio 计算用阈值 |
| `FUTURE_DIST_THRESHOLD` | **15.0 km** | `core/reward.py:14` | 未来违规惩罚阈值 |

**一致性检查**: ✅ 所有 SLA 阈值均为 **15.0 km**，保持内部一致。

### 1.3 Proactive 预警阈值

| 参数 | 值 | 定义位置 |
|------|-----|---------|
| `PROACTIVE_WARNING_KM` | **5.0 km** | `core/context.py:18` |

**物理含义**: 当预测轨迹上任意点到 Gateway 距离超过 5.0 km 时，触发 PROACTIVE 迁移决策。这创建了一个 **10 km 的缓冲区** (5-15 km)，用于生成丰富的 PROACTIVE 样本。

### 1.4 微服务 DAG 拓扑配置

| DAG 类型 | 概率 | 节点数 | 有状态节点 | 最大镜像 (MB) | 最大状态 (MB) |
|----------|------|--------|-----------|--------------|--------------|
| **Data_Heavy_DAG** | 40% | 6 | 1 (MS_37374) | 150 | 256 |
| **Compute_Heavy_DAG** | 35% | 4 | 0 | 200 | 0 |
| **IoT_Lightweight_DAG** | 25% | 3 | 3 (全部) | 50 | 10 |

**来源**: `core/microservice_dags.py`

**DAG 拓扑详细结构**:

```
Data_Heavy_DAG (6节点, 7边):
  USER ─617→ UNKNOWN ─7→ MS_4281 ─14→ MS_8099 ─8→ MS_37295
         └─12→ MS_8099 ──────────────────┼──13937→ MS_37374 [STATEFUL]
                   ↑────4587─────────────┘

Compute_Heavy_DAG (4节点, 3边):
  MS_10097 ─2→ UNKNOWN ─3→ MS_10370
                │
                └─22707→ MS_50265

IoT_Lightweight_DAG (3节点, 2边):
  MS_41612 [S] ─1→ MS_10041 [S] ─2→ MS_27421 [S]
  [S] = Stateful
```

### 1.5 候选服务器数量

| 参数 | 值 | 定义位置 |
|------|-----|---------|
| k-近邻服务器数量 | **3** | `algorithms/sa.py:199`, `algorithms/hybrid_sac.py:1359` |

**说明**: SA 和 SAC 均使用相同的 k=3 近邻服务器作为候选迁移目标。

### 1.6 地理计算常量

| 参数 | 值 | 定义位置 |
|------|-----|---------|
| 地球半径 | **6371 km** | `core/geo.py:6` |
| 距离计算方法 | **Haversine 公式** | `core/geo.py:4-12` |

---

## 2. SA 基线算法配置 (Baseline SA Configuration)

### 2.1 模拟退火核心参数

| 参数 | 默认值 | 定义位置 |
|------|--------|---------|
| `temp` (初始温度) | **100.0** | `algorithms/sa.py:26` |
| `cooling_rate` (冷却系数) | **0.95** | `algorithms/sa.py:26` |
| `max_iter` (最大迭代次数) | **30** | `algorithms/sa.py:26` |

### 2.2 算力限制分析

**迭代次数计算**:
- 单次决策最大迭代: **30 次**
- 每次迭代操作: 随机选择一个节点 + 随机选择一个候选服务器 + 评估奖励
- 接受/拒绝准则: Metropolis-Hastings 概率 `exp(-delta/temp)`

**温度衰减轨迹**:
```
iter  0: temp = 100.00
iter 10: temp =  59.87
iter 20: temp =  35.85
iter 29: temp =  22.89
```

**时延瓶颈分析**:
SA 的 4.04ms 平均决策时延主要由以下因素构成:
1. 30 次迭代的邻域搜索
2. 每次迭代调用 `calculate_microservice_reward()` 计算完整奖励
3. 候选服务器距离计算 (Haversine)

**结论**: SA 的搜索深度 (`max_iter=30`) 是人为限制的，属于**合理的 real-time 约束**。增加迭代次数可能提升解质量但会增加延迟。

---

## 3. Hybrid SAC 与 GAT 超参数 (RL Hyperparameters)

### 3.1 网络架构参数

#### TriggerAwareGAT (图注意力网络)

| 参数 | 值 | 定义位置 |
|------|-----|---------|
| `node_feat_dim` | **3** | `hybrid_sac.py:1167` |
| `trigger_dim` | **3** (v3.7) | `hybrid_sac.py:1169` |
| `hidden_dim` | **64** | `hybrid_sac.py:1145`, `1170` |
| `num_heads` | **2** | `hybrid_sac.py:1171` |
| `output_dim` (embedding_dim) | **64** | `hybrid_sac.py:1146`, `1172` |
| `dropout` | **0.1** | `hybrid_sac.py:1173` |
| GAT 层数 | **1** (单层) | 架构设计 |

**Trigger Context 结构 (v3.7)**:
```python
# PROACTIVE: [1.0, 0.0, risk_ratio]  # risk_ratio ∈ [0, 1]
# REACTIVE:  [0.0, 1.0, 1.0]         # 始终最大风险
```

#### SACDiscreteActor (策略网络)

| 参数 | 值 | 定义位置 |
|------|-----|---------|
| `embedding_dim` | **64** | `hybrid_sac.py:1176` |
| `sa_prior_dim` | **2** | `hybrid_sac.py:1177` |
| `hidden_dim` | **128** | `hybrid_sac.py:1178` |
| `action_dim` | **3** | `hybrid_sac.py:1147`, `1179` |

**Actor 网络结构**:
```
Input: 66 (64 embedding + 2 SA prior)
  └─ Linear(66, 128) + ReLU
       └─ Linear(128, 64) + ReLU
            └─ Linear(64, 3) → Action Logits
```

#### SACDiscreteCritic (双 Q 网络)

| 参数 | 值 | 定义位置 |
|------|-----|---------|
| `embedding_dim` | **64** | `hybrid_sac.py:1183` |
| `sa_prior_dim` | **2** | `hybrid_sac.py:1184` |
| `hidden_dim` | **128** | `hybrid_sac.py:1185` |
| `action_dim` | **3** | `hybrid_sac.py:1186` |

### 3.2 训练超参数

| 参数 | 值 | 定义位置 |
|------|-----|---------|
| `learning_rate` | **3e-4** | `hybrid_sac.py:1148` |
| `gamma` (折扣因子) | **0.95** | `hybrid_sac.py:1153` |
| `tau` (软更新系数) | **0.005** | `hybrid_sac.py:1154` |
| `batch_size` | **32** | `hybrid_sac.py:1155` |
| `memory_size` (回放池) | **10000** | `hybrid_sac.py:1156` |

### 3.3 熵温度调节 (Alpha)

| 参数 | 值 | 定义位置 |
|------|-----|---------|
| `alpha_init` | **0.05** | `hybrid_sac.py:1152` |
| `target_entropy` | **≈ 0.99** | `hybrid_sac.py:1162` |

**Target Entropy 计算**:
```python
target_entropy = -log(1.0 / action_dim) * 0.90
              = -log(1/3) * 0.90
              = log(3) * 0.90
              ≈ 1.099 * 0.90
              ≈ 0.99
```

**Alpha 自动调节逻辑** (`optimize_sac` 函数):
```python
# 当实际熵 < target_entropy 时，alpha 增加 → 鼓励探索
# 当实际熵 > target_entropy 时，alpha 减少 → 鼓励利用
alpha_loss = -(log_alpha.exp() * (log_probs[action].detach() + target_entropy))
```

### 3.4 行为克隆课程学习 (Curriculum BC)

| Epoch | BC 概率 | 说明 |
|-------|---------|------|
| 0 | **0.95** | 近乎纯模仿 |
| 1 | **0.85** | 轻微探索 |
| 2 | **0.65** | 平衡学习 |
| 3 | **0.35** | 更多探索 |
| 4 | **0.10** | SA 锚点防止漂移 |
| 5 (Eval) | **0.00** | 纯确定性评估 |

**来源**: `hybrid_sac.py:1122`

### 3.5 动作空间定义

| Action ID | 语义 | 说明 |
|-----------|------|------|
| 0 | **STAY** | 保持当前服务器 |
| 1 | **FOLLOW SA** | 遵循 SA 建议 |
| 2 | **NEAREST** | 移动到最近服务器 |

---

## 4. 奖励函数惩罚系数 (Reward Multipliers)

### 4.1 奖励函数结构

**总成本公式** (`core/reward.py:203-208`):
```python
total_cost = α * C_access + β * C_comm + γ * C_migrate + δ * C_future + C_tear
reward = -total_cost
```

### 4.2 权重系数

| 参数 | 值 | 定义位置 |
|------|-----|---------|
| α (access_latency) | **1.0** | `reward.py:62` |
| β (communication) | **0.05** | `reward.py:62` |
| γ (migration) | **1.5** | `reward.py:62` |
| δ (future_penalty) | **0.5** | `reward.py:63` |

### 4.3 SLA 违规惩罚

| 参数 | 值 | 定义位置 |
|------|-----|---------|
| `SLA_DISTANCE_THRESHOLD` | **15.0 km** | `reward.py:43` |
| `SLA_VIOLATION_MULTIPLIER` | **5.0** | `reward.py:44` |

**死亡惩罚逻辑**:
```python
if node_latency > SLA_DISTANCE_THRESHOLD:
    access_latency += node_latency * SLA_VIOLATION_MULTIPLIER  # 5x 惩罚
else:
    access_latency += node_latency
```

### 4.4 跨边通信惩罚 (DAG 撕裂)

| 参数 | 值 | 定义位置 |
|------|-----|---------|
| `EDGE_TEAR_DISTANCE_KM` | **20.0 km** | `reward.py:47` |
| `EDGE_TEAR_MULTIPLIER` | **3.0** | `reward.py:48` |
| 撕裂惩罚阈值 | **10.0 km** | `reward.py:134` |
| 撕裂惩罚系数 | **0.5** | `reward.py:135` |

**惩罚机制**:
```python
# 极端距离惩罚 (>20km)
if dist > EDGE_TEAR_DISTANCE_KM:
    edge_term *= EDGE_TEAR_MULTIPLIER  # 3x

# 中等距离惩罚 (>10km)
if dist > 10.0:
    tearing_penalty += norm_traffic * (dist - 10.0) * 0.5
```

### 4.5 状态迁移成本 (Asymmetric)

| 参数 | 值 | 定义位置 | 含义 |
|------|-----|---------|------|
| `STATE_COST_PROACTIVE` | **500.0** | `reward.py:36` | 背景同步 (state_mb / 500) |
| `STATE_COST_REACTIVE` | **5.0** | `reward.py:37` | 前台阻塞 (state_mb / 5) |
| `MIGRATION_BASE_MULTIPLIER` | **1.5** | `reward.py:40` | 基础迁移乘数 |
| `BANDWIDTH_MBPS` | **100.0** | `reward.py:12` | 网络带宽 |

**非对称成本比**:
- PROACTIVE 状态成本: `state_mb / 500` → **100x 折扣**
- REACTIVE 状态成本: `state_mb / 5`

### 4.6 JIT 动态折扣公式 (v3.7)

**核心公式** (`reward.py:174`):
```python
if trigger_type == TRIGGER_PROACTIVE:
    state_divisor = STATE_COST_REACTIVE + (STATE_COST_PROACTIVE - STATE_COST_REACTIVE) * (risk_ratio ** 2)
else:
    state_divisor = STATE_COST_REACTIVE  # 始终昂贵
```

**折扣曲线** (基于 `risk_ratio = dist / 15.0`):

| 距离 (km) | risk_ratio | state_divisor | 相对 REACTIVE 的折扣 |
|-----------|------------|---------------|---------------------|
| 5.0 | 0.33 | 59.5 | 11.9x |
| 10.0 | 0.67 | 227.8 | 45.6x |
| 12.0 | 0.80 | 322.0 | 64.4x |
| 14.0 | 0.93 | 430.7 | 86.1x |
| 15.0 | 1.00 | 500.0 | 100x |

**物理含义**: 二次方曲线确保折扣在**最后几公里**才急剧增加，实现 "Just-In-Time" 迁移。

### 4.7 未来惩罚配置

| 参数 | 值 | 定义位置 |
|------|-----|---------|
| `FUTURE_DECAY` | **0.9** | `reward.py:13` |
| `FUTURE_DIST_THRESHOLD` | **15.0 km** | `reward.py:14` |

**时间衰减公式**:
```python
for h, (pred_lat, pred_lon) in enumerate(predicted_locations):
    w_h = FUTURE_DECAY ** h  # 0.9^0, 0.9^1, 0.9^2, ...
    if future_dist > FUTURE_DIST_THRESHOLD:
        future_penalty += w_h * (future_dist - FUTURE_DIST_THRESHOLD)
```

---

## 5. 公平性分析 (Fairness Analysis)

### 5.1 SA vs SAC 公平性检查

| 维度 | SA | SAC | 一致性 |
|------|-----|-----|--------|
| 预测视界 | 15 步 | 15 步 | ✅ |
| 候选服务器 | k=3 | k=3 | ✅ |
| SLA 阈值 | 15.0 km | 15.0 km | ✅ |
| 奖励函数 | `calculate_microservice_reward` | `calculate_microservice_reward` | ✅ |
| DAG 拓扑 | `MICROSERVICE_DAGS` | `MICROSERVICE_DAGS` | ✅ |
| 触发逻辑 | `get_trigger_type` | `get_trigger_type` | ✅ |

### 5.2 潜在偏差点

| 偏差类型 | 描述 | 严重性 |
|----------|------|--------|
| **SA 计算预算** | SA 固定 30 次迭代，不受实际难度影响 | ⚠️ 中等 |
| **SAC 训练优势** | SAC 经过 5 epoch 训练后评估，SA 无训练 | ⚠️ 已知设计 |
| **随机种子** | 代码中未发现统一随机种子设置 | ⚠️ 可能影响复现性 |

### 5.3 时延测量公平性

| 算法 | 测量内容 | 计时方法 |
|------|----------|----------|
| SA | `microservice_simulated_annealing()` 调用 | `time.perf_counter()` |
| SAC | GAT 前向 + Actor 全节点推理 | `time.perf_counter()` |

**注意**: SAC 时延测量包含 SA 调用作为 prior 生成，这是**设计选择**而非偏差。

### 5.4 建议补充实验

1. **消融实验**: 调整 SA 的 `max_iter` (如 50, 100) 观察解质量与时延权衡
2. **种子控制**: 设置统一随机种子 `random.seed()`, `np.random.seed()`, `torch.manual_seed()`
3. **统计显著性**: 多次运行取均值和标准差

---

## 📊 参数速查表

```
┌─────────────────────────────────────────────────────────────────┐
│                     环境常量 (Environment)                       │
├─────────────────────────────────────────────────────────────────┤
│  FORECAST_HORIZON         = 15 步                               │
│  SLA_DISTANCE_THRESHOLD   = 15.0 km                             │
│  PROACTIVE_WARNING_KM     = 5.0 km                              │
│  k_nearest_servers        = 3                                   │
│  DAG 平均节点数           ≈ 4.3 (加权)                          │
├─────────────────────────────────────────────────────────────────┤
│                     SA 配置 (Baseline)                          │
├─────────────────────────────────────────────────────────────────┤
│  initial_temp             = 100.0                               │
│  cooling_rate             = 0.95                                │
│  max_iter                 = 30                                  │
├─────────────────────────────────────────────────────────────────┤
│                     SAC 超参数 (RL)                             │
├─────────────────────────────────────────────────────────────────┤
│  learning_rate            = 3e-4                                │
│  gamma                    = 0.95                                │
│  tau                      = 0.005                               │
│  batch_size               = 32                                  │
│  memory_size              = 10000                               │
│  alpha_init               = 0.05                                │
│  target_entropy           ≈ 0.99                                │
├─────────────────────────────────────────────────────────────────┤
│                     GAT 架构 (Graph)                            │
├─────────────────────────────────────────────────────────────────┤
│  hidden_dim               = 64                                  │
│  num_heads                = 2                                   │
│  output_dim               = 64                                  │
│  trigger_dim              = 3 (v3.7)                            │
│  dropout                  = 0.1                                 │
├─────────────────────────────────────────────────────────────────┤
│                     奖励权重 (Reward)                           │
├─────────────────────────────────────────────────────────────────┤
│  α (access)               = 1.0                                 │
│  β (communication)        = 0.05                                │
│  γ (migration)            = 1.5                                 │
│  δ (future)               = 0.5                                 │
│  SLA_VIOLATION_MULTIPLIER = 5.0                                 │
│  EDGE_TEAR_MULTIPLIER     = 3.0                                 │
│  STATE_COST_PROACTIVE     = 500.0                               │
│  STATE_COST_REACTIVE      = 5.0                                 │
│  MIGRATION_BASE_MULTIPLIER= 1.5                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

**审计结论**: 代码库配置内部一致，SA 与 SAC 使用相同的物理约束和奖励函数，实验设置公平。建议在论文中明确披露 SA 的 `max_iter=30` 限制及其对时延的影响。

*审计报告自动生成，未修改任何源代码。*
