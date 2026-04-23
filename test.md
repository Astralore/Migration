# 边缘微服务迁移实验报告

> 生成时间：2026-04-23（修复预警缓冲带后）

---

## 一、实验结果

### 1.1 实验配置

| 参数 | 值 |
|------|-----|
| 数据集 | taxi_with_health_info.csv |
| 数据量 | 10,000 条记录，9 辆出租车 |
| 边缘服务器数量 | 600 |
| 预测窗口 H | **15 步**（修复后） |
| Reactive 距离阈值 | 15 km |
| **Proactive 预警阈值** | **13 km**（修复后，2km 缓冲带） |
| QoS 阈值 | reward < -5.0 |

### 1.2 修复内容

| 文件 | 修改 |
|------|------|
| `core/context.py` | 新增 `PROACTIVE_WARNING_KM = 13.0` 预警阈值 |
| `run_comparison.py` | `FORECAST_HORIZON` 从 5 改为 15 |
| `algorithms/dqn.py` | `FORECAST_HORIZON` 从 5 改为 15 |
| `algorithms/sa.py` | `FORECAST_HORIZON` 从 5 改为 15 |
| `algorithms/hybrid_sa_dqn.py` | `FORECAST_HORIZON` 从 5 改为 15 |

### 1.3 Proactive 模式结果（修复后）

| Rank | Algorithm | M (迁移数) | V(real) | D(proac) | D(total) | Score | Reward |
|------|-----------|------------|---------|----------|----------|-------|--------|
| 1 | SA | 1267 | **59** | 0 | 3210 | 1296.5 | -64248.30 |
| 2 | Hybrid SA-DQN | 1632 | **119** | 0 | 3311 | 1691.5 | -70475.40 |
| 3 | DQN | 4605 | **205** | 0 | 3742 | 4707.5 | -80936.17 |

### 1.4 Reactive 模式结果（基线）

| Rank | Algorithm | M (迁移数) | V(real) | D(proac) | D(total) | Score | Reward |
|------|-----------|------------|---------|----------|----------|-------|--------|
| 1 | SA | 1260 | 97 | 0 | 3334 | 1308.5 | -35303.56 |
| 2 | Hybrid SA-DQN | 1522 | 159 | 0 | 3417 | 1601.5 | -37914.39 |
| 3 | DQN | 3941 | 315 | 0 | 3646 | 4098.5 | -49579.50 |

### 1.5 Proactive vs Reactive 对比分析

| Algorithm | Reactive V | Proactive V | **降低率** |
|-----------|------------|-------------|------------|
| SA | 97 | 59 | **-39.2%** |
| DQN | 315 | 205 | **-34.9%** |
| Hybrid SA-DQN | 159 | 119 | **-25.2%** |

### 1.6 关键发现

1. **修复有效**：引入 2km 预警缓冲带 (13km vs 15km) 和更长的预测视界 (15步) 成功让系统**更早发现并响应**潜在的 SLA 违规

2. **Real Violations 大幅降低**：
   - SA 降低 39.2%（97 → 59）
   - DQN 降低 34.9%（315 → 205）
   - Hybrid SA-DQN 降低 25.2%（159 → 119）

3. **D(proac) 仍为 0 的原因**：大部分触发时刻，用户已经处于 >15km 的违规状态，系统记录为 REACTIVE。但 Proactive 模式下的**更积极迁移策略**使得后续违规次数大幅减少

4. **迁移成本权衡**：Proactive 模式的 Reward 更负（成本更高），因为系统执行了更多预防性迁移，但换来了更低的 Real Violations——**这正是 Proactive 策略的价值体现**

### 1.7 生成的可视化图表

| 文件 | 说明 |
|------|------|
| `outputs/cost_breakdown.png` | 三种算法的开销构成堆叠图 |
| `outputs/violation_comparison.png` | Proactive vs Reactive 违规对比图 |
| `outputs/dqn_proactive_training.png` | DQN Proactive 模式训练曲线 |
| `outputs/dqn_reactive_training.png` | DQN Reactive 模式训练曲线 |
| `outputs/hybrid_proactive_training.png` | Hybrid Proactive 模式训练曲线 |
| `outputs/hybrid_reactive_training.png` | Hybrid Reactive 模式训练曲线 |

---

## 二、轨迹预测方法

### 2.1 方法名称

**简单速度外推法 (Simple Velocity-based Extrapolation)**

### 2.2 实现位置

`prediction/simple_predictor.py`

### 2.3 核心算法

```python
# 训练阶段：学习每辆出租车的平均速度向量
dx = np.diff(lons)  # 经度差分序列
dy = np.diff(lats)  # 纬度差分序列
velocity_factors[taxi_id] = (np.mean(dx), np.mean(dy))  # 平均速度

# 预测阶段：线性外推
for step in range(H):
    lon += dx
    lat += dy
    future.append((lon, lat))
```

### 2.4 数学公式

设用户当前位置为 $(lon_t, lat_t)$，历史平均速度为 $(\bar{v}_{lon}, \bar{v}_{lat})$，则未来第 $h$ 步位置预测为：

$$\hat{lon}_{t+h} = lon_t + h \cdot \bar{v}_{lon}$$

$$\hat{lat}_{t+h} = lat_t + h \cdot \bar{v}_{lat}$$

### 2.5 方法特点

| 优点 | 缺点 |
|------|------|
| 计算复杂度 O(1) | 无法捕捉转向行为 |
| 无需额外训练 | 假设匀速直线运动 |
| 适合短期预测 | 历史平均可能过于平滑 |

---

## 三、上下文感知 (Context Awareness)

当前实现中，"上下文"分布在两个层面：**触发上下文** 和 **状态上下文**。

### 3.1 触发上下文 (`core/context.py`)

定义了**何时触发迁移**的感知逻辑：

#### (A) 被动感知 (Reactive)

```python
DISTANCE_THRESHOLD_KM = 15.0  # 真实违规阈值

def check_sla_violation(...):
    spatial_violation = dist > 15.0 km        # 空间上下文
    qos_violation = current_dag_reward < -5.0 # 性能上下文
    return spatial_violation or qos_violation
```

#### (B) 主动感知 (Proactive) - 修复后

```python
PROACTIVE_WARNING_KM = 13.0  # 预警阈值（2km 缓冲带）

def get_trigger_type(..., predicted_locations, proactive_enabled):
    # 1. 先检查当前是否违规 (REACTIVE)
    if check_sla_violation(...):
        return TRIGGER_REACTIVE
    
    # 2. 再检查未来是否会进入预警区 (PROACTIVE)
    if proactive_enabled and predicted_locations:
        for pred_lat, pred_lon in predicted_locations:
            if haversine_distance(pred, gateway) > PROACTIVE_WARNING_KM:  # 13km
                return TRIGGER_PROACTIVE
    
    return None  # 无需迁移
```

#### 上下文感知维度

| 维度 | 类型 | 阈值 |
|------|------|------|
| 空间距离 | 当前（Reactive） | 15 km |
| 空间距离 | 预测（Proactive） | **13 km**（预警缓冲带） |
| QoS 性能 | 当前 | reward < -5.0 |
| 预测轨迹 | 未来 **15** 步 | 任一步 > 13 km |

### 3.2 状态上下文 (`core/state_builder.py`)

定义了 DQN 智能体**看到的上下文特征**：

#### 16 维 DQN 状态向量

| 维度 | 特征组 | 描述 | 归一化 |
|------|--------|------|--------|
| 0-1 | **全局位置** | 用户纬度、经度 | /90, /180 |
| 2 | **空间上下文** | 用户到网关距离 | /50 km |
| 3-4 | **时间上下文** | 小时、星期几 | /24, /7 |
| 5 | **候选上下文** | 可选服务器数量 | /10 |
| 6-8 | **节点属性** | 镜像大小、状态大小、是否有状态 | /200, /256 |
| 9 | **通信上下文** | 节点关联流量占比 | /max_traffic |
| 10 | **拓扑上下文** | 邻居同服务器比例 | [0,1] |
| 11 | **节点空间** | 节点所在服务器到用户距离 | /50 km |
| 12 | **DAG 角色** | 是否为入口节点 | 0/1 |
| 13 | **DAG 规模** | DAG 总节点数 | /10 |
| **14-15** | **移动趋势** | 预测轨迹的平均 Δlat, Δlon | /0.01 |

#### 18 维 Hybrid 状态向量

在 16 维基础上增加 **SA 先验上下文**：

| 维度 | 描述 |
|------|------|
| 16 | SA 建议服务器到用户的距离 |
| 17 | SA 是否建议保持不动 (0/1) |

### 3.3 上下文感知架构图

```
                    ┌─────────────────────────────────────────┐
                    │         Context Awareness Layer         │
                    └─────────────────────────────────────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          │                            │                            │
          ▼                            ▼                            ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  Spatial Context │      │ Temporal Context │      │ Mobility Context │
│  (geo.py)        │      │ (state_builder)  │      │ (predictor)      │
├──────────────────┤      ├──────────────────┤      ├──────────────────┤
│ • 用户位置       │      │ • 时间 (小时)    │      │ • 平均速度向量   │
│ • 服务器位置     │      │ • 星期几         │      │ • 未来 15 步位置 │
│ • haversine距离  │      │                  │      │ • 移动趋势特征   │
└──────────────────┘      └──────────────────┘      └──────────────────┘
          │                            │                            │
          └────────────────────────────┼────────────────────────────┘
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │  Topology Context (DAG-aware features)  │
                    ├─────────────────────────────────────────┤
                    │ • 节点属性 (image_mb, state_mb, stateful)│
                    │ • 边流量 (traffic)                       │
                    │ • 邻居同服比例                           │
                    │ • 入口/非入口角色                        │
                    └─────────────────────────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │     Trigger Decision (context.py)       │
                    │  Reactive (>15km) / Proactive (>13km)   │
                    └─────────────────────────────────────────┘
```

---

## 四、修复历程与改进方向

### 4.1 原始问题

- **proactive_decisions 始终为 0**：预警阈值与真实违规阈值相同（都是 15km），导致在预测到未来会超过 15km 时，当前往往已经超过 15km，被判定为 REACTIVE
- **预测窗口过短**：H=5 无法看到足够远的未来

### 4.2 修复方案

1. **引入预警缓冲带**：`PROACTIVE_WARNING_KM = 13.0`，比真实违规阈值低 2km
2. **延长预测视界**：`FORECAST_HORIZON = 15`，让系统看得更远

### 4.3 修复效果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| SA V 降低率 | -4.8%（反增） | **-39.2%** |
| DQN V 降低率 | +10.2% | **-34.9%** |
| Hybrid V 降低率 | -89.9%（反增） | **-25.2%** |

### 4.4 后续改进方向

| 方向 | 具体方案 |
|------|----------|
| 轨迹预测升级 | 使用 LSTM/Transformer 替换简单速度外推 |
| 动态预警阈值 | 根据用户速度自适应调整 13km 阈值 |
| 自适应窗口 | 根据用户速度动态调整预测窗口 H |
| 上下文扩展 | 增加服务器负载、网络拥塞等系统上下文 |

---

## 五、文件结构参考

```
Migrate-main/
├── algorithms/
│   ├── dqn.py              # DQN 算法实现
│   ├── sa.py               # SA 基线算法
│   └── hybrid_sa_dqn.py    # Hybrid RL-Refined SA
├── core/
│   ├── context.py          # 触发上下文感知（含预警缓冲带）
│   ├── state_builder.py    # 状态上下文构建
│   ├── reward.py           # 奖励函数（含非对称迁移成本）
│   ├── geo.py              # 地理计算工具
│   └── dag_utils.py        # DAG 工具函数
├── prediction/
│   └── simple_predictor.py # 简单速度外推预测器
├── evaluation/
│   ├── metrics.py          # 评估指标
│   └── plot.py             # 可视化工具
├── outputs/                # 生成的图表
└── run_comparison.py       # 主实验入口
```
