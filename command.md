# v3.9 系统优化方案（基于实验分析）

## 背景问题分析

根据 2026/4/26 实验结果，发现以下问题：
1. **Hybrid SAC Reactive 模式完全失效**：0 迁移、1987 违规（SA 仅 1017）
2. **Loss 曲线存在 1e7 量级尖峰**：Critic 训练不稳定
3. **Reward 存在 -40000 极端负值**：导致梯度爆炸

**根因**：`SLA_VIOLATION_MULTIPLIER=15.0` 惩罚过重 + MSE Loss 对极端值敏感

---

## 核心修改指令

### 任务 1：修复 Reactive 模式的"策略瘫痪"

**文件**：`core/reward.py`

**修改 1**：降低 Reactive 状态迁移成本（鼓励迁移）
```python
# 第 37 行：STATE_COST_REACTIVE 从 5.0 改为 50.0
# 原理：除数变大 → state_mb/50 < state_mb/5 → 迁移成本降低 10 倍
STATE_COST_REACTIVE = 50.0  # v3.9: 从 5.0 改为 50.0，降低 Reactive 迁移成本
```

**修改 2**：降低 SLA 死亡惩罚（避免过度惩罚）
```python
# 第 44 行：SLA_VIOLATION_MULTIPLIER 从 15.0 改为 8.0
# 原理：15x 太高导致模型"躺平"，8x 仍能惩罚违规但允许探索
SLA_VIOLATION_MULTIPLIER = 8.0  # v3.9: 从 15.0 降为 8.0
```

**修改 3**：添加 Reward Clipping（防止极端负值）
```python
# 在 calculate_microservice_reward 函数返回前（约第 208 行后）添加：
reward = max(reward, -2000.0)  # v3.9: 截断极端负奖励
```

⚠️ **不要修改** `gamma`（迁移权重 1.5）：增大会加剧"不迁移"问题

---

### 任务 2：引入 Huber Loss 稳定 Critic 训练

**文件**：`algorithms/hybrid_sac.py`

**修改位置**：`optimize_sac` 函数，第 912 行

**原代码**：
```python
critic_loss = F.mse_loss(q1_action, target_value) + F.mse_loss(q2_action, target_value)
```

**修改为**：
```python
# v3.9: 使用 Huber Loss (Smooth L1) 抑制极端误差的梯度爆炸
critic_loss = F.smooth_l1_loss(q1_action, target_value) + F.smooth_l1_loss(q2_action, target_value)
```

**原理**：Smooth L1 Loss 在误差 > 1 时为线性，消除 MSE 的二次放大效应

---

### 任务 3：保持 GAT 架构不变

**分析**：当前 `TriggerAwareGAT` 是精心设计的自定义实现，具有：
- Trigger-Conditioned Attention（触发类型感知）
- Stateful-Trigger Interaction Bias（状态节点交互偏置）
- Multi-Head Attention（2 头，已覆盖不同拓扑视角）

**结论**：对于 3-6 节点的 DAG，当前架构已足够。增加层数会：
- 增加过拟合风险
- 增加训练时间
- 破坏已有的 trigger-aware 机制

✅ **不修改 GAT 架构**

---

### 任务 4：为 Reactive 模式设置专用 BC Schedule

**文件**：`algorithms/hybrid_sac.py`

**修改位置**：约第 1122 行

**原代码**：
```python
bc_prob_schedule = [0.95, 0.85, 0.65, 0.35, 0.10]
```

**修改为**：
```python
# v3.9: 根据模式设置不同的 BC 衰减速度
if proactive:
    bc_prob_schedule = [0.95, 0.85, 0.65, 0.35, 0.10]  # Proactive: 原有 schedule
else:
    bc_prob_schedule = [0.98, 0.90, 0.75, 0.55, 0.30]  # Reactive: 更慢衰减，更多模仿
```

**原理**：Reactive 模式面对已违规的紧急情况，需要更长时间学习 SA 的"带伤迁移"策略

---

## 修改汇总表

| 文件 | 行号 | 参数 | 原值 | 新值 | 目的 |
|------|------|------|------|------|------|
| `reward.py` | 37 | `STATE_COST_REACTIVE` | 5.0 | **50.0** | 降低 Reactive 迁移成本 |
| `reward.py` | 44 | `SLA_VIOLATION_MULTIPLIER` | 15.0 | **8.0** | 降低死亡惩罚 |
| `reward.py` | ~208 | Reward Clipping | 无 | **-2000** | 防止极端负值 |
| `hybrid_sac.py` | 912 | Critic Loss | `mse_loss` | **`smooth_l1_loss`** | 稳定训练 |
| `hybrid_sac.py` | 1122 | `bc_prob_schedule` | 单一 | **分 Proactive/Reactive** | 模式专用 |

---

## 验收要求

1. 完成代码修改后，**删除** `checkpoints/` 下的旧权重文件
2. 设置 `INFERENCE_MODE = False`，运行 `python run_comparison.py` 进行训练
3. 观察 Loss 曲线是否平滑（尖峰 < 1e5）
4. 确认 Reactive 模式的迁移数 > 0
5. 设置 `INFERENCE_MODE = True`，运行推理验证
6. 检查 Reactive 违规数是否降至 ~1000（与 SA 相当）
