# v3.10 修复方案：解决 Proactive SAC 失效问题

## 问题诊断

### v3.9 实验结果
| 模式 | SAC Migrations | SAC Violations | SA Violations | 状态 |
|------|----------------|----------------|---------------|------|
| **Reactive** | 291 | 1013 | 1014 | ✅ 已修复 |
| **Proactive** | 0 | 1987 | 1020 | ⚠️ 仍失效 |

### 根因分析
Proactive SAC 失效的原因**不是奖励函数**，而是：

1. **BC Schedule 衰减太快**：
   - Proactive: `[0.95, 0.85, 0.65, 0.35, 0.10]` → 快速衰减
   - Reactive:  `[0.98, 0.90, 0.75, 0.55, 0.30]` → 慢速衰减 ✅ 有效
   
2. **Eval Epoch 完全依赖 Q 值**：
   - BC=0 时模型用 argmax(Q) 决策
   - Q 值没有正确学习"迁移优于不迁移"
   - 模型选择 Action 0 (STAY) 避免迁移成本

3. **训练时表现正常，Eval 时崩溃**：
   - 训练 Epoch 1-5：migrations > 900，正常
   - Eval Epoch 6：migrations = 0，崩溃

---

## 核心修改指令

### 任务 1：统一 BC Schedule（最关键）

**文件**：`algorithms/hybrid_sac.py`，约第 1117-1120 行

**原代码**：
```python
if proactive:
    bc_prob_schedule = [0.95, 0.85, 0.65, 0.35, 0.10]  # Proactive: 快速衰减
else:
    bc_prob_schedule = [0.98, 0.90, 0.75, 0.55, 0.30]  # Reactive: 慢速衰减
```

**修改为**：
```python
# v3.10: 统一使用慢衰减 schedule，确保模型充分学习 SA 策略
# Reactive 的成功证明慢衰减有效
bc_prob_schedule = [0.98, 0.90, 0.75, 0.55, 0.30]  # 统一使用慢衰减
```

**原理**：Reactive 用慢衰减成功了，Proactive 也应使用相同策略

---

### 任务 2：降低 Target Network 更新速率

**文件**：`algorithms/hybrid_sac.py`，约第 1152 行

**原代码**：
```python
tau = 0.005
```

**修改为**：
```python
tau = 0.001  # v3.10: 从 0.005 降为 0.001，提高 Q 值稳定性
```

**原理**：更慢的 soft update 让 target Q 更稳定，减少 Q 值震荡

---

### 任务 3：增加 Eval 时的安全保底（可选）

**文件**：`algorithms/hybrid_sac.py`，约第 1404 行

**原代码**：
```python
if is_eval_epoch or inference_mode:
    current_bc_prob = 0.0  # Eval: 完全依赖 Q 值
```

**修改为**：
```python
if is_eval_epoch or inference_mode:
    current_bc_prob = 0.05  # v3.10: 保留 5% SA 保底，防止完全崩溃
```

**原理**：即使在评估时，保留少量 SA 模仿作为安全网

---

### 任务 4：物理化奖励函数（学术优化，非必需）

如果需要让论文的数学模型更严谨，可以进行以下优化：

**文件**：`core/reward.py`

**修改 1**：将抽象除数改为带宽概念
```python
# 物理化常量（可选）
BACKGROUND_SYNC_RATE_MBPS = 500.0   # 背景同步带宽 (Proactive)
FOREGROUND_SYNC_RATE_MBPS = 50.0    # 前台传输带宽 (Reactive)

# JIT 公式（保持当前逻辑，只改名）
# 高 risk_ratio → 更高带宽 → 更低成本（鼓励迁移）
effective_bandwidth = FOREGROUND_SYNC_RATE_MBPS + \
    (BACKGROUND_SYNC_RATE_MBPS - FOREGROUND_SYNC_RATE_MBPS) * (risk_ratio ** 2)
state_transfer_time_ms = (state_mb / effective_bandwidth) * 1000
```

⚠️ **注意**：JIT 公式必须是 `risk_ratio ** 2`（不是 `1 - risk_ratio ** 2`），确保高风险时成本降低！

---

## 修改优先级

| 优先级 | 任务 | 预期效果 |
|--------|------|----------|
| **P0** | 统一 BC Schedule | 解决 Proactive 失效 |
| **P1** | 降低 τ | 稳定 Q 值学习 |
| **P2** | Eval 保底 5% BC | 防止极端崩溃 |
| **P3** | 物理化奖励 | 学术严谨性 |

---

## 验收要求

1. 执行修改后，清理旧权重：`checkpoints/sac_*.pth`
2. 设置 `INFERENCE_MODE = False`，运行训练
3. 验收标准：
   - Proactive Eval Epoch: **migrations > 200**
   - Proactive 测试集: **violations < 1200**
4. 设置 `INFERENCE_MODE = True`，运行推理验证
5. 更新 `result.md` 报告

---

## 为什么不先做物理化重构？

物理化奖励函数是**学术上的优化**，但：
1. 当前问题根因是 **BC Schedule**，不是奖励函数
2. Reactive 已用当前奖励函数成功修复
3. 物理化会增加代码复杂度，调试困难
4. 应先解决功能问题，再优化学术表达

**建议顺序**：
1. v3.10：修复 Proactive（BC Schedule + τ）
2. v3.11：物理化奖励函数（如果需要）
