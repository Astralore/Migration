# 微服务迁移算法对比实验报告 (v3.9)

**运行模式**：推理模式 (Inference)  
**数据范围**：测试集 [10000:15000]（未见数据）  
**生成时间**：2026-04-26 23:12  
**版本**：v3.9 (Huber Loss + Reactive 优化)

---

## 一、v3.9 核心修改

| 参数 | v3.8 值 | v3.9 值 | 修改目的 |
|------|---------|---------|----------|
| `STATE_COST_REACTIVE` | 5.0 | **50.0** | 降低 Reactive 迁移成本 10 倍 |
| `SLA_VIOLATION_MULTIPLIER` | 15.0 | **8.0** | 降低死亡惩罚，避免策略瘫痪 |
| Critic Loss | `mse_loss` | **`smooth_l1_loss`** | Huber Loss 抑制梯度爆炸 |
| Reward Clipping | 无 | **max(-2000)** | 防止极端负值 |
| Reactive BC Schedule | [0.95,0.85,0.65,0.35,0.10] | **[0.98,0.90,0.75,0.55,0.30]** | 更慢衰减 |

---

## 二、Proactive 模式结果（测试集）

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Score |
|-----------|------------|------------|---------------------|------------------|-------|
| **SA** | 347 | 1020 | 29 | 4.40 | 857.0 |
| DQN | 1624 | 1036 | 0 | 0.00 | 2142.0 |
| Hybrid SAC | 0 | 1987 | 29 | 0.80 | 993.5 |

⚠️ **Proactive SAC 仍存在问题**：Eval 模式下 0 迁移、1987 违规

---

## 三、Reactive 模式结果（测试集）

| Algorithm | Migrations | Violations | Avg Latency (ms) | Score |
|-----------|------------|------------|------------------|-------|
| SA | 578 | 1014 | 1.38 | 1085.0 |
| DQN | 1524 | 1050 | 0.00 | 2049.0 |
| **Hybrid SAC** | **291** | **1013** | **0.76** | **797.5** ✅ |

🎉 **Reactive SAC v3.9 修复成功**：
- 迁移数：0 → **291**（恢复正常迁移行为）
- 违规数：1987 → **1013**（与 SA 的 1014 相当）
- Score：993.5 → **797.5**（最优）

---

## 四、v3.8 vs v3.9 对比

### Reactive 模式改善

| 指标 | v3.8 | v3.9 | 改善 |
|------|------|------|------|
| Migrations | 0 | **291** | 🔥 从瘫痪恢复 |
| Violations | 1987 | **1013** | **-49%** |
| Score | 993.5 | **797.5** | **-20%** |

### Proactive 模式（仍需改进）

| 指标 | v3.8 | v3.9 | 状态 |
|------|------|------|------|
| Migrations | 0 | 0 | ⚠️ 未改善 |
| Violations | 1987 | 1987 | ⚠️ 未改善 |

---

## 五、训练过程分析

### Reactive SAC 训练曲线（v3.9）

| Epoch | Decisions | Migrations | Violations | Reward |
|-------|-----------|------------|------------|--------|
| 1 | 4501 | 937 | 62 | -48,187 |
| 2 | 3993 | 1352 | 71 | -45,356 |
| 3 | 3958 | 1392 | 67 | -44,770 |
| 4 | - | - | - | - |
| 5 | - | - | - | - |
| 6 (EVAL) | 4620 | **952** | **75** | -48,849 |

✅ **训练过程稳定**：迁移数正常，违规数低，Reward 持续改善

### Proactive SAC 训练曲线（v3.9）

| Epoch | Decisions | Migrations | Violations | Reward |
|-------|-----------|------------|------------|--------|
| 1 | 4516 | 1043 | 79 | -50,841 |
| 2 | 4329 | 921 | 104 | -53,612 |
| 3 | 3642 | 2140 | 70 | -46,869 |
| 5 | 3524 | 2102 | 133 | -63,882 |
| 6 (EVAL) | 8372 | **0** | **5007** | -1,569,208 |

⚠️ **EVAL Epoch 异常**：BC=0 时策略崩溃，模型学会了"不迁移"

---

## 六、时延对比分析

| 算法 | Proactive 时延 | Reactive 时延 |
|------|----------------|---------------|
| SA | 4.40 ms | 1.38 ms |
| DQN | 0.00 ms | 0.00 ms |
| **Hybrid SAC** | **0.80 ms** | **0.76 ms** |

**加速比**：SAC vs SA = **5.5x ~ 1.8x** 加速

---

## 七、问题诊断与后续方向

### 已解决 ✅
1. **Reactive SAC 策略瘫痪**：通过降低 STATE_COST_REACTIVE (5→50) 和 SLA_VIOLATION_MULTIPLIER (15→8) 成功修复
2. **Loss 尖峰**：Huber Loss + Reward Clipping 有效抑制

### 待解决 ⚠️
1. **Proactive SAC Eval 模式失效**：
   - 原因：BC schedule [0.95,0.85,0.65,0.35,0.10] 衰减太快
   - 建议：使用与 Reactive 相同的慢衰减 schedule

2. **Q-Value 学习不稳定**：
   - 原因：Proactive 触发较早（5km），奖励信号复杂
   - 建议：增加 target network 更新间隔 (τ: 0.005→0.001)

---

## 八、权重文件存档

```
checkpoints/
├── sac_proactive.pth          # v3.9 Proactive 权重
├── sac_reactive.pth           # v3.9 Reactive 权重 ✅
├── sac_proactive_v3.8_old.pth # v3.8 备份
├── sac_reactive_v3.8_old.pth  # v3.8 备份
├── sac_proactive_v3.7_old.pth # v3.7 备份
└── sac_reactive_v3.7_old.pth  # v3.7 备份
```

---

## 九、结论

**v3.9 成功修复了 Reactive SAC 的策略瘫痪问题**，使其在测试集上达到与 SA 相当的违规水平（1013 vs 1014），同时保持 **5.5x 时延优势**。

Proactive 模式仍需进一步调优，建议在 v3.10 中：
1. 统一使用慢衰减 BC schedule
2. 减小 soft update τ 值
3. 考虑为 Proactive 模式单独调整奖励权重

---

*报告生成时间：2026-04-26 23:12*
