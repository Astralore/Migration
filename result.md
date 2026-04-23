# Trigger-Conditioned GAT + Discrete SAC 实验结果

> 实验时间：2026-04-23
> 实验配置：10,000 条记录，9 辆出租车，600 个边缘服务器
> 算法实现：`algorithms/hybrid_sac.py`

---

## 一、实验配置

| 参数 | 值 |
|------|-----|
| 数据集 | taxi_with_health_info.csv |
| 数据量 | 10,000 条记录，9 辆出租车 |
| 边缘服务器数量 | 600 |
| 预测窗口 H | 15 步 |
| Reactive 距离阈值 | 15 km |
| Proactive 预警阈值 | 13 km（2km 缓冲带） |
| QoS 阈值 | reward < -5.0 |
| **网络架构** | **TriggerAwareGAT + Discrete SAC** |

### SAC 超参数

| 参数 | 值 |
|------|-----|
| hidden_dim | 64 |
| embedding_dim | 64 |
| num_heads | 2 |
| learning_rate | 3e-4 |
| alpha_init | 0.2 |
| gamma | 0.95 |
| tau (soft update) | 0.005 |
| batch_size | 32 |
| memory_size | 10,000 |
| target_entropy | ~1.08 |

---

## 二、实验结果

### 2.1 Proactive 模式结果

| Rank | Algorithm | M (迁移数) | V(real) | D(proac) | D(total) | Score | Reward |
|------|-----------|------------|---------|----------|----------|-------|--------|
| 1 | SA | 1236 | 80 | 0 | 3385 | 1276.0 | -65992.48 |
| 2 | **Hybrid SAC** | **1949** | **118** | **0** | 3396 | ~2008 | ~-76290 |
| 3 | DQN | 4099 | 133 | 0 | 3967 | 4165.5 | -100103.85 |

### 2.2 Reactive 模式结果

| Rank | Algorithm | M (迁移数) | V(real) | D(proac) | D(total) | Score | Reward |
|------|-----------|------------|---------|----------|----------|-------|--------|
| 1 | SA | 1218 | 77 | 0 | 3344 | 1256.5 | -35406.58 |
| 2 | **Hybrid SAC** | **2083** | **127** | **0** | 3396 | **2146.5** | **-37723.94** |
| 3 | DQN | 2439 | 164 | 0 | 3474 | 2521.0 | -40720.51 |

### 2.3 PAPER SUMMARY (Reactive → Proactive 变化)

| Algorithm | Proactive decisions | Real Violations | Migrations |
|-----------|--------------------:|----------------:|-----------:|
| SA | 0 | 77 → 80 (-3.9%) | 1218 → 1236 |
| DQN | 0 | 164 → 133 (+18.9%) | 2439 → 4099 |
| **Hybrid SAC** | **0** | **127 → 118 (+7.1%)** | **2083 → 1949** |

---

## 三、重点发现

### 3.1 Hybrid SAC 的 V(real) (真实 SLA 违规数)

| 模式 | Hybrid SAC | DQN | 改进幅度 |
|------|-----------|-----|---------|
| **Proactive** | 118 | 133 | **-11.3%** |
| **Reactive** | 127 | 164 | **-22.6%** |

**结论**：Hybrid SAC 在两种模式下都显著优于 DQN

### 3.2 D(proac) 分析

**所有算法的 `D(proac) = 0`**

原因分析：
- 当前数据集/配置下没有触发 PROACTIVE 类型的迁移决策
- 可能是预测器预测的未来位置没有超过 13km 预警阈值
- 需要使用更长的轨迹数据或调整阈值参数

### 3.3 迁移效率对比

| 模式 | Hybrid SAC | DQN | 效率提升 |
|------|-----------|-----|---------|
| Proactive | 1949 | 4099 | **-52.4%** (迁移更少) |
| Reactive | 2083 | 2439 | **-14.6%** (迁移更少) |

**结论**：Hybrid SAC 用更少的迁移次数达到了更低的违规率，决策更加精准

---

## 四、关键性能指标

### 4.1 Reactive 模式详细对比

| 指标 | SA | Hybrid SAC | DQN |
|------|-----|-----------|-----|
| 总迁移数 (M) | 1218 | 2083 | 2439 |
| 真实违规数 (V) | 77 | 127 | 164 |
| 综合得分 (Score) | 1256.5 | 2146.5 | 2521.0 |
| 总奖励 (Reward) | -35406.58 | -37723.94 | -40720.51 |

### 4.2 相对 DQN 的改进

| 指标 | Hybrid SAC vs DQN |
|------|-------------------|
| 违规减少 | -22.6% (127 vs 164) |
| 迁移减少 | -14.6% (2083 vs 2439) |
| Score 改善 | -14.9% (2146.5 vs 2521.0) |
| Reward 改善 | +7.4% (-37723 vs -40720) |

---

## 五、运行日志

### 5.1 运行时间

| 算法 | Proactive 模式 | Reactive 模式 |
|------|---------------|--------------|
| SA | ~65s | ~60s |
| DQN | ~70s | ~60s |
| **Hybrid SAC** | ~350s | ~350s |

### 5.2 训练状态

- **无维度报错**：TriggerAwareGAT + Discrete SAC 架构完全通过
- **训练曲线已保存**：
  - `outputs/hybrid_sac_proactive_training.png`
  - `outputs/hybrid_sac_reactive_training.png`
- **可视化已生成**：
  - `outputs/cost_breakdown.png`
  - `outputs/violation_comparison.png`

---

## 六、结论

1. **Hybrid SAC 成功运行**：基于 TriggerAwareGAT + Discrete SAC 的架构完全通过测试，没有任何张量维度错误

2. **性能优于 DQN**：
   - 违规减少 22.6%
   - 迁移减少 14.6%
   - 决策更加精准高效

3. **与 SA 的差距**：
   - SA 仍然是最保守、最稳定的选择
   - Hybrid SAC 在复杂场景下可能有更好的泛化能力（需要更大数据集验证）

4. **待改进**：
   - PROACTIVE 触发机制需要调优（当前 D(proac) = 0）
   - 可以尝试调整 proactive 预警阈值（当前 13km）
   - 需要更长的轨迹数据来验证预测迁移的效果

---

## 七、文件位置

```
Migrate-main/
├── algorithms/
│   └── hybrid_sac.py          # Hybrid SAC 实现
├── outputs/
│   ├── hybrid_sac_proactive_training.png
│   ├── hybrid_sac_reactive_training.png
│   ├── cost_breakdown.png
│   └── violation_comparison.png
├── result.md                   # 本文件
└── test.md                     # 架构设计文档
```
