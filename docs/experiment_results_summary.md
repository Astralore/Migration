# 实验结果整理（cov50 中等规模）

**最后更新**：2026-06-07（清理无用实验目录与日志）
**入口脚本**：`run_reward_v21_p3_cov50.py`（B1.1b 定版，reactive-only）
**命名规范**：`experiments/medium_validation_YYYYMMDD_HHMMSS_<tag>/`

---

## 1. 保留目录一览

| 目录 | 阶段 | 配置要点 | 推理 GAT（Reactive） |
|------|------|----------|----------------------|
| `medium_validation_20260607_175355_reward_v21_p3_b11b_8ep` | **B1.1b 定版** | P3 + CF gate + hard clip | **150 迁 / 8.1k ms / P95 4.9 km** |
| `medium_validation_20260605_170422_reward_v21_p3_8ep` | P3 全量 | γ 课程、无 gate | 1736 迁 / 45.8k ms |
| `medium_validation_20260605_145403_reward_v21_p1_v1` | P1 验证 | entry-first + max-1 | 1758 迁 |
| `medium_validation_20260605_reward_v21_softguard_reactive_8ep_v1` | v2.1 主线 | bypass 硬护栏 | 0 迁 |
| `medium_validation_20260526_phaseC_softguard_v1` | v1 标杆 | Phase C soft guard | 72 迁 / 17.7k ms |

**已删除**：P4 实验、B1.1a 死锁、P2 重复、Phase B/B2、scale10000、checkpoint/phaseC0 推理抽查、重复 P1、全部 `experiments/*.log`。

---

## 2f. P3+B1.1a 全量（20260607，8 epoch reactive）

**路径**：`experiments/medium_validation_20260607_001742_reward_v21_p3_b11a_8ep/`

**配置**：P3 + `MARL_CF_SLA_GAIN_GATE=1`（`sla_gain_ms > 0` 硬 mask）。

### 推理段

| 指标 | P3 GAT | C1 B1.1a（旧 ckpt） | **B1.1a 8ep** | SA |
|------|--------|---------------------|---------------|-----|
| 迁移 | 1736 | 349 | **0** | 14 |
| Avg Cost (ms) | 45840 | 71991 | **55212** | 9639 |
| P95 (km) | 6.72 | 31.0 | **31.0** | 6.63 |
| stay_ratio | 0.857 | 0.885 | **1.000** | — |

**结论**：CF gate 在 v2 internal-path 下 **improving 槽位恒为 0**，8ep 全训练仍 **0 迁移**；从过迁（1736）摆回 **stay 死锁**。C1 快验的 349 迁来自旧策略+推理期 gate，不可作为训练目标。

---

## 2e. P3 验证结论（20260605，8 epoch reactive）

**路径**：`experiments/medium_validation_20260605_170422_reward_v21_p3_8ep/`

**配置**：P1 + P2 + `REWARD_V2_INTERNAL_GAMMA=1`、warmup=2、8 epoch、无 P1 热启动。

### 推理段 vs P1/P2 / SA / Phase C

| 指标 | P3 GAT | P1/P2 GAT | Phase C GAT | SA |
|------|--------|-----------|-------------|-----|
| 迁移 | **1736** | 1758 | 72 | 11 |
| Avg Cost (ms) | **45840** | 88891 | 17674 | 9459 |
| P95 (km) | **6.72** | 31.0 | 6.63 | 6.63 |
| stay_ratio | 0.857 | 0.851 | 0.994 | — |

**部分成功** ✅：γ 课程显著改善 **成本（≈减半）** 与 **P95（31→6.7 km，接近 SA）**。

**未达标** ❌：推理迁移仍 **~1736**（目标 SA 量级 ~10–20）；γ 蒙眼未能打破「每决策几乎必迁」策略。

**分 epoch 观察**：γ→0.75 时 ep5 迁移峰值 18793；γ=1.0 后 ep6 降至 6229，但最终 eval ep8 仍 18209——策略在 full γ 下未收敛到少迁。

---

## 2c. P2 验证结论（20260605，2 epoch reactive）

**路径**：`experiments/medium_validation_20260605_162655_reward_v21_p2_2ep/`

**配置**：`REWARD_V2_CURRICULUM=1`、`MARL_SOFT_CF_BIAS=1`、P1 热启动（P1 checkpoint 全量加载）、S 150k→60k / αβ 0.5→1.0 / λ 0.6→1.25。

### 推理段 vs P1 / Phase C / SA

| 指标 | P2 GAT | P1 GAT | Phase C GAT | SA |
|------|--------|--------|-------------|-----|
| 迁移 | 1758 | 1758 | 72 | 13 |
| Avg Cost (ms) | 88891 | 88891 | 17674 | 7527 |
| P95 (km) | 31.0 | 31.0 | 6.6 | 8.6 |
| stay_ratio | 0.851 | 0.851 | 0.994 | — |

**P2 机制验证** ✅：日志 `P2: curriculum=on soft_cf=on`、分 epoch 打印 S/α/β/λ、Phase C 对照写入 `result.md`。

**效果**：2 epoch + P1 热启动下指标与 P1 几乎相同；迁移仍过频，需 **8 epoch 全量** 观察课程退火是否压低 stay/迁移。

---

## 2b. P1 验证结论（20260605，2 epoch reactive）

**路径**：`experiments/medium_validation_20260605_145403_reward_v21_p1_v1/`

**配置**：`MARL_P1=1`（entry-first mask + 每步 max-1 迁）、DAG family one-hot（6d）、entry CF 特征（cand 12d / node +3d）。

### 推理段（Reactive）

| 算法 | 迁移 | Avg Total Cost (ms) | P95 Excess (km) | stay_ratio |
|------|------|---------------------|-----------------|------------|
| **SA** | 15 | **6660** | **7.5** | — |
| DQN | 1295 | 71274 | 4.9 | — |
| **GAT-MARL** | **1758** | 88891 | 31.0 | **0.851** |

### GAT-MARL 分 epoch（train.reactive.epoch_stats）

| Epoch | Phase | 迁移 | stay_ratio | p1_suppressed | 说明 |
|-------|-------|------|------------|---------------|------|
| 0 | train | 7127 | 0.795 | 6564 | ε-greedy 探索 + P1 协调生效 |
| 1 | eval | 10987 | 0.857 | 52643 | 每决策几乎 1 次 entry 迁（10987/10987） |

**P1 机制验证** ✅：`P1: entry-first + max-1` 日志、`p1_multi_migration_suppressed_count` 非零；**打破 v2.1 推理 0 迁移**（1758 vs 0）。

**待 P2**：迁移过频、成本 ~89k ms（SA ~6.6k）、P95 ~31 km 仍差；需 reward 课程 / λ 调度让策略在「能迁」与「少迁」间收敛。

---

## 2. v2.1 主线结论（20260605，8 epoch reactive）

**路径**：`experiments/medium_validation_20260605_reward_v21_softguard_reactive_8ep_v1/`

### 推理段（Reactive）

| 算法 | 迁移 | Avg Total Cost (ms) | P95 Excess (km) | Severe SLA |
|------|------|---------------------|-----------------|------------|
| **SA** | 13 | **9814** | 6.6 | 751 |
| Nearest | 874 | 270719 | 4.9 | 443 |
| DQN | 735 | 66173 | 7.3 | 784 |
| **GAT-MARL** | **0** | **55212** | **31.0** | 1000 |

### GAT-MARL 分 epoch 探索（train.reactive.epoch_stats）

| Epoch | Phase | 迁移 | stay_ratio | act_1/2/3 | 说明 |
|-------|-------|------|------------|-----------|------|
| 0 | train | 4181 | **0.893** | 2145/757/1279 | ✅ 探索成功 |
| 1–6 | train | 936–2461 | 0.956–0.987 | 有但递减 | 策略向 stay 收敛 |
| 7 | eval | 0 | **1.000** | 0/0/0 | ε=0 argmax，报表顶层指标来源 |

**已验证**：
- `cost_guard_blocked_count = 0`（硬护栏 bypass 生效）
- `migrate_opt_decisions` = 100%（非 mask 问题）
- 瓶颈：**负 reward 下策略学会 stay**，非探索缺失

---

## 3. 历史对照（精简）

| 目录 | GAT 推理成本 (ms) | GAT P95 (km) | 备注 |
|------|-------------------|--------------|------|
| Phase C v1（Pro） | ~18584 | ~5.2 | v1 + proactive 最佳 |
| v2 scale10000（Re） | ~55k | ~31 | 硬护栏时代，Re 0 迁移 |
| **v2.1 8ep（Re）** | ~55212 | ~31 | bypass 后 Epoch0 有迁移，eval 仍 0 |

SA 锚点在 D0 后始终健康（本轮推理 **9814 ms**，~13 迁）。

---

## 4. 路线图（与 test.md 一致）

| 优先级 | 内容 |
|--------|------|
| **P0** ✅ | 分 epoch 指标 + 8 epoch reactive 全量 |
| **P1** ✅ | 动作空间向 SA 邻域靠拢（entry-first / 每步最多 1 迁） |
| **P1** ✅ | DAG type 条件化 + entry counterfactual 特征 |
| **P2** ✅ | reward 课程（S / αβ 退火）+ 软 logit bias |
| **P2** ✅ | Phase C checkpoint 迁移学习对照（报告内历史基线 + P1 热启动） |

---

## 5. 启动新实验

```bash
# B1.1b 定版（默认 8ep reactive-only）
python -u run_reward_v21_p3_cov50.py

# 快筛 2ep
set MEDIUM_VALIDATION_MARL_EPOCHS=2 && python -u run_reward_v21_p3_cov50.py

# C1 推理快验（已有 checkpoint）
set MEDIUM_VALIDATION_STAMP=20260607_175355_reward_v21_p3_b11b_8ep
set MEDIUM_VALIDATION_INFERENCE_ONLY=1
set MEDIUM_VALIDATION_INFERENCE_FORCE=1
python -u run_reward_v21_p3_cov50.py
```
