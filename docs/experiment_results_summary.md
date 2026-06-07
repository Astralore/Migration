# 实验结果整理（cov50 中等规模）

**最后更新**：2026-06-05（P3 8ep 全量完成）
**命名规范**：`experiments/medium_validation_YYYYMMDD_HHMMSS_<tag>/`（由 `MEDIUM_VALIDATION_STAMP_TAG` 或完整 `MEDIUM_VALIDATION_STAMP` 控制）

---

## 1. 保留目录一览

| 目录 | 阶段 | 配置要点 | 状态 |
|------|------|----------|------|
| `medium_validation_20260607_001742_reward_v21_p3_b11a_8ep` | **P3+B1.1a 8ep** | CF gate、8 epoch | ✅ 完整 |
| `medium_validation_20260605_170422_reward_v21_p3_8ep` | **P3 全量** | P1+P2+γ 课程、8 epoch、无 warmstart | ✅ 完整 |
| `medium_validation_20260605_162655_reward_v21_p2_2ep` | **P2 验证** | 课程+soft CF、P1 warmstart、2 epoch | ✅ 完整 |
| `medium_validation_20260605_145403_reward_v21_p1_v1` | **P1 验证** | entry-first+max-1、DAG/CF 特征、2 epoch | ✅ 完整 |
| `medium_validation_20260605_reward_v21_softguard_reactive_8ep_v1` | **v2.1 主线（当前）** | Reactive-only、S=100k、硬护栏 bypass、8 epoch、`epoch_stats` | ✅ 完整 |
| `medium_validation_20260526_phaseC_softguard_v1` | v1 标杆 | Pro+Re、Phase C soft guard | ✅ 对照 |
| `medium_validation_20260526_phaseB2_reward_align_v1` | v1 公平对照 | Pro+Re、MAX=1 | ✅ 对照 |
| `medium_validation_20260526_reward_v2_scale10000_v1` | v2 尺度试验 | S=10000、2 epoch | ✅ 对照 |
| `checkpoint_inference_20260526_phaseA_guard_top16` | 推理抽查 | Phase A checkpoint | ✅ 参考 |
| `phaseC0_inference_20260526_phaseC0_max2_infer` | 推理抽查 | Phase C0 MAX=2 | ✅ 参考 |

**已删除（被上表替代或重复）**：`scale5000_v1`、两次 `s100k` 2-epoch 快筛、`20260527` 版 8-epoch 重复跑、`phaseB_reward_align_v1`（保留 B2）。

---

**待后续**：B1.1a 全量导致 0 迁死锁，需 B1.1b 放宽 CF 门槛，详见 [Reward_v21_P3总结与P4修改计划.md](Reward_v21_P3总结与P4修改计划.md) §12b。

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
# P2（推荐）：课程 + soft CF + 可选 P1 热启动
python -u run_reward_v21_p2_cov50.py
set MEDIUM_VALIDATION_MARL_EPOCHS=2 && python -u run_reward_v21_p2_cov50.py

# P1 / v2.1 基线
python -u run_reward_v21_cov50.py
```
