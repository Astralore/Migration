# Reward v2 奖励函数设计总结

**版本**：2026-05-26  
**代码**：`core/reward.py`（`REWARD_SCHEME=v2`）、`core/marl_reward.py`  
**关联记录**：`test.md` 第十六节（实验阶段与 guard 对照）

---

## 1. 设计动机

GAT-MARL 在 **Reward v1** 下配合 `PROACTIVE_MAX_MIGRATIONS_PER_DECISION=1` 可在中等规模实验中将迁移成本压到可接受区间，但存在结构性问题：

| 问题 | 说明 |
|------|------|
| 训练目标与 SA 不对齐 | SA 直接最小化物理 `total_cost_ms`（线性 ms）；v1 使用 `-log1p(J/1000)`，大代价区间梯度饱和 |
| 依赖硬上限控迁 | v1 + 仅 soft guard（无每步节点数上限）时，单步迁移成本显著高于 SA，策略不会自发「少而精」 |

**v2 目标**：用 **奖励形状** 替代「每步最多迁 1 个节点」的硬规则，使迁 0 / 迁 1 / 迁多个由 **SLA 边际收益 vs 迁移边际成本** 决定；训练信号与「最小化 ms 量纲目标」更接近。

---

## 2. 启用方式

```bash
export REWARD_SCHEME=v2    # 亦支持 2 / reward_v2
```

| 资源 | 路径 |
|------|------|
| 实现 | `core/reward.py` → `is_reward_v2()` 分支 |
| MARL 分解 | `core/marl_reward.py` → `_local_migration_costs()` 复用同一非线性函数 |
| v1/v2 场景对比 | `scripts/compare_reward_v1_v2.py` |
| 冒烟 / 全量 | **`run_reward_v21_cov50.py`**（`MARL_EPOCHS=2` 快筛，默认 8 epoch） |
| v2.0 归档 | `scripts/archive/run_reward_v2_*.py` |

---

## 2.1 Reward v2.1（D1：拓扑感知训练目标）

在 D0 关键路径 comm 进入 `total_cost_ms` 后，v2.1 将 **`L_internal` 纳入训练用 `E_q`**：

```text
L_e2e = L_acc + L_internal
E_q_ms = max(0, L_e2e - USER_SLA_TOLERANCE_MS)    # 仅当 REWARD_V2_USE_INTERNAL_PATH=1
E_q = E_q_ms × FIBER_SPEED_KM_MS
P_SLA_objective = α·E_d² + β·E_q²
J = P_SLA_objective + C_mig^nl
```

**报表 `total_cost_ms` 不变**：`sla_penalty_ms` 仍用空口 `E_q`；`L_internal` 只出现在 `communication_cost` / `internal_critical_path_ms` 行，避免 `L_acc` 在 penalty 与 access 项双重计入。

---

## 3. 超参数（环境变量）

| 环境变量 | 默认值 | 含义 |
|----------|--------|------|
| `REWARD_V2_SLA_ALPHA_MS_PER_KM2` | 80 | 距离超额二次系数 α（ms/km²） |
| `REWARD_V2_QOS_BETA_MS_PER_KM2` | 80 | QoS 超额二次系数 β（ms/km²） |
| `REWARD_V2_MIGRATION_LAMBDA` | 0.75 | 迁移整体权重 λ |
| `REWARD_V2_MIGRATION_TAU_MB` | 100 | exp 尺度 τ（MB） |
| `REWARD_V2_EXP_EXPONENT_CLIP` | 8.0 | 指数上限 c_max |
| `REWARD_V2_OBJECTIVE_SCALE_MS` | 50000 | reward 分母 S；v2.1 脚本默认 100000 |
| `MAX_SLA_PENALTY_MS` | 500000 | D1.5：`P_SLA` 硬上限，防梯度爆炸 |
| `REWARD_RECOVERY_BONUS_MAX` | 0 | 恢复 bonus（关闭） |
| `REWARD_DISTANCE_BONUS_WEIGHT` | 0 | 距离改善 bonus（关闭） |

与 v1 共用：`NONLINEAR_MIGRATION_COST_CLIP_MS = 300000`、带宽模型、迁移底噪 `BASE_MIGRATION_OVERHEAD_MS = 200` 等。

---

## 4. 符号定义

| 符号 | 含义 |
|------|------|
| `d_max` | DAG 所有 **entry 节点** 到用户的最大球面距离（km） |
| `L_acc` | `calc_access_latency_ms(d_max) = d_max / v_fiber + L_0` |
| `D_th` | `DISTANCE_THRESHOLD_KM = 15` km |
| `L_qos` | `USER_SLA_TOLERANCE_MS`（≈ 15 km 接入时延 × 0.99） |
| `S_i` | 节点 i 迁移数据量 `image_mb + state_mb`（MB） |
| `δ_i^raw` | 节点 i 的物理线性迁移时延（ms） |

物理常量：`v_fiber = FIBER_SPEED_KM_MS = 200` km/ms，`L_0 = BASE_ROUTER_DELAY_MS = 2` ms。

**超额量：**

```text
E_d = max(0, d_max - D_th)                         # 距离超额 (km)
E_q = max(0, L_acc - L_qos) * FIBER_SPEED_KM_MS     # QoS 超额折算 (km)
```

---

## 5. SLA 惩罚

v2 **无** base 惩罚、**无**线性项；距离与 QoS **分开**二次惩罚：

```text
P_SLA = 0                         若 E_d = 0 且 E_q = 0
P_SLA = α · E_d² + β · E_q²       否则
```

默认 `α = β = 80` ms/km²。

**SLA 改善量**（counterfactual / guard 使用，按当前 scheme 计算）：

```text
ΔP_SLA = max(0, P_SLA_old - P_SLA_new)
```

预测 horizon 的 **future_gain** 使用 `future_mean_excess_penalty_gain_ms()`：v2 下为 `α·(Ē_old² - Ē_new²)`，v1 下为线性 `(Ē_old - Ē_new)·k₁`（与训练 penalty 形状一致）。

---

## 6. 物理线性迁移时延

与 v1 共用物理层（写入 `details.migration_cost`，并进入评测 `C_total`）。

**有效带宽（随 SLA 风险升高而升高）：**

```text
ρ = min(d_max / D_th, 1)
B_eff = B_min + (B_max - B_min) · ρ²          # B_min=50, B_max=500 Mbps
```

**单节点线性时延**（目标机 t 上本步并发迁移数 `n_t`）：

```text
B_i = B_eff / max(1, n_t)
δ_i^raw = (S_i · 8 / B_i) · 1000 + 200        # 8 = MB→Mbit；200ms 为容器启动底噪
```

Reactive 触发：`δ_i^raw ← 1.5 · δ_i^raw`。

**整步线性迁移和：**

```text
T_mig^lin = Σ_{i: 发生迁移} δ_i^raw
```

---

## 7. 非线性迁移成本（v2 核心）

**尺寸乘子：**

```text
M_exp(S_i) = exp( min(S_i / τ, c_max) )         # τ=100 MB, c_max=8
```

**单节点非线性成本：**

```text
C_i^nl = λ · δ_i^raw · M_exp(S_i)
```

**整步（进入训练目标 J）：**

```text
C_mig^nl = min( Σ_i C_i^nl, 300000 )
```

**直观含义：**

- `S_i ≈ 100 MB` → `M_exp ≈ e ≈ 2.72`
- `S_i ≥ 800 MB` → 指数触顶 `e^8 ≈ 2981`
- 同一步多节点、大镜像迁移的 **边际代价陡增**，无需 `MAX=1` 硬截断

v2 **不使用** v1 的 `CORE_MIGRATION_REWARD_WEIGHT`（λ 已内嵌于迁移项）。

---

## 8. 训练目标与共享 reward

**训练用目标：**

```text
J = P_SLA + C_mig^nl

r_shared = -max(J, 0) / S + reward_bonus
```

- `S = REWARD_V2_OBJECTIVE_SCALE_MS`（默认 **5000** ms；原 1000 在中等测试中引发策略全 STAY）
- v2 下 `reward_bonus = 0`（距离/恢复 bonus 权重均为 0）

**评测用物理总成本**（与 SA 一致，**不**进入 J）：

```text
C_total = L_acc
        + T_mig^lin
        + tearing_delay_ms
        + communication_delay_ms
        + future_delay_ms
        + P_SLA
```

`access` / `tearing` / `comm` / `future` 仍完整计算并写入 `details.total_cost_ms`，供实验报表与 SA 对比。

---

## 9. MARL 智能体级奖励

```text
agent_reward_i = r_shared
                 + dense_distance_bonus_i      # 默认 0
                 + entry_sla_bonus_i           # 默认 0
                 - λ_train · (C_i^nl / 1000)
```

- `C_i^nl`：`_local_migration_costs()`，v2 下同样为 `exp(size/τ)` 形式
- `λ_train`：`lambda_migration`，按 epoch warmup 调度（与 v1 阶段 B 相同）

---

## 10. 与 Reward v1 对照

| 模块 | v1（默认） | v2 |
|------|------------|-----|
| `P_SLA` | `2000 + k₁·E_eq + k₂·E_eq²`，`E_eq = E_d + E_q` | `α·E_d² + β·E_q²` |
| `C_i^nl` | `δ^raw · (1 + 0.75·(S/100)³ + 0.5·log1p(state/100))` | `λ·δ^raw·exp(min(S/τ, 8))` |
| `J` | `P_SLA + w·C_mig^nl`，`w = CORE_MIGRATION_REWARD_WEIGHT` | `P_SLA + C_mig^nl` |
| `r_shared` | `-log1p(J / 1000)` | `-J / S`（默认 S=10000） |
| 手写 bonus | 可配置（当前多为 0） | 强制为 0 |

---

## 11. 推荐 Guard 与实验配置

v2 训练脚本默认 **不按节点个数硬截断**，仅保留 budget + ROI：

```text
PROACTIVE_MAX_MIGRATIONS_PER_DECISION=0
REACTIVE_MAX_MIGRATIONS_PER_DECISION=0
PROACTIVE_MIGRATION_BUDGET_MS=6000
```

### 11.1 成功标准

1. 无 `MAX=1` 时，Proactive Migration Cost / Migration Share 接近 SA。
2. `Avg Total System Cost`、`Severe SLA Violations`、`P95 SLA Excess` 不劣于 SA 或 v1+B2。
3. 单步多迁由策略自发抑制，而非 guard 硬砍。

### 11.2 实验对照矩阵

| 列 | REWARD_SCHEME | MAX/步 | 说明 |
|----|---------------|--------|------|
| B2 | v1 | 1 | 已跑通基线 |
| C softguard | v1 | 0 | 已证 v1 无 MAX 时迁移偏高 |
| v2 softguard | v2 | 0 | 验证奖励是否替代硬上限 |
| v2 PhaseB 对齐 | v2 | 1 | 公平公式对比（仅换 scheme） |

### 11.3 计划实验标识

```text
medium_validation_20260526_reward_v2_scale10000_v1      # 2 epoch 快筛
medium_validation_20260526_reward_v2_softguard_scale10000_v1  # 8 epoch soft guard
medium_validation_20260526_reward_v2_phaseB_aligned_v1  # 8 epoch MAX=1
```

### 11.4 调参提示

若训练初期 reward 量级过大或 **stay_action_ratio → 1**，优先 **调大** `REWARD_V2_OBJECTIVE_SCALE_MS`（默认 10000），并确保 MARL 局部惩罚与 counterfactual 使用同一 S；future_gain 已与 v2 二次 SLA 对齐。

---

## 12. 数据流示意

```text
                    ┌─────────────────────────────────────┐
  部署 / 迁移动作 ──►│ 物理层：δ^raw, tearing, comm, future │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  P_SLA = α·E_d² + β·E_q²            │
                    │  C_mig^nl = Σ λ·δ^raw·exp(S/τ)      │
                    └──────────────┬──────────────────────┘
                                   │
              训练 ◄── J = P_SLA + C_mig^nl ──► r = -J/S
              评测 ◄── C_total（含 access 等全项）──────► 与 SA 对比
```

---

## 13. 一句话归纳

**Reward v2 = 纯二次 SLA + 指数型迁移成本 + 线性负 reward；训练只优化 `J`，报表仍用完整 `C_total`，目标是用可微代价曲面替代「每步最多迁 1 节点」的硬规则，使 GAT-MARL 在 soft guard 下仍能学到接近 SA 的成本—质量权衡。**
