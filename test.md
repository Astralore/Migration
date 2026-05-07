# Hybrid SAC：Smoke 测试与判读报告

> 本文档按请求**覆盖写入**：记录一次带 `HYBRID_SAC_DEBUG_STEPS` 的 `run_comparison.py` 烟雾跑法、终端结论与日志片段。  
> 根因与代码级修复（`argmax(softmax)` → masked logits + tie-break、reward 埋点）见仓库内 `algorithms/hybrid_sac.py`、`core/reward.py`。

---

## 一、Smoke 配置（临时，跑完后已恢复默认）

在 `run_comparison.py` 顶层曾临时设为：

- `ACTIVE_USERS_LIMIT = 5`（意图：极少车辆）
- `NUM_EPOCHS = 1`（Hybrid SAC 仅 1 轮）
- `INFERENCE_MODE = False`（走训练流程）

**命令（Windows PowerShell）：**

```powershell
$env:HYBRID_SAC_DEBUG_STEPS="200"
python run_comparison.py
```

**说明（重要）：**

1. **`ACTIVE_USERS_LIMIT=5` 未缩小本次语料**：`load_data` 命中已存在的预清洗 CSV `taxi_cleaned_active100_min100_eps2h.csv`，仍为 **约 100 辆车、20 万+ 行**。真·5 车极速需 `processed_csv=False` 或提供 `active_users_limit=5` 对应的 processed 文件。  
2. **`NUM_EPOCHS=1`** 时，`is_eval_epoch = (epoch == num_epochs - 1)` 在唯一一轮上恒为 **True**，Hybrid SAC **整段为 EVAL**，**不会调用 `sample_action`**，因此流水线日志中**不会出现** `[HYBRID_SAC_DBG action]` / `sampled_action` 行；这不表示修复失效。  
3. 测试结束后 **`run_comparison.py` 已恢复**为正式默认：`ACTIVE_USERS_LIMIT=100`，Hybrid SAC **`num_epochs=6`**。

---

## 二、判读结论（针对两项核心问题）

### 1. 死锁是否解除？

**结论：已解除。**

- **端到端汇总**：`PAPER SUMMARY` 中 Hybrid SAC 为 **`Migrations: 2448 -> 2664`**，迁移计数非零，与「评估阶段永远 STAY」的旧 bug 不一致。  
- **确定性路径**：对接近全零的 logits，`get_action_deterministic` 输出为 **`1`（FOLLOW_SA）**（masked logits + tie-break），而非旧版在均匀 softmax 上 `argmax` 得到的 **`0`（STAY）**。  
- **`sampled_action` 脑电波**：因 `NUM_EPOCHS=1` 无探索分支，在同一环境下用**独立微脚本**连续调用 `sample_action`，可见 `sampled_action` 在 **0 / 1 / 2** 间变化（见下文片段 B）。

### 2. 奖励截断（Reward Clipping）是否严重？

**结论：在前 200 条 `[HYBRID_SAC_DBG reward]` 窗口内，未发现「`total_cost_ms` 极大、`reward` 死死卡在 `-10000`」的顶格截断。**

- 该窗口内 `total_cost_ms` 峰值约 **6000+ ms**（例如 ~6088），`reward(clipped)` 与 `-total_cost_ms` 一致。  
- 计数在 **`core/reward.py` 模块 import 时**初始化，**前 200 条**多来自 **SA/DQN** 较早阶段的 reward 调用，不等同于「纯 SAC 专属」序列。  
- 全量日志中若出现大量 `total_cost_ms >> 10000` 且 `reward` 恒为 `-10000`，再单独评估是否调 `REWARD_CLIP_MIN` / 尺度；**本次测试未修改 reward 公式**。

---

## 三、代表性日志片段

**A）Smoke 日志中的 reward 埋点（`total_cost` 与 `reward` 未顶格 `-10000`）：**

```text
[HYBRID_SAC_DBG reward] total_cost_ms=2.027105227329716 migration_delay_ms=0.0 tearing_delay_ms=0.0 sla_penalty_ms=0.0 access_latency_ms=2.027105227329716 reward(clipped)=-2.027105227329716 REWARD_CLIP_MIN=-10000.0
[HYBRID_SAC_DBG reward] total_cost_ms=6088.549187775138 migration_delay_ms=6085.93107302262 tearing_delay_ms=0.58 sla_penalty_ms=0.0 access_latency_ms=2.027105227329716 reward(clipped)=-6088.549187775138 REWARD_CLIP_MIN=-10000.0
[HYBRID_SAC_DBG reward] total_cost_ms=2610.2032967848795 migration_delay_ms=2555.723320695299 tearing_delay_ms=50.42 sla_penalty_ms=0.0 access_latency_ms=2.027105227329716 reward(clipped)=-2610.2032967848795 REWARD_CLIP_MIN=-10000.0
```

**B）微脚本下的动作采样（均匀 softmax 时采样到 2 / 1 / 0）：**

```text
[HYBRID_SAC_DBG action] logits_raw=[-5.4554183e-10  1.6606618e-09 -4.0260391e-09] logits_masked=[-5.4554183e-10  1.6606618e-09 -4.0260391e-09] softmax_probs=[0.33333334 0.33333334 0.33333334] mask=[True, True, True] sampled_action=2
[HYBRID_SAC_DBG action] ... softmax_probs=[0.33333334 0.33333334 0.33333334] mask=[True, True, True] sampled_action=1
[HYBRID_SAC_DBG action] ... softmax_probs=[0.33333334 0.33333334 0.33333334] mask=[True, True, True] sampled_action=0
```

**C）端到端汇总（迁移非零）：**

```text
  Hybrid SAC:
    - Migrations: 2448 -> 2664
```

---

## 四、产物与复现提示

- 一次完整 smoke 的终端输出曾写入项目根目录 **`smoke_hybrid_sac_log.txt`**（与 `Tee-Object` 行为有关；若需与 Cursor 终端文件对照，以当时会话的 `terminals/*.txt` 为准）。  
- 复现「动作埋点」：`HYBRID_SAC_DEBUG_STEPS` 在**启动进程前**设置；若要在 **`run_comparison` 流水线内**看到 `sampled_action`，需 **`num_epochs >= 2`**（存在非最后一轮的训练轮）或单独写微脚本调用 `sample_action`。

---

*若需恢复更早版本的 `test.md`（例如阶段一/二数据管线长文），请从 Git 历史中检出。*
