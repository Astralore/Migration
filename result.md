# 微服务迁移算法对比实验报告（全量流水线）

**生成时间**：2026-04-27 17:44:49  
**流程**：启动前已删除 `sac_proactive.pth` / `sac_reactive.pth`（若存在）→ **训练** `[0:10000)` → 保存新权重 → **推理** `[10000:15000)` 加载新权重评测 Hybrid SAC。

**工程上下文（与指标相关）**：

- **物理与奖励**：`total_cost_ms`（接入/迁移/tearing/通信/future/SLA）与 `context` 触发解耦（Reactive 空间 + QoS）。
- **Hybrid SAC**：离散 Actor **Logits 动作掩码**（合法 NEAREST、全非法回退 FOLLOW_SA），训练 `num_epochs=6`。
- **性能路径**：`core/geo.py`、`context.py`、`reward.py`、`state_builder.py` **NumPy 向量化**（Haversine 批量、近邻 `argpartition` 等），降低仿真墙钟时间但不改变公式。

**墙钟时间**：训练阶段约 **6568 s**，推理阶段约 **104 s**。

---

## 一、训练段结果 `[0:10000)`

### Proactive（上表）/ Reactive（下表）

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Score |
|-----------|------------|------------|---------------------|------------------|-------|
| SA | 320 | 65 | 5893 | 2.71 | 352.5 |
| DQN | 1853 | 995 | 3799 | 0.00 | 2350.5 |
| Hybrid SAC | 3560 | 45 | 2889 | 2.33 | 3582.5 |

| Algorithm | Migrations | Violations | Avg Latency (ms) | Score |
|-----------|------------|------------|------------------|-------|
| SA | 286 | 550 | 5.35 | 561.0 |
| DQN | 945 | 748 | 0.00 | 1319.0 |
| Hybrid SAC | 1317 | 68 | 2.66 | 1351.0 |


---

## 二、测试段推理结果 `[10000:15000)`

*Hybrid SAC 使用训练段刚写入的 checkpoint；DQN/SA 无磁盘权重，在测试段上按既有脚本逻辑运行。*

### Proactive（上表）/ Reactive（下表）

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Score |
|-----------|------------|------------|---------------------|------------------|-------|
| SA | 116 | 1009 | 1978 | 7.54 | 620.5 |
| DQN | 578 | 1400 | 906 | 0.00 | 1278.0 |
| Hybrid SAC | 1761 | 1004 | 461 | 2.48 | 2263.0 |

| Algorithm | Migrations | Violations | Avg Latency (ms) | Score |
|-----------|------------|------------|------------------|-------|
| SA | 122 | 1009 | 5.05 | 626.5 |
| DQN | 521 | 1383 | 0.00 | 1212.5 |
| Hybrid SAC | 588 | 1006 | 2.44 | 1091.0 |


---

## 三、Hybrid SAC 泛化对比（训练 → 测试）

- **Proactive Violations**（训练段 → 测试段）: 45 → 1004

- **Proactive Migrations**（训练段 → 测试段）: 3560 → 1761

- **Reactive Violations**（训练段 → 测试段）: 68 → 1006

- **Reactive Migrations**（训练段 → 测试段）: 1317 → 588


**测试段决策时延（Hybrid SAC）**：

- Proactive：**2.48 ms**（训练段末次 eval 统计：**2.33 ms**）
- Reactive：**2.44 ms**（训练段：**2.66 ms**）

---

## 四、时延对比（测试段 Proactive：SAC vs SA）

- Hybrid SAC: **2.48 ms**；SA: **7.54 ms**；比值 SA/SAC ≈ **3.0x**

---

## 五、结果解读与工程影响摘要

- **SAC 权重**：本次流水线启动前已删除旧 `sac_*.pth`；训练结束后写入新文件，推理阶段 **仅加载本轮** 权重，与历史实验 **解耦**。
- **训练稳定性**：完整日志中 **未出现** `NaN` / `nan`（与 Actor Logits Mask、全非法回退等实现一致）。
- **Hybrid SAC 泛化**：训练段 Proactive **Violations=45**、Reactive **68**；测试段分别 **1004 / 1006**，与 SA/DQN 在测试段 **Violation 量级相近**（测试段轨迹更难 / 分布偏移）。Proactive 下 SAC **Migrations** 训练 **3560** → 测试 **1761**，仍高于 SA 的 **116**，说明策略较「激进」。
- **决策时延**：向量化与 GAT 路径下，Hybrid SAC 在训练末 eval 与测试段均维持 **约 2.3–2.7 ms**；测试段 Proactive 上 SA 约 **7.54 ms**，SAC 约 **3.0×** 更快（见第四节）。
- **DQN 说明**：当前脚本在测试段 **重新从随机初始化训练**（无磁盘 checkpoint），与 Hybrid SAC「加载训练段权重再 eval」**不对等**；若需公平 OOD 对比，后续可为 DQN 增加与 SAC 对称的 save/load。

---

*报告由 `run_comparison.py --pipeline` 自动生成*
