角色设定：
你是一位顶级的强化学习专家。基于最新的全量实验数据，系统已经彻底修复了策略坍缩，Proactive 触发极度健康，且综合成本超越了基线。但目前的 Actor 网络陷入了“过度模仿（Over-imitation）”，几乎 100% 依赖 `FOLLOW_SA` 动作，缺乏对 `NEAREST` 和 `STAY` 的独立探索。现进入 **【第七阶段：SAC 断奶与动作多样性激发】**。

**当前任务修改范围（绝对红线）**：
允许修改 `algorithms/hybrid_sac.py` 及环境交互中涉及动作掩码和 BC 计算的部分。严禁修改图神经网络（GAT）和底层的 Reward 物理公式！

**核心代码重构任务：**

**1. 引入 BC 权重退火机制（BC Weight Annealing）**
- **现状**：固定权重的 Imitation Loss 导致网络找到了“无脑跟随 SA”的捷径。
- **要求**：在 `hybrid_sac.py` 的主训练循环或 `optimize_sac` 中，使 `bc_weight` 随训练进度动态衰减。
- **公式参考**：设定初始 `bc_weight = 1.0`。如果当前处于第 `e` 个 Epoch（总计 `E` 个），则 `bc_weight = max(0.0, 1.0 - (e / (E * 0.5)))`。即在训练进行到一半时，模仿损失衰减为 0，让纯粹的 RL 彻底接管。

**2. 实施严格的 SA_STAY 动态掩码剥离**
- **现状**：大量决策表现为 `FOLLOW_SA but SA_STAY`，Actor 在 SA 建议不动时仍然输出动作 1 以逃避选择。
- **要求**：在 Actor 生成动作时的 **Action Mask（合法动作掩码）** 阶段增加硬约束。对于当前决策节点，如果其 `sa_proposal[node] == current_server`（即 SA 建议不迁移），则在当步的 Action Mask 中**强制将 `FOLLOW_SA`（动作 1）设为非法（Mask Out）**。
- **目的**：逼迫 Actor 在 SA 躺平时，必须独立承担责任，明确输出 `STAY`（0）或者去探索 `NEAREST`（2）。

**3. Q 值优势引导（可选但推荐的保护）**
- **要求**：在 `optimize_sac` 计算 Imitation Loss 时，增加一个 Q-Filter。只有当 Target Q 评估 `FOLLOW_SA` 的价值 大于等于 当前 Actor 偏好动作的价值时，才应用交叉熵损失。如果 Actor 已经发现了一个比 SA Q值更高的动作，则直接忽略 Imitation Loss。

**极速验证协议：**
1. 采用小规模验证配置（例如 5~10 辆车，`NUM_EPOCHS=4`）进行测试。
2. 重点观察 `result.md` 或终端日志中输出的 Actor Eval 分布：
   - 确认在 Epoch 3 或 4 时，`bc_weight` 已经衰减。
   - 确认动作分布中 `STAY` 和 `NEAREST` 的占比显著提升，`FOLLOW_SA but SA_STAY` 的数量大幅下降或归零。
3. 验证无误后，回复我：“第七阶段动作多样性激发完成！网络已成功断奶，请启动下一轮全量测试评估真实的 SAC 独立决策能力！”