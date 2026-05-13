# 第八阶段 Hybrid SAC 修复记录

本次修改针对最新全量实验中出现的 `NEAREST` 动作坍缩：`FOLLOW_SA but SA_STAY` 已经为 0，但 Hybrid SAC 几乎不再使用 `FOLLOW_SA`，大量选择 `NEAREST`，导致 Proactive 推理违规和成本升高。

## 核心修改

- `algorithms/hybrid_sac.py`：修复 BC target 写入条件。`has_bc_target` 不再由 `used_bc_target` 决定；只要 SA 确实建议迁移，就写入 `FOLLOW_SA` 监督目标，避免 BC 概率衰减后 `FOLLOW_SA` 学习信号消失。
- `algorithms/hybrid_sac.py`：加入双轨 BC 权重。`FOLLOW_SA` 使用 `follow_sa_bc_scale=10.0`，轻量 `STAY` 监督使用 `stay_bc_scale=1.0`，避免重新坍缩到“不迁移”。
- `algorithms/hybrid_sac.py`：加入 `NEAREST` 软正则，仅在 `FOLLOW_SA` 合法且 SA 建议迁移时惩罚 `ACTION_NEAREST` 概率；该概率保留梯度，不使用 `.detach()`。
- `algorithms/hybrid_sac.py`：收紧 Q-Filter。启用条件改为更晚、更保守，并将 `q_filter_margin` 提高到 `0.5`，降低早期 critic 噪声阻断 SA 引导的风险。
- `algorithms/hybrid_sac.py`：增加 eval/inference 诊断指标，包括动作概率均值、Q 值均值、SA 建议迁移/不迁移场景下的动作分布，以及 `q_filter_blocked_ratio`。
- `run_medium_validation_cov50.py` 与 `run_full_pipeline_cov50.py`：结果汇总保留新增 SAC 诊断指标，方便后续判断是 actor logits 坍缩还是 critic Q 高估。

## 验证计划

下一步先运行中等规模 cov50 验证，结果单独保存，不覆盖旧全量实验目录和旧 checkpoint。重点检查：

- `eval_follow_sa_stay` 必须继续为 0。
- `FOLLOW_SA` 在 `eval_action_counts` 和 `eval_sa_migrate_action_counts` 中应恢复为非零。
- `NEAREST` 不应继续占绝对多数。
- `q_filter_blocked_ratio` 应低于 20%。
- Hybrid SAC 的 Proactive 推理违规和平均成本不应显著劣于 SA/DQN。
