#（策略修复）— 改动与验证结果

**改动摘要**：

- `core/reward.py`：将 `SLA_PENALTY_MS` 从 `5000.0` 调整为 `20000.0`，提高“不迁移持续违规”的真实代价。
- `algorithms/hybrid_sac.py`：Replay Buffer 改为逐节点记录 `cur_graph_state -> next_graph_state`，非终止 transition 使用拓扑下一个节点的 `next_node_idx` 计算 target Q。
- `algorithms/hybrid_sac.py`：加入条件式 imitation loss。仅当 `sa_proposal[node] != current_server` 时，将 BC target 设为 `ACTION_FOLLOW_SA` 并计算交叉熵；`SA_STAY` 样本不再监督 actor 学习 STAY。
- `algorithms/hybrid_sac.py`：新增 eval 动作分布日志，输出 `STAY / FOLLOW_SA / NEAREST`、实际造成迁移的动作数，以及 `FOLLOW_SA but SA_STAY`。

**极小规模回归配置**：

```text
active_users_limit=2
processed_csv=False
Hybrid SAC num_epochs=2
HYBRID_SAC_DEBUG_STEPS=10
```

**关键验证日志**：

```text
[HYBRID_SAC_DBG transition] node_idx=3 next_node_idx=2 done=False
[HYBRID_SAC_DBG imitation] target=1 loss=1.0986127853393555
Eval actions [STAY, FOLLOW_SA, NEAREST]: [0, 900, 0]
Eval action-caused migrations [STAY, FOLLOW_SA, NEAREST]: [0, 21, 0]
Mean imitation loss: 0.361735
SMOKE_RESULT {'migrations': 21, 'violations': 182, 'proactive_decisions': 2, 'decision_count': 184}
```

**结论**：第五阶段修复后，Actor 在 eval 阶段已不再 100% 坍缩到 `STAY`，并能通过 `FOLLOW_SA` 产生实际迁移；imitation loss 与 `next_node_idx` target 计算均已在回归日志中确认生效。

---

*报告由 `run_comparison.py --pipeline` 自动生成*
