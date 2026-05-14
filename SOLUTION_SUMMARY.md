# 🎯 GAT-MARL 超参数调整总结报告

## 问题排查结果

### Q1: dense_distance_bonus 为何是 0？

**答**: Proactive 模式下 `total_migrations = 0`，所以没有机会计算距离奖励
- `_dense_distance_bonuses()` 只在 `old_server != new_server` 时才给分
- 所以没有迁移 = 没有奖励 = 恶性循环

**根本原因**: 
```
shared_reward + distance_bonus (0.0) - local_penalty 
< 
stay_action 的安全收益
```

---

## ✅ 实施的修改

### 修改 1: Reactive 模式硬约束 【Reactive 减搬迁】

**文件**: `algorithms/marl_gat.py` 第 258-262 行

```diff
- max_lambda_migration = 0.15 if use_proactive else 0.20
+ max_lambda_migration = 0.15 if use_proactive else 0.25  ⬆️ +25%

- max_lambda_split = 0.05 if use_proactive else 0.08  
+ max_lambda_split = 0.05 if use_proactive else 0.10  ⬆️ +25%
```

**效果**: 迁移成本惩罚↑，抑制"一次搬多个节点"的贪心行为


### 修改 2: Proactive 模式硬激励 【Proactive 多靠近】

**文件**: `core/marl_reward.py` 第 192-204 行

```diff
def calculate_marl_rewards(
    ...
+   dense_distance_bonus_max=None,  # 新增参数
):
+   if dense_distance_bonus_max is None:
+       dense_distance_bonus_max = 8.0 if trigger_type == TRIGGER_PROACTIVE else 0.0
```

**文件**: `algorithms/marl_gat.py` 第 545 行

```diff
  shared_reward, agent_rewards, details = calculate_marl_rewards(
      ...
      lambda_migration=lm,
      lambda_split=ls,
+     dense_distance_bonus_max=8.0 if use_proactive else 0.0,
  )
```

**核心逻辑**:
- Proactive: `distance_bonus_max` 4.0 → 8.0 (+100% 激励)
- 让 `agent_rewards = shared_reward + 8.0 * distance_bonus - penalty` 更有吸引力
- 打破"stay action"的局部最优陷阱


### 修改 3: 增强日志 【可观测性】

**文件**: `algorithms/marl_gat.py` 第 263 行

```python
print(f"... Lambda: migration={max_lambda_migration:.3f}, split={max_lambda_split:.3f}")
```

---

## 📊 预期效果对比

| 指标 | 修改前 | 修改后期望 | 目标 |
|------|--------|----------|------|
| **Proactive Migrations** | 0 | > 100 | ✓ 有迁移 |
| **Proactive dense_distance_bonus_sum** | 0.0 | > 100.0 | ✓ 有奖励 |
| **Proactive stay_action_ratio** | 1.0 | < 0.5 | ✓ 敢迁移 |
| **Reactive Migrations** | 13432 | < 10000 | ✓ 少搬点 |
| **Reactive lambda_migration** | 0.20 | 0.25 | ✓ 约束强 |

---

## 🚀 运行新实验

### 步骤 1: 清理旧检查点（重新训练）
```bash
cd d:\Migration\Migrate-main
rm -r checkpoints/marl_gat_*.pth
```

### 步骤 2: 启动新一轮实验
```bash
python run_medium_validation_cov50.py
```

**预计耗时**: ~65-70 分钟（同前次）

### 步骤 3: 检查结果
```bash
# 查看 Proactive 关键指标
cat experiments/medium_validation_*/result.md

# 查看完整统计数据
cat experiments/medium_validation_*/results.json | jq '.train.proactive.GAT-MARL'
```

---

## 📈 关键监控点

### 在 results.json 中验证

**Proactive 模式** (训练段):
```json
"GAT-MARL": {
  "total_migrations": XXX,          // 期望: > 0 (原: 0)
  "dense_distance_bonus_sum": YYY,   // 期望: > 0 (原: 0.0)
  "stay_action_ratio": 0.ZZZ,        // 期望: < 1.0 (原: 1.0)
  "lambda_migration": 0.150,         // 保持
  "lambda_split": 0.050              // 保持
}
```

**Reactive 模式** (训练段):
```json
"GAT-MARL": {
  "total_migrations": XXX,          // 期望: < 13432 (原: 13432)
  "lambda_migration": 0.250,         // 新值
  "lambda_split": 0.100              // 新值
}
```

---

## 💡 调整思路

如果新实验结果仍未达到预期:

### 方案 A: 增强 Proactive 激励（激进）
```python
# core/marl_reward.py 或 marl_gat.py
dense_distance_bonus_max = 10.0  # 从 8.0 升级
或
dense_distance_bonus_max = 12.0  # 更激进
```

### 方案 B: 调整 Proactive lambda 值（谨慎）
```python
# algorithms/marl_gat.py
max_lambda_migration = 0.10 if use_proactive else 0.25  # 从 0.15 降低
max_lambda_split = 0.03 if use_proactive else 0.10      # 从 0.05 降低
```

### 方案 C: 增加 Proactive 训练轮数
```bash
export MEDIUM_VALIDATION_MARL_EPOCHS=8  # 从 4 升级
python run_medium_validation_cov50.py
```

---

## 📝 文件修改清单

| 文件 | 修改行数 | 修改内容 |
|-----|--------|--------|
| `algorithms/marl_gat.py` | 258-262 | 硬编码 Reactive lambda 值 |
| `algorithms/marl_gat.py` | 263 | 增强日志输出 |
| `algorithms/marl_gat.py` | 545 | 传递 dense_distance_bonus_max |
| `core/marl_reward.py` | 192-204 | 添加 dense_distance_bonus_max 参数 |

---

## ✨ 总结

修改前后的核心差异：

```
【修改前】Proactive 模式
agent_rewards = shared_reward + 4.0 * 0 - small_penalty
            = shared_reward - small_penalty
            < 0（保持 stay）

【修改后】Proactive 模式  
agent_rewards = shared_reward + 8.0 * distance_bonus - small_penalty
            ≈ shared_reward + 8.0 * positive_value
            > shared_reward（激励迁移！）
```

同时，Reactive 模式的约束被强化，应该减少不必要的迁移。

**核心理念**: 用 `dense_distance_bonus` 直接奖励"靠近用户"这个策略本身，而不是依赖 `shared_reward` 的间接激励。
