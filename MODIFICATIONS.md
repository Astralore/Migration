# GAT-MARL 超参数调整与 Dense Distance Bonus 激励强化

**修改日期**: 2026-05-14  
**目标**: 解决 Proactive 模式下 Migrations = 0 的问题，强化 Reactive 模式的约束

## 修改总览

### 问题诊断
1. **Proactive 模式下 dense_distance_bonus_sum = 0 的根本原因**
   - `total_migrations = 0`，`stay_action_ratio = 1.0`
   - 在 `_dense_distance_bonuses` 中，只有迁移时才计算奖励
   - 没有迁移 → 没有距离改进 → 没有奖励 → 恶性循环

2. **为什么 Proactive 下全是 stay actions**
   - 共享奖励 + 距离奖励 < 迁移成本惩罚
   - 即使有理论上的激励，实际激励值 (4.0) 不足以打破 stay 的局部最优

3. **Reactive 模式下有大量迁移的问题**
   - lambda_migration = 0.20，lambda_split = 0.08（相对较低）
   - 共享奖励增长幅度大，足以压倒迁移成本

---

## 具体修改

### 1. **algorithms/marl_gat.py** - Reactive 模式超参数硬编码

**位置**: 第 258-262 行

**修改前**:
```python
if max_lambda_migration is None:
    max_lambda_migration = 0.15 if use_proactive else 0.20
if max_lambda_split is None:
    max_lambda_split = 0.05 if use_proactive else 0.08
```

**修改后**:
```python
if max_lambda_migration is None:
    max_lambda_migration = 0.15 if use_proactive else 0.25  # 硬调 Reactive 到 0.25
if max_lambda_split is None:
    max_lambda_split = 0.05 if use_proactive else 0.10  # 硬调 Reactive 到 0.10
```

**目的**: 
- ↑ lambda_migration: 0.20 → 0.25 (+25%)
- ↑ lambda_split: 0.08 → 0.10 (+25%)
- 增加 Reactive 模式下的迁移成本惩罚，抑制"冗余搬迁"

---

### 2. **algorithms/marl_gat.py** - 增强日志输出

**位置**: 第 263 行

**修改后**:
```python
print(f"  Device: {device}  |  Proactive: {use_proactive}  |  Model: CTDE-GAT-MARL  |  Lambda: migration={max_lambda_migration:.3f}, split={max_lambda_split:.3f}")
```

**目的**: 清楚地看到每次运行使用的实际超参数值

---

### 3. **algorithms/marl_gat.py** - 传递 dense_distance_bonus_max

**位置**: 第 533-545 行

**修改前**:
```python
shared_reward, agent_rewards, details = calculate_marl_rewards(
    taxi_id,
    dag_info,
    taxi_dag_assignments[taxi_id],
    old_assignments,
    (current_lat, current_lon),
    servers_info,
    predicted_locations=predicted_locations,
    trigger_type=trigger_type,
    lambda_migration=lm,
    lambda_split=ls,
)
```

**修改后**:
```python
shared_reward, agent_rewards, details = calculate_marl_rewards(
    taxi_id,
    dag_info,
    taxi_dag_assignments[taxi_id],
    old_assignments,
    (current_lat, current_lon),
    servers_info,
    predicted_locations=predicted_locations,
    trigger_type=trigger_type,
    lambda_migration=lm,
    lambda_split=ls,
    dense_distance_bonus_max=8.0 if use_proactive else 0.0,  # 增加 Proactive 激励
)
```

**目的**: 
- Proactive 模式: 8.0 (原 4.0，+100% 激励)
- Reactive 模式: 0.0 (保持无距离激励)

---

### 4. **core/marl_reward.py** - 添加 dense_distance_bonus_max 参数

**位置**: 第 179-204 行

**修改前**:
```python
def calculate_marl_rewards(
    taxi_id,
    dag_info,
    current_assignments,
    previous_assignments,
    user_location,
    servers_info,
    *,
    predicted_locations=None,
    trigger_type=TRIGGER_REACTIVE,
    lambda_migration=0.0,
    lambda_split=0.0,
    local_cost_scale_ms=1000.0,
    dense_distance_bonus=True,
):
    ...
    distance_bonuses = (
        _dense_distance_bonuses(
            dag_info,
            current_assignments,
            previous_assignments,
            user_location,
            servers_info,
            trigger_type,
        )
        if dense_distance_bonus else {node: 0.0 for node in dag_info["nodes"]}
    )
```

**修改后**:
```python
def calculate_marl_rewards(
    taxi_id,
    dag_info,
    current_assignments,
    previous_assignments,
    user_location,
    servers_info,
    *,
    predicted_locations=None,
    trigger_type=TRIGGER_REACTIVE,
    lambda_migration=0.0,
    lambda_split=0.0,
    local_cost_scale_ms=1000.0,
    dense_distance_bonus=True,
    dense_distance_bonus_max=None,  # 新增参数，支持自定义 bonus 上限
):
    """..."""
    if dense_distance_bonus_max is None:
        dense_distance_bonus_max = 8.0 if trigger_type == TRIGGER_PROACTIVE else 0.0  # 增加 Proactive bonus 到 8.0
    
    ...
    distance_bonuses = (
        _dense_distance_bonuses(
            dag_info,
            current_assignments,
            previous_assignments,
            user_location,
            servers_info,
            trigger_type,
            proactive_bonus_max=dense_distance_bonus_max if trigger_type == TRIGGER_PROACTIVE else 4.0,
            reactive_bonus_max=dense_distance_bonus_max if trigger_type != TRIGGER_PROACTIVE else 0.0,
        )
        if dense_distance_bonus else {node: 0.0 for node in dag_info["nodes"]}
    )
```

**目的**:
- 引入灵活的 `dense_distance_bonus_max` 参数
- Proactive: 默认 8.0，鼓励靠近用户的迁移
- Reactive: 默认 0.0，保持"尽量少搬"的策略

---

## 预期效果

### 对 Proactive 模式的影响
```
距离奖励激励 ↑ (4.0 → 8.0)
  ↓
agent_rewards = shared_reward + 8.0 * distance_bonus - local_penalty
  ↓
迁移变得更有吸引力
  ↓
智能体尝试靠近用户
  ↓
dense_distance_bonus_sum > 0 ✓
migrations > 0 ✓
```

### 对 Reactive 模式的影响
```
迁移成本权重 ↑ (lambda: 0.20/0.08 → 0.25/0.10)
  ↓
迁移惩罚更强
  ↓
抑制"一次性搬多个节点"的贪心行为
  ↓
total_migrations ↓（目标）
```

---

## 运行实验

```bash
# 重新训练 GAT-MARL（清空旧检查点）
rm -rf checkpoints/marl_gat_*.pth
python run_medium_validation_cov50.py
```

---

## 监控指标

在 `results.json` 中关注:

**Proactive 模式**:
- `dense_distance_bonus_sum`: 期望 > 0 (原来 = 0)
- `migrations`: 期望 > 100 (原来 = 0)
- `stay_action_ratio`: 期望 < 1.0 (原来 = 1.0)

**Reactive 模式**:
- `total_migrations`: 期望 < 13432 (原来 = 13432)
- `lambda_migration`: 实际值应显示 0.25
- `lambda_split`: 实际值应显示 0.10

---

## 备注

- 修改向后兼容：旧检查点可能不适用，建议清空重训
- `dense_distance_bonus_max` 是新参数，默认值已设定，无需改动其他调用
- 如果效果仍不理想，可考虑进一步调整 `proactive_bonus_max` 到 10.0-12.0
