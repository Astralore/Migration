# Hybrid SAC 工程化改造指令

## 角色设定

你是一位顶级 AI 系统工程专家。目前我们的 Hybrid SAC 算法在准确性上已经完美收敛（v3.7 JIT 机制）。现在我们需要将其工程化，重点证明其在"在线推理阶段（Online Inference）"的计算时延（Latency）优于传统的启发式 SA 算法，并且具备强大的未知数据泛化能力。

## 核心任务

请对 `run_comparison.py`, `algorithms/hybrid_sac.py` 及 `core/data_loader.py` 进行综合重构，引入数据切分、模型持久化、推理模式以及高精度时延探针，并将完整的实验结果输出到 `result.md` 文件中。

---

## 当前代码结构说明

**重要**：当前 `hybrid_sac.py` 采用**函数式实现**，不存在 `HybridSAC` 类：

```python
# 当前结构
def run_hybrid_sac_microservice(df, servers_df, predictor, proactive, num_epochs=6):
    # 网络在函数内部创建
    gat_network = TriggerAwareGAT(...)
    actor = SACDiscreteActor(...)
    critic = SACDiscreteCritic(...)      # 内含 q1_net, q2_net
    target_critic = SACDiscreteCritic(...)
    log_alpha = torch.tensor(...)
    
    # 训练循环
    for epoch in range(num_epochs):
        ...
    
    return results_dict  # 当前只返回指标，不返回网络
```

---

## 核心修改指令

### 任务 1：修改数据加载模块，支持索引范围切分

**文件**：`core/data_loader.py`

修改 `load_data()` 函数签名，新增 `start_index` 和 `end_index` 参数：

```python
def load_data(file_path=None, sample_fraction=1.0, start_index=None, end_index=None):
    """
    Load taxi trajectory CSV.
    
    Parameters
    ----------
    start_index : int, optional
        起始索引（包含）
    end_index : int, optional
        结束索引（不包含）
    
    如果同时指定 start_index 和 end_index，则截取 df[start_index:end_index]
    """
    ...
    # 原有逻辑保持不变（读取、清洗、排序）
    
    # 新增：按索引范围截取（在 chunk_size 逻辑之前）
    if start_index is not None and end_index is not None:
        df = df.iloc[start_index:end_index].reset_index(drop=True)
        print(f"  Sliced data: [{start_index}:{end_index}], {len(df):,} records")
    
    # 原有的 chunk_size 逻辑可以删除或保留作为备用
    
    return df
```

---

### 任务 2：定义全局模式开关与数据切分配置

**文件**：`run_comparison.py`

在文件顶部添加配置：

```python
# =============================================================================
# 工程化配置
# =============================================================================
INFERENCE_MODE = False  # False=训练模式, True=推理模式

# 数据切分配置
TRAIN_START_INDEX = 0
TRAIN_END_INDEX = 10000      # 训练数据: [0, 10000)
TEST_START_INDEX = 10000
TEST_END_INDEX = 15000       # 测试数据: [10000, 15000)

# 权重保存路径
CHECKPOINT_DIR = "checkpoints"
SAC_CHECKPOINT_PROACTIVE = "checkpoints/sac_proactive.pth"
SAC_CHECKPOINT_REACTIVE = "checkpoints/sac_reactive.pth"
```

---

### 任务 3：增加模型权重持久化函数

**文件**：`algorithms/hybrid_sac.py`

在文件末尾新增两个独立函数（不是类方法）：

```python
def save_sac_weights(filepath, gat_network, actor, critic, target_critic, log_alpha):
    """
    保存 SAC 所有网络权重到单个 .pth 文件。
    
    Parameters
    ----------
    filepath : str
        保存路径，如 "checkpoints/sac_proactive.pth"
    gat_network : TriggerAwareGAT
    actor : SACDiscreteActor
    critic : SACDiscreteCritic
    target_critic : SACDiscreteCritic
    log_alpha : torch.Tensor
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'gat_network': gat_network.state_dict(),
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
        'target_critic': target_critic.state_dict(),
        'log_alpha': log_alpha.detach().cpu(),
    }
    torch.save(checkpoint, filepath)
    print(f"  [SAVE] Weights saved to {filepath}")


def load_sac_weights(filepath, gat_network, actor, critic, target_critic, device):
    """
    从 .pth 文件加载 SAC 网络权重。
    
    Returns
    -------
    log_alpha : torch.Tensor
        加载的 alpha 参数
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    gat_network.load_state_dict(checkpoint['gat_network'])
    actor.load_state_dict(checkpoint['actor'])
    critic.load_state_dict(checkpoint['critic'])
    target_critic.load_state_dict(checkpoint['target_critic'])
    log_alpha = checkpoint['log_alpha'].to(device)
    
    print(f"  [LOAD] Weights loaded from {filepath}")
    return log_alpha
```

---

### 任务 4：修改 SAC 主函数，支持训练/推理模式

**文件**：`algorithms/hybrid_sac.py`

修改 `run_hybrid_sac_microservice()` 函数签名和逻辑：

```python
def run_hybrid_sac_microservice(
    df, servers_df, predictor=None, proactive=False, num_epochs=6,
    inference_mode=False,           # 新增：是否为推理模式
    checkpoint_path=None,           # 新增：权重文件路径（推理时加载）
    save_checkpoint_path=None,      # 新增：训练后保存路径
):
    """
    Hybrid SAC Microservice Migration.
    
    Parameters
    ----------
    inference_mode : bool
        True: 跳过训练，只运行 1 个推理 epoch
        False: 正常训练 + 评估
    checkpoint_path : str
        推理模式下，从此路径加载权重
    save_checkpoint_path : str
        训练模式下，训练完成后保存权重到此路径
    """
    ...
    
    # 网络初始化（保持不变）
    gat_network = TriggerAwareGAT(...).to(device)
    actor = SACDiscreteActor(...).to(device)
    critic = SACDiscreteCritic(...).to(device)
    target_critic = SACDiscreteCritic(...).to(device)
    
    # === 推理模式：加载权重，跳过训练 ===
    if inference_mode:
        if checkpoint_path is None:
            raise ValueError("inference_mode=True 但未提供 checkpoint_path")
        log_alpha = load_sac_weights(checkpoint_path, gat_network, actor, critic, target_critic, device)
        num_epochs = 1  # 只运行 1 个推理 epoch
        print(f"  [INFERENCE MODE] Loaded weights, running 1 eval epoch only")
        
        # ⚠️ 关键：推理模式下强制关闭 BC（行为克隆）
        # 覆盖 bc_prob_schedule，确保不会执行 SA 模仿
        bc_prob_schedule = [0.0]  # 只有 1 个 epoch，BC 概率为 0
    
    # === 训练循环 ===
    for epoch in range(num_epochs):
        # 推理模式：始终是评估 epoch
        if inference_mode:
            is_eval_epoch = True
            current_bc_prob = 0.0  # ⚠️ 强制关闭 BC
        else:
            is_eval_epoch = (epoch == num_epochs - 1)
            current_bc_prob = bc_prob_schedule[epoch] if epoch < len(bc_prob_schedule) else 0.0
        
        # ... 现有逻辑 ...
        
        # 在动作选择部分，使用 current_bc_prob 替代原来的 bc_prob_schedule[epoch]
        # if random.random() < current_bc_prob:
        #     action = 1  # BC: Follow SA
        # else:
        #     action = actor.get_action_deterministic(...)  # 推理用 deterministic
    
    # === 训练模式：保存权重 ===
    if not inference_mode and save_checkpoint_path:
        save_sac_weights(save_checkpoint_path, gat_network, actor, critic, target_critic, log_alpha)
    
    return results
```

**⚠️ 关键点**：推理模式下必须确保：
1. `num_epochs = 1`
2. `is_eval_epoch = True`
3. `current_bc_prob = 0.0`（完全关闭 SA 模仿）
4. 使用 `actor.get_action_deterministic()` 而非 `sample_action()`

---

### 任务 5：埋入高精度决策时延探针

**文件**：`algorithms/hybrid_sac.py` 和 `algorithms/sa.py`

#### 5.1 SAC 时延探针

在 `run_hybrid_sac_microservice()` 的评估/推理循环中：

```python
import time

# 在函数开头初始化
total_decision_time = 0.0
decision_count_for_latency = 0

# 在评估/推理循环内
if is_eval_epoch:
    # ⚠️ 关键：将整个决策过程包裹在 torch.no_grad() 和计时器中
    t_start = time.perf_counter()
    
    # === 纯决策时间开始（包含 GAT + 所有节点的 Actor 推理）===
    with torch.no_grad():
        # 1. GAT 前向传播
        embeddings = gat_network(node_feat_t, adj_t, trigger_t)
        
        # 2. 所有节点的 Actor 决策
        for ms_node in sorted_nodes:
            node_idx = node_to_idx[ms_node]
            node_emb = embeddings[node_idx]
            node_sa_prior = sa_prior_t[node_idx]
            
            # 使用 deterministic 动作（argmax）
            action = actor.get_action_deterministic(node_emb, node_sa_prior)
            
            # 执行动作（更新 taxi_dag_assignments）
            if action == 0:
                target_server = current_node_server
            elif action == 1:
                target_server = sa_proposed_server
            else:
                target_server = nearest_server_id
            taxi_dag_assignments[taxi_id][ms_node] = target_server
    # === 纯决策时间结束 ===
    
    t_end = time.perf_counter()
    total_decision_time += (t_end - t_start)
    decision_count_for_latency += 1

# 返回结果中添加时延信息
return {
    ...
    'total_decision_time': total_decision_time,
    'decision_count_for_latency': decision_count_for_latency,
    'avg_decision_time_ms': (total_decision_time / decision_count_for_latency * 1000) if decision_count_for_latency > 0 else 0,
}
```

**⚠️ 关键点**：`torch.no_grad()` 必须包裹**整个决策过程**（GAT + 所有节点的 Actor），而不仅仅是 GAT。这样才能消除计算图开销，得到真实的推理时延。

#### 5.2 SA 时延探针

在 `algorithms/sa.py` 的 `run_sa_microservice_fair()` 中：

```python
import time

# 在函数开头初始化
total_decision_time = 0.0
decision_count_for_latency = 0

# 在每次调用 SA 时计时
t_start = time.perf_counter()
sa_proposal, sa_cost = microservice_simulated_annealing(
    taxi_id, dag_info,
    taxi_dag_assignments[taxi_id],
    candidates,
    user_location=(current_lat, current_lon),
    servers_info=servers_info,
    previous_assignments=old_assignments,
    predicted_locations=predicted_locations,
    trigger_type=trigger_type,
)
t_end = time.perf_counter()

total_decision_time += (t_end - t_start)
decision_count_for_latency += 1

# 返回结果中添加时延信息
return {
    ...
    'total_decision_time': total_decision_time,
    'decision_count_for_latency': decision_count_for_latency,
    'avg_decision_time_ms': (total_decision_time / decision_count_for_latency * 1000) if decision_count_for_latency > 0 else 0,
}
```

---

### 任务 6：修改主入口，整合训练/推理流程

**文件**：`run_comparison.py`

```python
def main():
    import os
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 加载服务器数据和预测器（两种模式都需要）
    servers_df = pd.read_csv(DEFAULT_SERVER_PATH)
    
    if not INFERENCE_MODE:
        # =====================================================================
        # 训练模式：使用训练数据，训练后保存权重
        # =====================================================================
        print(f"[MODE] Training Mode - Data: [{TRAIN_START_INDEX}:{TRAIN_END_INDEX}]")
        df = load_data(DEFAULT_TAXI_PATH, start_index=TRAIN_START_INDEX, end_index=TRAIN_END_INDEX)
        
        # 训练预测器
        predictor = SimpleTrajectoryPredictor(forecast_horizon=FORECAST_HORIZON)
        predictor.fit(df)
        
        # --- Proactive 模式 ---
        proactive_results = {}
        
        # SA (无需训练)
        proactive_results["SA"] = run_sa_microservice_fair(
            df, servers_df, predictor=predictor, proactive=True
        )
        
        # DQN (需要训练)
        proactive_results["DQN"] = run_dqn_microservice_fair(
            df, servers_df, predictor=predictor, proactive=True
        )
        
        # Hybrid SAC (训练 + 保存权重)
        proactive_results["Hybrid SAC"] = run_hybrid_sac_microservice(
            df, servers_df, predictor=predictor, proactive=True, num_epochs=6,
            inference_mode=False,
            save_checkpoint_path=SAC_CHECKPOINT_PROACTIVE,
        )
        
        # --- Reactive 模式 ---
        reactive_results = {}
        
        reactive_results["SA"] = run_sa_microservice_fair(
            df, servers_df, predictor=predictor, proactive=False
        )
        
        reactive_results["DQN"] = run_dqn_microservice_fair(
            df, servers_df, predictor=predictor, proactive=False
        )
        
        reactive_results["Hybrid SAC"] = run_hybrid_sac_microservice(
            df, servers_df, predictor=predictor, proactive=False, num_epochs=6,
            inference_mode=False,
            save_checkpoint_path=SAC_CHECKPOINT_REACTIVE,
        )
        
    else:
        # =====================================================================
        # 推理模式：使用测试数据，加载权重，纯推理对比
        # =====================================================================
        print(f"[MODE] Inference Mode - Data: [{TEST_START_INDEX}:{TEST_END_INDEX}]")
        
        # ⚠️ 关键：用训练数据拟合预测器（保持历史学习到的移动模式）
        # 这样才是真正的"在线推理"场景：预测器基于历史数据，面对全新轨迹
        train_df = load_data(DEFAULT_TAXI_PATH, start_index=TRAIN_START_INDEX, end_index=TRAIN_END_INDEX)
        predictor = SimpleTrajectoryPredictor(forecast_horizon=FORECAST_HORIZON)
        predictor.fit(train_df)
        print(f"  Predictor fitted on TRAINING data [{TRAIN_START_INDEX}:{TRAIN_END_INDEX}]")
        
        # 加载测试数据进行评测（预测器从未见过这些数据）
        df = load_data(DEFAULT_TAXI_PATH, start_index=TEST_START_INDEX, end_index=TEST_END_INDEX)
        
        # --- Proactive 模式 ---
        proactive_results = {}
        
        # SA (启发式，每次都重新计算)
        proactive_results["SA"] = run_sa_microservice_fair(
            df, servers_df, predictor=predictor, proactive=True
        )
        
        # DQN (也需要在测试集上运行，但不训练)
        # 注意：DQN 如果没有 save/load 机制，这里会重新训练
        # 如果要公平对比，DQN 也应该添加 save/load 逻辑
        proactive_results["DQN"] = run_dqn_microservice_fair(
            df, servers_df, predictor=predictor, proactive=True
        )
        
        # Hybrid SAC (加载权重，纯推理)
        proactive_results["Hybrid SAC"] = run_hybrid_sac_microservice(
            df, servers_df, predictor=predictor, proactive=True,
            inference_mode=True,
            checkpoint_path=SAC_CHECKPOINT_PROACTIVE,
        )
        
        # --- Reactive 模式 ---
        reactive_results = {}
        
        reactive_results["SA"] = run_sa_microservice_fair(
            df, servers_df, predictor=predictor, proactive=False
        )
        
        reactive_results["DQN"] = run_dqn_microservice_fair(
            df, servers_df, predictor=predictor, proactive=False
        )
        
        reactive_results["Hybrid SAC"] = run_hybrid_sac_microservice(
            df, servers_df, predictor=predictor, proactive=False,
            inference_mode=True,
            checkpoint_path=SAC_CHECKPOINT_REACTIVE,
        )
    
    # 打印结果（含时延对比）
    print("\n" + "#" * 80)
    print("  Proactive Mode Results (with Latency)")
    print("#" * 80)
    print_ranking_with_latency(proactive_results)
    
    print("\n" + "#" * 80)
    print("  Reactive Mode Results (with Latency)")
    print("#" * 80)
    print_ranking_with_latency(reactive_results)
    
    # 自动生成实验报告
    generate_experiment_report(proactive_results, reactive_results, INFERENCE_MODE)
```

---

### 任务 7：修改排行榜打印，新增时延列

**文件**：`evaluation/metrics.py`

新增 `print_ranking_with_latency()` 函数：

```python
def print_ranking_with_latency(results):
    """
    打印排行榜，含决策时延。
    
    Parameters
    ----------
    results : dict
        算法名 -> 结果字典
    """
    # 计算 Score 并排序
    scored_results = []
    for name, res in results.items():
        # Score = Migrations + 0.5 * Violations（越低越好）
        score = res['total_migrations'] + 0.5 * res['total_violations']
        scored_results.append((name, res, score))
    
    # 按 Score 升序排序
    sorted_results = sorted(scored_results, key=lambda x: x[2])
    
    # 打印表头
    header = f"{'Rank':<6}{'Algorithm':<15}{'M':>8}{'V(real)':>10}{'D(proac)':>10}{'D(total)':>10}{'Latency(ms)':>14}{'Score':>10}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    # 打印每行
    for rank, (name, res, score) in enumerate(sorted_results, 1):
        latency_ms = res.get('avg_decision_time_ms', 0)
        proactive_decisions = res.get('proactive_decisions', 0)
        decision_count = res.get('decision_count', 0)
        
        print(f"{rank:<6}{name:<15}{res['total_migrations']:>8}{res['total_violations']:>10}"
              f"{proactive_decisions:>10}{decision_count:>10}"
              f"{latency_ms:>14.2f}{score:>10.1f}")
    
    print("-" * len(header))
```

---

### 任务 8：自动生成实验报告

**文件**：`run_comparison.py`

```python
from datetime import datetime

def generate_experiment_report(proactive_results, reactive_results, is_inference_mode):
    """自动生成 result.md 实验报告。"""
    mode_str = "推理模式 (Inference)" if is_inference_mode else "训练模式 (Training)"
    data_range = f"[{TEST_START_INDEX}:{TEST_END_INDEX}]" if is_inference_mode else f"[{TRAIN_START_INDEX}:{TRAIN_END_INDEX}]"
    
    report = f"""# 微服务迁移算法对比实验报告

**运行模式**：{mode_str}  
**数据范围**：{data_range}  
**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 一、Proactive 模式结果

| Algorithm | Migrations | Violations | Proactive Decisions | Avg Latency (ms) | Score |
|-----------|------------|------------|---------------------|------------------|-------|
"""
    
    for name, res in proactive_results.items():
        latency = res.get('avg_decision_time_ms', 0)
        score = res['total_migrations'] + 0.5 * res['total_violations']
        report += f"| {name} | {res['total_migrations']} | {res['total_violations']} | {res.get('proactive_decisions', 0)} | {latency:.2f} | {score:.1f} |\n"
    
    report += """
---

## 二、Reactive 模式结果

| Algorithm | Migrations | Violations | Avg Latency (ms) | Score |
|-----------|------------|------------|------------------|-------|
"""
    
    for name, res in reactive_results.items():
        latency = res.get('avg_decision_time_ms', 0)
        score = res['total_migrations'] + 0.5 * res['total_violations']
        report += f"| {name} | {res['total_migrations']} | {res['total_violations']} | {latency:.2f} | {score:.1f} |\n"
    
    report += """
---

## 三、时延对比分析

"""
    
    # 提取时延数据进行对比
    if 'Hybrid SAC' in proactive_results and 'SA' in proactive_results:
        sac_latency = proactive_results['Hybrid SAC'].get('avg_decision_time_ms', 0)
        sa_latency = proactive_results['SA'].get('avg_decision_time_ms', 0)
        
        if sa_latency > 0:
            speedup = sa_latency / sac_latency if sac_latency > 0 else float('inf')
            report += f"- **Hybrid SAC 平均决策时延**: {sac_latency:.2f} ms\n"
            report += f"- **SA 平均决策时延**: {sa_latency:.2f} ms\n"
            report += f"- **加速比**: SAC 比 SA 快 **{speedup:.1f}x**\n"
    
    report += "\n---\n\n*报告自动生成*\n"
    
    with open("result.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n  [REPORT] Experiment report saved to result.md")
```

---

## 验收要求

1. **第一步**：设置 `INFERENCE_MODE = False`，运行 `python run_comparison.py`
   - 训练 SAC（Proactive + Reactive）
   - 生成权重文件 `checkpoints/sac_proactive.pth` 和 `sac_reactive.pth`
   - 记录训练模式下的指标（含 SA、DQN、SAC 三者对比）

2. **第二步**：设置 `INFERENCE_MODE = True`，再次运行 `python run_comparison.py`
   - 加载权重，在全新的 [10000:15000] 测试数据上纯推理
   - 对比 SA、DQN、Hybrid SAC 三者的决策时延
   - 验证 SAC 在未见数据上的泛化能力

3. **最终输出**：将所有修改说明、训练/推理实验数据、时延对比分析覆盖写入 `result.md`

---

## 文件修改清单

| 文件 | 修改内容 |
|------|----------|
| `core/data_loader.py` | 新增 `start_index`, `end_index` 参数 |
| `algorithms/hybrid_sac.py` | 新增 `save_sac_weights()`, `load_sac_weights()` 函数；修改 `run_hybrid_sac_microservice()` 支持推理模式 + 强制关闭 BC；添加时延探针（`torch.no_grad()` 包裹完整决策） |
| `algorithms/sa.py` | 添加时延探针 |
| `run_comparison.py` | 添加模式开关、数据切分配置、训练/推理流程分支（含 SA + DQN + SAC 三者）、报告生成 |
| `evaluation/metrics.py` | 新增 `print_ranking_with_latency()` 函数（含 score 计算和排序） |

---

## 修复记录

| 问题 | 修复内容 |
|------|----------|
| 任务 7 `score` 未定义 | 在循环前计算 `score = migrations + 0.5 * violations` 并排序 |
| 任务 5.1 `torch.no_grad()` 作用域太小 | 扩大作用域，包裹 GAT + 所有节点的 Actor 决策 |
| 任务 6 推理模式遗漏 DQN | 补充 DQN 的推理调用，确保三者对比完整 |
| 任务 4 未强制关闭 BC | 推理模式下 `current_bc_prob = 0.0`，覆盖 `bc_prob_schedule` |
| 任务 6 预测器用测试数据拟合（数据泄露） | 改为用**训练数据**拟合预测器，测试数据仅用于评测 |
