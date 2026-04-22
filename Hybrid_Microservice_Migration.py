"""
Hybrid RL-Refined SA Microservice Migration
核心创新：SA 全局启发式搜索 → DQN 序列化逐节点审查
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from collections import deque
from tqdm import tqdm
import random
import copy
import time as _time
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from microservice_knowledge_base import MICROSERVICE_DAGS
from DQN_Microservice_Migration import (
    haversine_distance,
    find_k_nearest_servers,
    check_migration_criteria,
    get_entry_nodes,
    topological_sort,
    assign_dag_type,
    initialize_dag_assignment,
    build_servers_info,
    calculate_microservice_reward,
    optimize_model,
    load_data,
)
from SA_Microservice_Migration import microservice_simulated_annealing


# ============================================================
# Hybrid 状态构建函数 (per-node, 16 维)
# ============================================================

def build_hybrid_node_state(
    user_lat, user_lon, timestamp, gateway_dist,
    ms_node, dag_info, current_assignments, candidates,
    servers_info, entry_nodes_set,
    sa_proposed_server,
):
    """
    在原 14 维状态向量基础上追加 2 维 SA 先验建议特征，共 16 维。

    额外特征 (dim 15-16):
      - SA 建议服务器到出租车的归一化距离
      - SA 是否建议该节点保持原地 (binary)
    """
    node_info = dag_info['nodes'][ms_node]

    # ---- 全局特征 (6 维) ----
    feat_lat = user_lat / 90.0
    feat_lon = user_lon / 180.0
    feat_dist = min(gateway_dist / 50.0, 1.0)
    feat_hour = timestamp.hour / 24.0
    feat_weekday = timestamp.weekday() / 7.0
    feat_candidates = len(candidates) / 10.0

    # ---- 节点特征 (4 维) ----
    feat_image = node_info['image_mb'] / 200.0
    feat_state_mb = node_info['state_mb'] / 256.0
    feat_stateful = float(node_info['is_stateful'])

    node_traffic = 0.0
    max_traffic = 0.0
    for (src, dst), traffic in dag_info['edges'].items():
        if src == ms_node or dst == ms_node:
            node_traffic += traffic
        if traffic > max_traffic:
            max_traffic = traffic
    feat_traffic = node_traffic / max_traffic if max_traffic > 0 else 0.0

    # ---- 拓扑上下文 (4 维) ----
    current_server = current_assignments[ms_node]
    neighbors = set()
    for (src, dst) in dag_info['edges'].keys():
        if src == ms_node:
            neighbors.add(dst)
        elif dst == ms_node:
            neighbors.add(src)
    total_neighbors = len(neighbors)
    same_count = sum(1 for n in neighbors if current_assignments[n] == current_server)
    feat_same_ratio = same_count / total_neighbors if total_neighbors > 0 else 0.0

    srv_lat, srv_lon = servers_info[current_server]
    feat_node_dist = min(
        haversine_distance(user_lat, user_lon, srv_lat, srv_lon) / 50.0, 1.0
    )
    feat_is_entry = 1.0 if ms_node in entry_nodes_set else 0.0
    feat_dag_size = len(dag_info['nodes']) / 10.0

    # ---- SA 先验建议特征 (2 维) ----
    sa_lat, sa_lon = servers_info[sa_proposed_server]
    feat_sa_dist = min(
        haversine_distance(user_lat, user_lon, sa_lat, sa_lon) / 50.0, 1.0
    )
    feat_sa_stay = 1.0 if sa_proposed_server == current_server else 0.0

    return np.array([
        feat_lat, feat_lon, feat_dist, feat_hour, feat_weekday, feat_candidates,
        feat_image, feat_state_mb, feat_stateful, feat_traffic,
        feat_same_ratio, feat_node_dist, feat_is_entry, feat_dag_size,
        feat_sa_dist, feat_sa_stay,
    ], dtype=np.float32)


# ============================================================
# Hybrid DQN 网络 (input=16, action=3)
# ============================================================

class HybridMicroserviceDQN(nn.Module):
    """
    action_size=3:
      0 = Stay        (保持当前服务器)
      1 = Follow SA   (接受 SA 建议)
      2 = Nearest     (移到绝对最近候选)
    """
    def __init__(self, input_size=16, hidden_size=128, action_size=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# 主仿真决策循环
# ============================================================

def run_hybrid_microservice_fair(df, servers_df):
    """
    Hybrid RL-Refined SA 微服务迁移主仿真函数。

    流程：违规触发 → SA 生成全局草案 → DQN 逐节点审查（Stay / Follow SA / Nearest）

    Returns
    -------
    results : dict
    """
    servers_info = build_servers_info(servers_df)

    taxi_dag_type = {}
    taxi_dag_assignments = {}
    total_migrations = 0
    total_violations = 0
    total_reward_sum = 0.0

    # ---- Hybrid DQN 初始化 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    q_network = HybridMicroserviceDQN(input_size=16, hidden_size=128, action_size=3).to(device)
    target_network = HybridMicroserviceDQN(input_size=16, hidden_size=128, action_size=3).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)

    memory = deque(maxlen=5000)
    epsilon = 0.3
    epsilon_decay = 0.995
    epsilon_min = 0.01
    batch_size = 32
    gamma_td = 0.95
    target_update_freq = 50

    loss_history = []
    reward_history = []
    epsilon_history = []

    timestamps = sorted(df['date_time'].unique())
    df_grouped = df.groupby('date_time')

    decision_count = 0
    pbar = tqdm(total=len(timestamps), desc="Hybrid Microservice Migration")

    for timestamp in timestamps:
        current_rows = df_grouped.get_group(timestamp)
        for _, row in current_rows.iterrows():
            taxi_id = row['taxi_id']
            current_lat = row['latitude']
            current_lon = row['longitude']

            # ========== 首次出现：DAG 分配 + 全节点初始部署 ==========
            if taxi_id not in taxi_dag_assignments:
                nearest = find_k_nearest_servers(
                    current_lat, current_lon, servers_df, k=1
                )[0]
                chosen_dag = assign_dag_type()
                taxi_dag_type[taxi_id] = chosen_dag
                taxi_dag_assignments[taxi_id] = initialize_dag_assignment(
                    chosen_dag, nearest[0]
                )
                continue

            # ========== 违规检测 ==========
            dag_type = taxi_dag_type[taxi_id]
            dag_info = MICROSERVICE_DAGS[dag_type]
            entry_nodes = get_entry_nodes(dag_info)
            entry_nodes_set = set(entry_nodes)
            gateway_node = entry_nodes[0]

            gateway_server_id = taxi_dag_assignments[taxi_id][gateway_node]
            gw_lat, gw_lon = servers_info[gateway_server_id]
            gateway_dist = haversine_distance(
                current_lat, current_lon, gw_lat, gw_lon
            )

            if gateway_dist <= 15.0:
                continue

            total_violations += 1
            if not check_migration_criteria(row, gateway_dist):
                continue

            # ========== Phase A: SA 生成全局先验草案 ==========
            candidates = find_k_nearest_servers(
                current_lat, current_lon, servers_df, k=3
            )
            old_assignments = copy.copy(taxi_dag_assignments[taxi_id])

            sa_proposal, _sa_cost = microservice_simulated_annealing(
                taxi_id, dag_info,
                taxi_dag_assignments[taxi_id],
                candidates,
                user_location=(current_lat, current_lon),
                servers_info=servers_info,
                previous_assignments=old_assignments,
            )

            # ========== Phase B: DQN 逐节点审查 ==========
            sorted_nodes = topological_sort(dag_info)
            node_transitions = []
            nearest_server_id = candidates[0][0]

            for ms_node in sorted_nodes:
                sa_proposed_server = sa_proposal[ms_node]

                state = build_hybrid_node_state(
                    current_lat, current_lon, timestamp, gateway_dist,
                    ms_node, dag_info, taxi_dag_assignments[taxi_id],
                    candidates, servers_info, entry_nodes_set,
                    sa_proposed_server,
                )

                # ---- epsilon-greedy ----
                if random.random() < epsilon:
                    action = random.randint(0, 2)
                else:
                    with torch.no_grad():
                        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                        q_values = q_network(state_t)
                        action = q_values.argmax().item()

                # ---- 动作映射 ----
                current_node_server = taxi_dag_assignments[taxi_id][ms_node]
                if action == 0:
                    target_server = current_node_server
                elif action == 1:
                    target_server = sa_proposed_server
                else:
                    target_server = nearest_server_id

                taxi_dag_assignments[taxi_id][ms_node] = target_server
                node_transitions.append((state, action))

            # ========== DAG 级 Reward ==========
            reward, _details = calculate_microservice_reward(
                taxi_id, dag_info,
                taxi_dag_assignments[taxi_id], old_assignments,
                (current_lat, current_lon), servers_info,
            )
            total_reward_sum += reward
            reward_history.append(reward)

            # ========== 统计 M ==========
            nodes_migrated = sum(
                1 for n in sorted_nodes
                if old_assignments[n] != taxi_dag_assignments[taxi_id][n]
            )
            total_migrations += nodes_migrated

            # ========== 稀疏奖励 → Replay Buffer ==========
            for i, (s, a) in enumerate(node_transitions):
                is_last = (i == len(node_transitions) - 1)
                if is_last:
                    next_s = s
                    step_reward = reward
                    done = True
                else:
                    next_s = node_transitions[i + 1][0]
                    step_reward = 0.0
                    done = False
                memory.append((s, a, step_reward, next_s, done))

            # ========== DQN 训练步 ==========
            loss_val = optimize_model(
                memory, q_network, target_network, optimizer, device,
                batch_size=batch_size, gamma=gamma_td,
            )
            if loss_val is not None:
                loss_history.append(loss_val)
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay

            epsilon_history.append(epsilon)
            decision_count += 1

            if decision_count % target_update_freq == 0:
                target_network.load_state_dict(q_network.state_dict())

        pbar.update(1)
    pbar.close()

    return {
        'total_migrations': total_migrations,
        'total_violations': total_violations,
        'total_reward': total_reward_sum,
        'loss_history': loss_history,
        'reward_history': reward_history,
        'epsilon_history': epsilon_history,
        'decision_count': decision_count,
    }


# ============================================================
# 训练曲线绘制
# ============================================================

def plot_training_curves(results, save_path="hybrid_microservice_training.png"):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

    ax = axes[0]
    losses = results['loss_history']
    if losses:
        ax.plot(losses, alpha=0.3, color='steelblue', linewidth=0.5)
        window = min(50, max(1, len(losses) // 10))
        if len(losses) >= window:
            smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
            ax.plot(range(window - 1, len(losses)), smoothed,
                    color='steelblue', linewidth=2, label=f'MA-{window}')
        ax.set_ylabel('MSE Loss')
        ax.set_title('Hybrid DQN Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax = axes[1]
    rewards = results['reward_history']
    if rewards:
        ax.plot(rewards, alpha=0.3, color='coral', linewidth=0.5)
        window = min(50, max(1, len(rewards) // 10))
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
            ax.plot(range(window - 1, len(rewards)), smoothed,
                    color='coral', linewidth=2, label=f'MA-{window}')
        ax.set_ylabel('DAG-level Reward')
        ax.set_title('Per-Decision Reward (Hybrid)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax = axes[2]
    epsilons = results['epsilon_history']
    if epsilons:
        ax.plot(epsilons, color='seagreen', linewidth=1.5)
        ax.set_ylabel('Epsilon')
        ax.set_xlabel('Decision Step')
        ax.set_title('Exploration Rate Decay')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Training curves saved to: {save_path}")


# ============================================================
# 独立运行入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Hybrid RL-Refined SA Microservice Migration — Full Run")
    print("=" * 60)

    DATA_PATH = "Migrate-main/taxi_with_health_info.csv"
    SERVER_PATH = "Migrate-main/edge_server_locations.csv"
    CHUNK_SIZE = 10000

    df = load_data(DATA_PATH, sample_fraction=1.0, chunk_size=CHUNK_SIZE)
    servers_df = pd.read_csv(SERVER_PATH)
    print(f"  Edge servers loaded: {len(servers_df)}")

    print(f"\n{'=' * 60}")
    print("  Running Hybrid Microservice Migration ...")
    print(f"{'=' * 60}")
    t_start = _time.time()
    results = run_hybrid_microservice_fair(df, servers_df)
    elapsed = _time.time() - t_start

    M = results['total_migrations']
    V = results['total_violations']
    score = M + 0.5 * V
    R = results['total_reward']
    n_decisions = results['decision_count']
    n_losses = len(results['loss_history'])

    print(f"\n{'=' * 60}")
    print(f"  RESULTS — Hybrid RL-Refined SA Microservice Migration")
    print(f"{'=' * 60}")
    print(f"  Migrations (M)         : {M}")
    print(f"  Violations (V)         : {V}")
    print(f"  Score (M + 0.5*V)      : {score:.1f}")
    print(f"  Total Reward           : {R:.2f}")
    print(f"  Hybrid Decisions       : {n_decisions}")
    print(f"  Training Steps (loss)  : {n_losses}")
    if n_losses > 0:
        print(f"  Avg Loss (last 100)    : "
              f"{np.mean(results['loss_history'][-100:]):.4f}")
    if results['reward_history']:
        print(f"  Avg Reward (last 100)  : "
              f"{np.mean(results['reward_history'][-100:]):.4f}")
    print(f"  Elapsed Time           : {elapsed:.1f}s")
    print(f"{'=' * 60}")

    if n_losses > 0:
        plot_training_curves(results, save_path="hybrid_microservice_training.png")
    else:
        print("  No training steps recorded, skipping plot.")
