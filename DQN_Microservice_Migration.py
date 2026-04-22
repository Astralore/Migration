"""
DQN Microservice Migration — 微服务 DAG 协同迁移版本
基于 Context_Integratoin.py 中的 run_dqn_fair() 改造
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from collections import deque, defaultdict
from tqdm import tqdm
import random
import math
import copy
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from microservice_knowledge_base import MICROSERVICE_DAGS

# ============================================================
# 复用的基础工具函数（来自 Context_Integratoin.py）
# ============================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def find_k_nearest_servers(lat, lon, servers_df, k=3):
    distances = []
    for _, server in servers_df.iterrows():
        dist = haversine_distance(lat, lon, server['latitude'], server['longitude'])
        distances.append((server['edge_server_id'], dist, server['latitude'], server['longitude']))
    distances.sort(key=lambda x: x[1])
    return distances[:k]


def check_migration_criteria(row, current_dist):
    if current_dist <= 15.0:
        return False
    age = row.get('Age', 0)
    physical_activity = row.get('Physical_Activity', '')
    cvd_risk_score = row.get('CVD_Risk_Score', 0)
    hypertension = row.get('Hypertension', '')
    diabetes = row.get('Diabetes', '')
    health_risk = (
        age > 75
        or str(physical_activity).strip().lower() == 'high'
        or cvd_risk_score > 70
        or str(hypertension).strip().lower() == 'yes'
        or str(diabetes).strip().lower() == 'yes'
        or current_dist > 30.0
    )
    return health_risk


# ============================================================
# 新增辅助函数
# ============================================================

def get_entry_nodes(dag_info):
    """找出 DAG 中所有入度 == 0 的节点（无入边 → 入口节点）"""
    all_nodes = set(dag_info['nodes'].keys())
    nodes_with_incoming = set()
    for (src, dst) in dag_info['edges'].keys():
        nodes_with_incoming.add(dst)
    return list(all_nodes - nodes_with_incoming)


def topological_sort(dag_info):
    """基于入度的 BFS 拓扑排序（Kahn's algorithm）"""
    all_nodes = list(dag_info['nodes'].keys())
    in_degree = {n: 0 for n in all_nodes}
    adj = {n: [] for n in all_nodes}
    for (src, dst) in dag_info['edges'].keys():
        adj[src].append(dst)
        in_degree[dst] += 1

    queue = deque([n for n in all_nodes if in_degree[n] == 0])
    sorted_nodes = []
    while queue:
        node = queue.popleft()
        sorted_nodes.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return sorted_nodes


# ============================================================
# 状态追踪初始化
# ============================================================

def assign_dag_type():
    """按概率为出租车分配一个 DAG 类型"""
    dag_names = list(MICROSERVICE_DAGS.keys())
    dag_probs = [MICROSERVICE_DAGS[d]['probability'] for d in dag_names]
    return np.random.choice(dag_names, p=dag_probs)


def initialize_dag_assignment(dag_type, nearest_server_id):
    """将所选 DAG 的所有微服务节点初始部署在离用户最近的同一个服务器上。"""
    dag_info = MICROSERVICE_DAGS[dag_type]
    return {node: nearest_server_id for node in dag_info['nodes']}


# ============================================================
# Reward 函数
# ============================================================

BANDWIDTH_MBPS = 100.0


def build_servers_info(servers_df):
    """将 servers_df 预构建为 {server_id: (lat, lon)} 字典"""
    info = {}
    for _, row in servers_df.iterrows():
        info[row['edge_server_id']] = (row['latitude'], row['longitude'])
    return info


def calculate_microservice_reward(
    taxi_id, dag_info, current_assignments, previous_assignments,
    user_location, servers_info,
    alpha=1.0, beta=0.01, gamma=1.0,
):
    """
    计算微服务 DAG 的三项加权奖励。

    Returns
    -------
    reward  : float  — 总加权惩罚的负数
    details : dict   — 包含三项明细的字典
    """
    user_lat, user_lon = user_location

    # ---- 1. 外部延迟惩罚 (access_latency) ----
    entry_nodes = get_entry_nodes(dag_info)
    access_latency = 0.0
    for node in entry_nodes:
        srv_id = current_assignments[node]
        srv_lat, srv_lon = servers_info[srv_id]
        access_latency += haversine_distance(user_lat, user_lon, srv_lat, srv_lon)

    # ---- 2. 图内通信惩罚 (communication_cost) ----
    max_traffic = max(dag_info['edges'].values()) if dag_info['edges'] else 0
    communication_cost = 0.0
    for (src, dst), traffic in dag_info['edges'].items():
        src_server = current_assignments[src]
        dst_server = current_assignments[dst]
        if src_server != dst_server:
            src_lat, src_lon = servers_info[src_server]
            dst_lat, dst_lon = servers_info[dst_server]
            dist = haversine_distance(src_lat, src_lon, dst_lat, dst_lon)
            norm_traffic = traffic / max_traffic if max_traffic > 0 else 0.0
            communication_cost += norm_traffic * dist

    # ---- 3. 状态迁移惩罚 (migration_cost) ----
    migration_cost = 0.0
    for node, node_props in dag_info['nodes'].items():
        if current_assignments[node] != previous_assignments[node]:
            migration_cost += (node_props['image_mb'] + node_props['state_mb']) / BANDWIDTH_MBPS

    # ---- 汇总 ----
    total_cost = alpha * access_latency + beta * communication_cost + gamma * migration_cost
    reward = -total_cost

    details = {
        'access_latency': access_latency,
        'communication_cost': communication_cost,
        'migration_cost': migration_cost,
        'total_cost': total_cost,
        'reward': reward,
    }
    return reward, details


# ============================================================
# 状态构建函数 (per-node, 14 维)
# ============================================================

def build_node_state(
    user_lat, user_lon, timestamp, gateway_dist,
    ms_node, dag_info, current_assignments, candidates,
    servers_info, entry_nodes_set,
):
    """为 DAG 中的单个微服务节点构建 14 维归一化状态向量。"""
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

    return np.array([
        feat_lat, feat_lon, feat_dist, feat_hour, feat_weekday, feat_candidates,
        feat_image, feat_state_mb, feat_stateful, feat_traffic,
        feat_same_ratio, feat_node_dist, feat_is_entry, feat_dag_size,
    ], dtype=np.float32)


# ============================================================
# DQN 网络
# ============================================================

class MicroserviceDQN(nn.Module):
    def __init__(self, input_size=14, hidden_size=128, action_size=4):
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
# DQN 经验回放训练函数
# ============================================================

def optimize_model(memory, policy_net, target_net, optimizer, device,
                   batch_size=32, gamma=0.95):
    """从 replay buffer 中采样一批经验进行一步 DQN 训练。"""
    if len(memory) < batch_size:
        return None

    batch = random.sample(memory, batch_size)
    states_b = torch.FloatTensor(np.array([e[0] for e in batch])).to(device)
    actions_b = torch.LongTensor([e[1] for e in batch]).to(device)
    rewards_b = torch.FloatTensor([e[2] for e in batch]).to(device)
    next_states_b = torch.FloatTensor(np.array([e[3] for e in batch])).to(device)
    dones_b = torch.BoolTensor([e[4] for e in batch]).to(device)

    current_q = policy_net(states_b).gather(1, actions_b.unsqueeze(1))
    with torch.no_grad():
        next_q = target_net(next_states_b).max(1)[0].unsqueeze(1)
    target_q = rewards_b.unsqueeze(1) + (gamma * next_q * ~dones_b.unsqueeze(1))

    loss = F.mse_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# ============================================================
# 数据加载
# ============================================================

def load_data(file_path, sample_fraction=1.0, chunk_size=None):
    """加载出租车轨迹+健康数据 CSV"""
    print(f"Loading data from {file_path} ...")
    df = pd.read_csv(file_path)

    if sample_fraction < 1.0:
        unique_taxis = df['taxi_id'].unique()
        sampled_taxis = np.random.choice(
            unique_taxis,
            size=int(len(unique_taxis) * sample_fraction),
            replace=False,
        )
        df = df[df['taxi_id'].isin(sampled_taxis)]
        print(f"  Sampled {len(sampled_taxis)} taxis.")

    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values(['taxi_id', 'date_time']).reset_index(drop=True)
    df = df.dropna(subset=['longitude', 'latitude'])

    if chunk_size is not None and chunk_size > 0:
        df = df.head(chunk_size)
        print(f"  Truncated to {chunk_size} rows.")

    print(f"  Final: {len(df):,} records, {df['taxi_id'].nunique():,} unique taxis.")
    return df


# ============================================================
# 主仿真决策循环
# ============================================================

def run_dqn_microservice_fair(df, servers_df, predictor=None):
    """
    微服务 DAG 协同迁移主仿真函数。

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    q_network = MicroserviceDQN(input_size=14, hidden_size=128, action_size=4).to(device)
    target_network = MicroserviceDQN(input_size=14, hidden_size=128, action_size=4).to(device)
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
    pbar = tqdm(total=len(timestamps), desc="DQN Microservice Migration")

    for timestamp in timestamps:
        current_rows = df_grouped.get_group(timestamp)
        for _, row in current_rows.iterrows():
            taxi_id = row['taxi_id']
            current_lat = row['latitude']
            current_lon = row['longitude']

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

            candidates = find_k_nearest_servers(
                current_lat, current_lon, servers_df, k=3
            )
            old_assignments = copy.copy(taxi_dag_assignments[taxi_id])
            sorted_nodes = topological_sort(dag_info)
            node_transitions = []

            for ms_node in sorted_nodes:
                state = build_node_state(
                    current_lat, current_lon, timestamp, gateway_dist,
                    ms_node, dag_info, taxi_dag_assignments[taxi_id],
                    candidates, servers_info, entry_nodes_set,
                )

                if random.random() < epsilon:
                    action = random.randint(0, 3)
                else:
                    with torch.no_grad():
                        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                        q_values = q_network(state_t)
                        action = q_values.argmax().item()

                current_node_server = taxi_dag_assignments[taxi_id][ms_node]
                if action == 0:
                    new_server = current_node_server
                elif action == 1 and len(candidates) >= 1:
                    new_server = candidates[0][0]
                elif action == 2 and len(candidates) >= 2:
                    new_server = candidates[1][0]
                elif action == 3 and len(candidates) >= 3:
                    new_server = candidates[2][0]
                else:
                    new_server = candidates[0][0] if candidates else current_node_server

                taxi_dag_assignments[taxi_id][ms_node] = new_server
                node_transitions.append((state, action))

            reward, _details = calculate_microservice_reward(
                taxi_id, dag_info,
                taxi_dag_assignments[taxi_id], old_assignments,
                (current_lat, current_lon), servers_info,
            )
            total_reward_sum += reward
            reward_history.append(reward)

            nodes_migrated = sum(
                1 for n in sorted_nodes
                if old_assignments[n] != taxi_dag_assignments[taxi_id][n]
            )
            total_migrations += nodes_migrated

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

def plot_training_curves(results, save_path="dqn_microservice_training.png"):
    """绘制 Loss / Reward / Epsilon 三合一训练曲线并保存"""
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
        ax.set_title('DQN Training Loss')
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
        ax.set_title('Per-Decision Reward')
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
    import time as _time

    print("=" * 60)
    print("  DQN Microservice DAG Migration — Full Run")
    print("=" * 60)

    DATA_PATH = "Migrate-main/taxi_with_health_info.csv"
    SERVER_PATH = "Migrate-main/edge_server_locations.csv"
    CHUNK_SIZE = 10000

    df = load_data(DATA_PATH, sample_fraction=1.0, chunk_size=CHUNK_SIZE)
    servers_df = pd.read_csv(SERVER_PATH)
    print(f"  Edge servers loaded: {len(servers_df)}")

    print(f"\n{'=' * 60}")
    print("  Running DQN Microservice Migration ...")
    print(f"{'=' * 60}")
    t_start = _time.time()
    results = run_dqn_microservice_fair(df, servers_df)
    elapsed = _time.time() - t_start

    M = results['total_migrations']
    V = results['total_violations']
    score = M + 0.5 * V
    R = results['total_reward']
    n_decisions = results['decision_count']
    n_losses = len(results['loss_history'])

    print(f"\n{'=' * 60}")
    print(f"  RESULTS — DQN Microservice DAG Migration")
    print(f"{'=' * 60}")
    print(f"  Migrations (M)         : {M}")
    print(f"  Violations (V)         : {V}")
    print(f"  Score (M + 0.5*V)      : {score:.1f}")
    print(f"  Total Reward           : {R:.2f}")
    print(f"  DAG Decisions          : {n_decisions}")
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
        plot_training_curves(results, save_path="dqn_microservice_training.png")
    else:
        print("  No training steps recorded, skipping plot.")
