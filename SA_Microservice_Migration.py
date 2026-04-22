"""
SA Microservice Migration — 拓扑感知模拟退火微服务 DAG 迁移基线
作为 DQN 微服务迁移的纯 SA 对比方法
"""

import numpy as np
import pandas as pd
import math
import copy
import random
import time as _time
from collections import deque
from tqdm import tqdm

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
    load_data,
    BANDWIDTH_MBPS,
)


# ============================================================
# 微服务级模拟退火
# ============================================================

def microservice_simulated_annealing(
    taxi_id, dag_info, current_assignments, candidates,
    user_location, servers_info,
    previous_assignments=None,
    temp=100.0, cooling_rate=0.95, max_iter=30,
):
    """
    对一个 DAG 的全部微服务节点执行模拟退火优化部署方案。

    每次迭代随机挑选 1 个节点，将其迁移到 candidates 中的随机服务器，
    用 calculate_microservice_reward 作为目标函数评估。

    Parameters
    ----------
    taxi_id             : 出租车 ID
    dag_info            : MICROSERVICE_DAGS[dag_type]
    current_assignments : {node: server_id} 当前部署方案（会被复制，不修改原始）
    candidates          : find_k_nearest_servers 返回的候选列表
    user_location       : (lat, lon)
    servers_info        : {server_id: (lat, lon)}
    previous_assignments: {node: server_id} 迁移前的基准（用于计算迁移成本）
    temp                : 初始温度
    cooling_rate        : 降温系数
    max_iter            : 最大迭代次数

    Returns
    -------
    best_assignments : dict — 最优部署方案
    best_cost        : float — 最优方案的 cost
    """
    if previous_assignments is None:
        previous_assignments = current_assignments

    candidate_server_ids = [c[0] for c in candidates]
    all_nodes = list(dag_info['nodes'].keys())

    # 初始解 & 初始 cost
    current_sol = dict(current_assignments)
    current_reward, _ = calculate_microservice_reward(
        taxi_id, dag_info, current_sol, previous_assignments,
        user_location, servers_info,
    )
    current_cost = -current_reward

    best_sol = dict(current_sol)
    best_cost = current_cost

    for _iteration in range(max_iter):
        # ---- 邻域扰动：随机选 1 个节点，换到随机候选服务器 ----
        node = random.choice(all_nodes)
        old_server = current_sol[node]

        other_servers = [s for s in candidate_server_ids if s != old_server]
        if not other_servers:
            temp *= cooling_rate
            continue
        new_server = random.choice(other_servers)

        neighbor_sol = dict(current_sol)
        neighbor_sol[node] = new_server

        neighbor_reward, _ = calculate_microservice_reward(
            taxi_id, dag_info, neighbor_sol, previous_assignments,
            user_location, servers_info,
        )
        neighbor_cost = -neighbor_reward

        # ---- 接受准则 ----
        delta = neighbor_cost - current_cost
        if delta < 0:
            accept = True
        else:
            accept = random.random() < math.exp(-delta / temp) if temp > 1e-10 else False

        if accept:
            current_sol = neighbor_sol
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_sol = dict(current_sol)
                best_cost = current_cost

        # ---- 降温 ----
        temp *= cooling_rate

    return best_sol, best_cost


# ============================================================
# 主仿真循环
# ============================================================

def run_sa_microservice_fair(df, servers_df):
    """
    模拟退火微服务 DAG 迁移主仿真函数。
    框架与 DQN 版本完全一致，仅决策部分替换为 SA。

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
    reward_history = []

    timestamps = sorted(df['date_time'].unique())
    df_grouped = df.groupby('date_time')

    decision_count = 0
    pbar = tqdm(total=len(timestamps), desc="SA Microservice Migration")

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

            # ========== 违规检测：入口网关距离 > 15km ==========
            dag_type = taxi_dag_type[taxi_id]
            dag_info = MICROSERVICE_DAGS[dag_type]
            entry_nodes = get_entry_nodes(dag_info)
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

            # ========== SA 决策 ==========
            candidates = find_k_nearest_servers(
                current_lat, current_lon, servers_df, k=3
            )
            old_assignments = copy.copy(taxi_dag_assignments[taxi_id])

            best_assignments, best_cost = microservice_simulated_annealing(
                taxi_id, dag_info,
                taxi_dag_assignments[taxi_id],
                candidates,
                user_location=(current_lat, current_lon),
                servers_info=servers_info,
                previous_assignments=old_assignments,
            )

            taxi_dag_assignments[taxi_id] = best_assignments

            # ========== 记录 reward ==========
            reward = -best_cost
            total_reward_sum += reward
            reward_history.append(reward)

            # ========== 统计 M：逐节点比较 old vs new ==========
            nodes_migrated = sum(
                1 for n in dag_info['nodes']
                if old_assignments[n] != best_assignments[n]
            )
            total_migrations += nodes_migrated
            decision_count += 1

        pbar.update(1)
    pbar.close()

    return {
        'total_migrations': total_migrations,
        'total_violations': total_violations,
        'total_reward': total_reward_sum,
        'reward_history': reward_history,
        'decision_count': decision_count,
    }


# ============================================================
# 独立运行入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  SA Microservice DAG Migration — Full Run")
    print("=" * 60)

    DATA_PATH = "Migrate-main/taxi_with_health_info.csv"
    SERVER_PATH = "Migrate-main/edge_server_locations.csv"
    CHUNK_SIZE = 10000

    df = load_data(DATA_PATH, sample_fraction=1.0, chunk_size=CHUNK_SIZE)
    servers_df = pd.read_csv(SERVER_PATH)
    print(f"  Edge servers loaded: {len(servers_df)}")

    print(f"\n{'=' * 60}")
    print("  Running SA Microservice Migration ...")
    print(f"{'=' * 60}")
    t_start = _time.time()
    results = run_sa_microservice_fair(df, servers_df)
    elapsed = _time.time() - t_start

    M = results['total_migrations']
    V = results['total_violations']
    score = M + 0.5 * V
    R = results['total_reward']
    n_decisions = results['decision_count']

    print(f"\n{'=' * 60}")
    print(f"  RESULTS — SA Microservice DAG Migration")
    print(f"{'=' * 60}")
    print(f"  Migrations (M)         : {M}")
    print(f"  Violations (V)         : {V}")
    print(f"  Score (M + 0.5*V)      : {score:.1f}")
    print(f"  Total Reward           : {R:.2f}")
    print(f"  SA Decisions           : {n_decisions}")
    if results['reward_history']:
        print(f"  Avg Reward (last 100)  : "
              f"{np.mean(results['reward_history'][-100:]):.4f}")
    print(f"  Elapsed Time           : {elapsed:.1f}s")
    print(f"{'=' * 60}")
