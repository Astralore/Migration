from core.geo import haversine_distance
from core.dag_utils import get_entry_nodes

BANDWIDTH_MBPS = 100.0
FUTURE_DECAY = 0.9
FUTURE_DIST_THRESHOLD = 15.0


def build_servers_info(servers_df):
    """Pre-build {server_id: (lat, lon)} lookup dict."""
    info = {}
    for _, row in servers_df.iterrows():
        info[row['edge_server_id']] = (row['latitude'], row['longitude'])
    return info


def calculate_microservice_reward(
    taxi_id, dag_info, current_assignments, previous_assignments,
    user_location, servers_info,
    alpha=1.0, beta=0.01, gamma=1.0,
    predicted_locations=None, delta=0.5,
):
    """
    Four-component weighted reward for a microservice DAG placement.

    Components
    ----------
    C_access   : access latency  (user -> entry-node servers)
    C_comm     : inter-service communication cost
    C_migrate  : state migration cost
    C_future   : predicted future topology violation penalty (proactive)
                 Only computed when *predicted_locations* is provided.

    total_cost = alpha*C_access + beta*C_comm + gamma*C_migrate + delta*C_future
    reward     = -total_cost

    Parameters
    ----------
    predicted_locations : list of (lat, lon) or None
        H-step predicted user positions.  Each step h incurs a penalty
        weighted by FUTURE_DECAY^h if the distance to any entry-node server
        exceeds FUTURE_DIST_THRESHOLD.
    delta : float
        Weight for the future violation penalty (default 0.5).

    Returns (reward, details_dict).
    """
    user_lat, user_lon = user_location

    # 1. Access latency: user -> entry-node servers
    entry_nodes = get_entry_nodes(dag_info)
    access_latency = 0.0
    for node in entry_nodes:
        srv_id = current_assignments[node]
        srv_lat, srv_lon = servers_info[srv_id]
        access_latency += haversine_distance(user_lat, user_lon, srv_lat, srv_lon)

    # 2. Inter-service communication cost
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

    # 3. State migration cost
    migration_cost = 0.0
    for node, node_props in dag_info['nodes'].items():
        if current_assignments[node] != previous_assignments[node]:
            migration_cost += (node_props['image_mb'] + node_props['state_mb']) / BANDWIDTH_MBPS

    # 4. Future topology violation penalty (proactive)
    future_penalty = 0.0
    if predicted_locations:
        for h, (pred_lat, pred_lon) in enumerate(predicted_locations):
            w_h = FUTURE_DECAY ** h
            for node in entry_nodes:
                srv_id = current_assignments[node]
                srv_lat, srv_lon = servers_info[srv_id]
                future_dist = haversine_distance(pred_lat, pred_lon, srv_lat, srv_lon)
                if future_dist > FUTURE_DIST_THRESHOLD:
                    future_penalty += w_h * (future_dist - FUTURE_DIST_THRESHOLD)

    total_cost = (alpha * access_latency
                  + beta * communication_cost
                  + gamma * migration_cost
                  + delta * future_penalty)
    reward = -total_cost

    details = {
        'access_latency': access_latency,
        'communication_cost': communication_cost,
        'migration_cost': migration_cost,
        'future_penalty': future_penalty,
        'total_cost': total_cost,
        'reward': reward,
    }
    return reward, details
