"""
Reward calculation for microservice DAG placement.
Implements asymmetric migration cost based on trigger_type:
  - PROACTIVE: background silent sync (low state penalty)
  - REACTIVE:  foreground blocking transfer (high state penalty)
"""

from core.geo import haversine_distance
from core.dag_utils import get_entry_nodes
from core.context import TRIGGER_REACTIVE, TRIGGER_PROACTIVE

BANDWIDTH_MBPS = 100.0
FUTURE_DECAY = 0.9
FUTURE_DIST_THRESHOLD = 15.0

# =============================================================================
# v3.7: JIT (Just-In-Time) Dynamic Migration Discount
# =============================================================================
# v3.6 Problem: "Premature Handover Trap"
# - Proactive trigger fired at 5km → agent immediately migrated
# - Fixed low cost made migration "too cheap" too early
# - Result: Proactive violations (108) > Reactive violations (97)
#
# v3.7 Solution: Dynamic discount based on continuous risk_ratio
# - risk_ratio = max_entry_dist / SLA_THRESHOLD (0.0 to 1.0)
# - At low risk (5km): migration cost ≈ REACTIVE (expensive, don't migrate)
# - At high risk (14km): migration cost ≈ PROACTIVE (cheap, migrate now)
# - Transition uses quadratic curve for "last-moment" urgency
#
# Formula: state_divisor = REACTIVE + (PROACTIVE - REACTIVE) * (risk_ratio ** 2)
# - risk=0.33 (5km):  divisor ≈ 5 + 495*0.11 = 59.5 (expensive)
# - risk=0.67 (10km): divisor ≈ 5 + 495*0.45 = 227.8 (moderate)
# - risk=0.93 (14km): divisor ≈ 5 + 495*0.86 = 430.7 (cheap)
# - risk=1.00 (15km): divisor = 500 (full proactive discount)
# =============================================================================
STATE_COST_PROACTIVE = 500.0   # background sync: state_mb / 500 (low cost at high risk)
STATE_COST_REACTIVE = 5.0      # foreground block: state_mb / 5 (high cost at low risk)

# v3.5: Reduced base migration multiplier (from 2.0 to 1.5)
MIGRATION_BASE_MULTIPLIER = 1.5  # 1.5x penalty for all migrations

# v3.5: SLA Violation "Death Penalty" - Non-linear punishment
SLA_DISTANCE_THRESHOLD = 15.0  # km - same as DISTANCE_THRESHOLD_KM in context.py
SLA_VIOLATION_MULTIPLIER = 15.0  # v3.8: 15x "Death Penalty" for SLA violations (was 5.0)

# v3.6: Cross-edge extreme distance — significant multiplier on comm cost (topology)
EDGE_TEAR_DISTANCE_KM = 20.0
EDGE_TEAR_MULTIPLIER = 3.0


def build_servers_info(servers_df):
    """Pre-build {server_id: (lat, lon)} lookup dict."""
    info = {}
    for _, row in servers_df.iterrows():
        info[row['edge_server_id']] = (row['latitude'], row['longitude'])
    return info


def calculate_microservice_reward(
    taxi_id, dag_info, current_assignments, previous_assignments,
    user_location, servers_info,
    alpha=1.0, beta=0.05, gamma=1.5,  # v3.5: gamma 2.0->1.5 (relax migration penalty)
    predicted_locations=None, delta=0.5,
    trigger_type=TRIGGER_REACTIVE,
):
    """
    Four-component weighted reward for a microservice DAG placement.

    Components
    ----------
    C_access   : access latency  (user -> entry-node servers)
    C_comm     : inter-service communication cost
    C_migrate  : state migration cost (ASYMMETRIC based on trigger_type)
    C_future   : predicted future topology violation penalty

    Asymmetric Migration Cost
    -------------------------
    For stateless nodes: cost = image_mb / 100
    For stateful nodes:
      - PROACTIVE trigger: cost = image_mb/100 + state_mb/1000 (background sync)
      - REACTIVE trigger:  cost = image_mb/100 + state_mb/10   (foreground block)

    Parameters
    ----------
    trigger_type : str
        'REACTIVE' or 'PROACTIVE', affects stateful node migration cost.

    Returns (reward, details_dict).
    """
    user_lat, user_lon = user_location

    # 1. Access latency: user -> entry-node servers
    # v3.5: With non-linear "Death Penalty" for SLA violations
    entry_nodes = get_entry_nodes(dag_info)
    access_latency = 0.0
    sla_violations = 0  # Track number of SLA violations
    
    for node in entry_nodes:
        srv_id = current_assignments[node]
        srv_lat, srv_lon = servers_info[srv_id]
        node_latency = haversine_distance(user_lat, user_lon, srv_lat, srv_lon)
        
        # v3.5: Non-linear SLA Violation "Death Penalty"
        # If distance exceeds SLA threshold (15km), apply 5x penalty multiplier
        # This tells SAC that SLA violations are HARD constraints, not soft preferences
        if node_latency > SLA_DISTANCE_THRESHOLD:
            # Death penalty: dramatically increase cost for violations
            access_latency += node_latency * SLA_VIOLATION_MULTIPLIER
            sla_violations += 1
        else:
            access_latency += node_latency

    # 2. Inter-service communication cost + v3.4 Topology Tearing Penalty
    max_traffic = max(dag_info['edges'].values()) if dag_info['edges'] else 0
    communication_cost = 0.0
    tearing_penalty = 0.0  # v3.4: Extra penalty for DAG fragmentation
    
    for (src, dst), traffic in dag_info['edges'].items():
        src_server = current_assignments[src]
        dst_server = current_assignments[dst]
        if src_server != dst_server:
            src_lat, src_lon = servers_info[src_server]
            dst_lat, dst_lon = servers_info[dst_server]
            dist = haversine_distance(src_lat, src_lon, dst_lat, dst_lon)
            norm_traffic = traffic / max_traffic if max_traffic > 0 else 0.0
            edge_term = norm_traffic * dist
            # v3.6: Blow up cost for extreme DAG splits (traffic-weighted)
            if dist > EDGE_TEAR_DISTANCE_KM:
                edge_term *= EDGE_TEAR_MULTIPLIER
            communication_cost += edge_term
            
            # v3.4: Additional tearing penalty based on distance
            # Penalize long-distance DAG splits more heavily
            if dist > 10.0:  # Only penalize if servers are >10km apart
                tearing_penalty += norm_traffic * (dist - 10.0) * 0.5

    # 3. State migration cost (ASYMMETRIC) - v3.7 JIT Dynamic Discount
    # ================================================================
    # v3.7 JIT Logic: Migration cost depends on HOW URGENT the situation is
    # 
    # Problem Solved: "Premature Handover Trap"
    # - Before: Proactive always used low cost → agent migrated too early
    # - Now: Cost depends on distance-to-SLA (risk_ratio)
    #
    # Physical intuition:
    # - At 5km (risk=0.33): "No rush, migration is expensive"
    #   → Prevents unnecessary early migration that tears DAG
    # - At 12km (risk=0.80): "Getting urgent, migration is affordable"
    #   → Allows timely migration before SLA violation
    # - At 15km (risk=1.00): "Critical, migration is cheap"
    #   → Full proactive discount to avoid violation
    #
    # The quadratic (risk_ratio ** 2) ensures discount kicks in late:
    # - Most of the discount happens in the last few kilometers
    # - This is the "Just-In-Time" behavior we want
    # ================================================================
    migration_cost = 0.0
    
    # Calculate current risk level from entry node distances
    max_entry_dist = 0.0
    for node in entry_nodes:
        srv_id = current_assignments[node]
        srv_lat, srv_lon = servers_info[srv_id]
        node_dist = haversine_distance(user_lat, user_lon, srv_lat, srv_lon)
        max_entry_dist = max(max_entry_dist, node_dist)
    
    risk_ratio = min(max_entry_dist / SLA_DISTANCE_THRESHOLD, 1.0)
    
    # v3.7: Dynamic state_divisor based on risk_ratio
    if trigger_type == TRIGGER_PROACTIVE:
        # JIT dynamic discount: quadratic curve for last-moment urgency
        # At low risk: divisor ≈ STATE_COST_REACTIVE (expensive)
        # At high risk: divisor → STATE_COST_PROACTIVE (cheap)
        state_divisor = STATE_COST_REACTIVE + (STATE_COST_PROACTIVE - STATE_COST_REACTIVE) * (risk_ratio ** 2)
    else:
        # REACTIVE: always expensive (no discount)
        state_divisor = STATE_COST_REACTIVE

    for node, node_props in dag_info['nodes'].items():
        if current_assignments[node] != previous_assignments[node]:
            # Base image cost with multiplier
            image_cost = node_props['image_mb'] / BANDWIDTH_MBPS * MIGRATION_BASE_MULTIPLIER
            if node_props['is_stateful']:
                # Stateful nodes: cost depends on state_divisor (JIT-adjusted)
                state_cost = node_props['state_mb'] / state_divisor
            else:
                state_cost = 0.0
            migration_cost += image_cost + state_cost

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

    # v3.4: Include tearing penalty in total cost
    total_cost = (alpha * access_latency
                  + beta * communication_cost
                  + gamma * migration_cost
                  + delta * future_penalty
                  + tearing_penalty)  # v3.4: Anti-tearing reward
    reward = -total_cost

    details = {
        'access_latency': access_latency,
        'communication_cost': communication_cost,
        'migration_cost': migration_cost,
        'future_penalty': future_penalty,
        'tearing_penalty': tearing_penalty,  # v3.4
        'sla_violations': sla_violations,    # v3.5: Track SLA violations
        'risk_ratio': risk_ratio,            # v3.7: JIT risk level
        'state_divisor': state_divisor,      # v3.7: JIT-adjusted cost divisor
        'total_cost': total_cost,
        'reward': reward,
        'trigger_type': trigger_type,
    }
    return reward, details
