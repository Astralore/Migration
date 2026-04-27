"""
SLA-based migration trigger for edge microservice migration.
QoS 与 reward 尺度解耦：仅用距离 (km) 与 calc_access_latency_ms (ms)。
"""

from core.geo import haversine_distance
from core.physics_utils import calc_access_latency_ms

# Reactive threshold: actual SLA violation (user perceives outage)
DISTANCE_THRESHOLD_KM = 15.0

# 与 command.md 一致：略小于「15km 对应延迟」，使 qos 可与 spatial 形成互补（仍 OR）
USER_SLA_TOLERANCE_MS = calc_access_latency_ms(DISTANCE_THRESHOLD_KM) * 0.99

# Proactive warning threshold: early-warning buffer zone
PROACTIVE_WARNING_KM = 5.0

# Trigger type constants
TRIGGER_REACTIVE = "REACTIVE"
TRIGGER_PROACTIVE = "PROACTIVE"


def check_sla_violation(
    user_lat, user_lon,
    gateway_server_lat, gateway_server_lon,
):
    """
    Reactive 触发：空间超阈 OR 接入延迟（ms）超 QoS 容限。
    不再依赖 current_dag_reward。
    """
    dist_km = haversine_distance(
        user_lat, user_lon,
        gateway_server_lat, gateway_server_lon,
    )
    spatial_violation = dist_km > DISTANCE_THRESHOLD_KM
    qos_violation = calc_access_latency_ms(dist_km) > USER_SLA_TOLERANCE_MS
    return spatial_violation or qos_violation


def check_proactive_sla_violation(
    user_lat, user_lon,
    gateway_server_lat, gateway_server_lon,
    predicted_locations=None,
):
    """Proactive：先 Reactive 检查，再前瞻轨迹。"""
    if check_sla_violation(
        user_lat, user_lon,
        gateway_server_lat, gateway_server_lon,
    ):
        return True

    if predicted_locations:
        for pred_lat, pred_lon in predicted_locations:
            future_dist = haversine_distance(
                pred_lat, pred_lon,
                gateway_server_lat,
                gateway_server_lon,
            )
            if future_dist > PROACTIVE_WARNING_KM:
                return True

    return False


def get_trigger_type(
    user_lat, user_lon,
    gateway_server_lat, gateway_server_lon,
    predicted_locations=None,
    proactive_enabled=False,
):
    """
    Determine the trigger type for migration decision.

    Returns
    -------
    str or None
        'REACTIVE'  — current state already violates SLA
        'PROACTIVE' — predicted future violation (preemptive)
        None        — no trigger needed
    """
    reactive_violation = check_sla_violation(
        user_lat, user_lon,
        gateway_server_lat, gateway_server_lon,
    )

    if reactive_violation:
        return TRIGGER_REACTIVE

    if proactive_enabled and predicted_locations:
        for pred_lat, pred_lon in predicted_locations:
            future_dist = haversine_distance(
                pred_lat, pred_lon,
                gateway_server_lat,
                gateway_server_lon,
            )
            if future_dist > PROACTIVE_WARNING_KM:
                return TRIGGER_PROACTIVE

    return None
