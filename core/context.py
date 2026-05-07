"""
SLA-based migration trigger for edge microservice migration.
QoS 与 reward 尺度解耦：仅用距离 (km) 与 calc_access_latency_ms (ms)。
"""

import numpy as np

from core.geo import haversine_distance
from core.physics_utils import calc_access_latency_ms

# Reactive threshold: actual SLA violation (user perceives outage)
DISTANCE_THRESHOLD_KM = 15.0

# 与 command.md 一致：略小于「15km 对应延迟」，使 qos 可与 spatial 形成互补（仍 OR）
USER_SLA_TOLERANCE_MS = calc_access_latency_ms(DISTANCE_THRESHOLD_KM) * 0.99

# 旧版距离缓冲（仅供 check_proactive_sla_violation 等兼容；get_trigger_type 已改 TTV）
PROACTIVE_WARNING_KM = 5.0

# C3：前瞻时间步长（秒）与保守迁移耗时兜底（秒），不扩展 get_trigger_type 形参
_DEFAULT_FORECAST_STEP_DT_SEC = 60.0
_ESTIMATED_MIGRATION_TIME_S_FALLBACK = 2.0

# Trigger type constants
TRIGGER_REACTIVE = "REACTIVE"
TRIGGER_PROACTIVE = "PROACTIVE"


def _future_distances_km_to_gateway(predicted_locations, gateway_server_lat, gateway_server_lon):
    """(H,) 距离向量；与逐点 haversine 数学等价，向量化广播。"""
    if not predicted_locations:
        return np.zeros(0, dtype=np.float64)
    arr = np.asarray(predicted_locations, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        arr = np.reshape(arr, (-1, 2))
    plats = arr[:, 0]
    plons = arr[:, 1]
    return np.asarray(
        haversine_distance(plats, plons, gateway_server_lat, gateway_server_lon),
        dtype=np.float64,
    )


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
        fd = _future_distances_km_to_gateway(
            predicted_locations, gateway_server_lat, gateway_server_lon,
        )
        if fd.size and bool(np.any(fd > PROACTIVE_WARNING_KM)):
            return True

    return False


def _ttv_seconds_to_sla_breach(fd, step_dt_sec):
    """首次达到 SLA 空间阈值的预测时间（秒）；达不到则 +inf。fd 为前瞻到 gateway 距离 (H,)。"""
    if fd.size == 0:
        return float("inf")
    dt = float(step_dt_sec) if float(step_dt_sec) > 0 else _DEFAULT_FORECAST_STEP_DT_SEC
    for h in range(fd.size):
        if fd[h] >= DISTANCE_THRESHOLD_KM:
            return (h + 1) * dt
    return float("inf")


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
        fd = _future_distances_km_to_gateway(
            predicted_locations, gateway_server_lat, gateway_server_lon,
        )
        ttv_s = _ttv_seconds_to_sla_breach(fd, _DEFAULT_FORECAST_STEP_DT_SEC)
        est_mig_s = _ESTIMATED_MIGRATION_TIME_S_FALLBACK
        if np.isfinite(ttv_s) and ttv_s <= est_mig_s + 1.0:
            return TRIGGER_PROACTIVE

    return None
