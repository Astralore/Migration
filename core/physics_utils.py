"""
Physical latency helpers (ms). Single source for propagation constants.
Used by core/reward.py and core/context.py — must not import from those modules.
"""

# 光在光纤中有效传播量级：200 km/ms ≈ 2e5 km/s（工程近似）
FIBER_SPEED_KM_MS = 200.0  # 单位：km/ms
BASE_ROUTER_DELAY_MS = 2.0


def calc_access_latency_ms(distance_km: float) -> float:
    """一次真实接入：传播 + 固定路由/协议基线 (ms)。distance_km >= 0。"""
    d = max(0.0, float(distance_km))
    return (d / FIBER_SPEED_KM_MS) + BASE_ROUTER_DELAY_MS


def propagation_latency_ms(distance_km: float) -> float:
    """仅传播段 (ms)，不含 BASE_ROUTER_DELAY。用于 Future 超额等非新开端口语义。"""
    if distance_km <= 0.0:
        return 0.0
    return float(distance_km) / FIBER_SPEED_KM_MS
