"""Compare reward v1 vs v2 on representative SLA / migration scenarios."""

import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.context import DISTANCE_THRESHOLD_KM, USER_SLA_TOLERANCE_MS
from core.physics_utils import calc_access_latency_ms


def _load_reward(scheme: str):
    os.environ["REWARD_SCHEME"] = scheme
    import core.reward as reward_mod

    return importlib.reload(reward_mod)


def _row(label, dist_km, raw_mig_ms, image_mb, state_mb=0.0):
    access_ms = calc_access_latency_ms(dist_km)
    schemes = {}
    for name in ("v1", "v2"):
        r = _load_reward(name)
        sla = r.calculate_sla_penalty_ms(dist_km, access_ms)
        mig = r.calculate_nonlinear_migration_cost_ms(raw_mig_ms, image_mb, state_mb)
        if r.is_reward_v2():
            objective = sla + mig
            reward = -objective / r.REWARD_V2_OBJECTIVE_SCALE_MS
        else:
            objective = sla + r.CORE_MIGRATION_REWARD_WEIGHT * mig
            reward = -float(__import__("numpy").log1p(max(objective, 0.0) / r.REWARD_COST_SCALE_MS))
        excess = max(0.0, dist_km - DISTANCE_THRESHOLD_KM)
        schemes[name] = {
            "E_km": excess,
            "sla_ms": sla,
            "mig_ms": mig,
            "objective_ms": objective,
            "reward": reward,
        }
    return label, schemes


def main():
    scenarios = [
        ("合规 (E=0)", DISTANCE_THRESHOLD_KM, 500.0, 50, 0),
        ("轻违规 E=1km", DISTANCE_THRESHOLD_KM + 1.0, 500.0, 50, 0),
        ("中违规 E=5km", DISTANCE_THRESHOLD_KM + 5.0, 500.0, 50, 0),
        ("重违规 E=10km", DISTANCE_THRESHOLD_KM + 10.0, 500.0, 50, 0),
        ("轻违规+轻迁移 50MB", DISTANCE_THRESHOLD_KM + 2.0, 800.0, 50, 0),
        ("轻违规+重迁移 500MB", DISTANCE_THRESHOLD_KM + 2.0, 800.0, 450, 50),
        ("重违规+轻迁移 50MB", DISTANCE_THRESHOLD_KM + 10.0, 800.0, 50, 0),
        ("重违规+重迁移 500MB", DISTANCE_THRESHOLD_KM + 10.0, 800.0, 450, 50),
    ]
    print(f"SLA threshold = {DISTANCE_THRESHOLD_KM} km, QoS tolerance = {USER_SLA_TOLERANCE_MS} ms\n")
    print(f"{'场景':<22} | {'':^6} | {'SLA(ms)':>10} | {'Mig(ms)':>10} | {'Objective':>10} | {'Reward':>8}")
    print("-" * 80)
    for label, schemes in (_row(*s) for s in scenarios):
        for i, name in enumerate(("v1", "v2")):
            s = schemes[name]
            tag = label if i == 0 else ""
            print(
                f"{tag:<22} | {name:^6} | {s['sla_ms']:10.1f} | {s['mig_ms']:10.1f} "
                f"| {s['objective_ms']:10.1f} | {s['reward']:8.4f}"
            )
        v1, v2 = schemes["v1"], schemes["v2"]
        if v1["sla_ms"] > 0:
            ratio = v2["sla_ms"] / v1["sla_ms"]
            print(f"{'':22} | SLA v2/v1 = {ratio:.3f}x (E={v1['E_km']:.1f}km)")
        if v1["mig_ms"] > 0:
            ratio = v2["mig_ms"] / v1["mig_ms"]
            print(f"{'':22} | Mig v2/v1 = {ratio:.3f}x")
        print()


if __name__ == "__main__":
    main()
