"""一次性验证脚本：阶段一 physics_utils、阶段二 reward、阶段四 context。"""
from __future__ import annotations

import pandas as pd

from core.physics_utils import (
    FIBER_SPEED_KM_MS,
    BASE_ROUTER_DELAY_MS,
    calc_access_latency_ms,
    propagation_latency_ms,
)
from core.context import (
    DISTANCE_THRESHOLD_KM,
    USER_SLA_TOLERANCE_MS,
    check_sla_violation,
    get_trigger_type,
    TRIGGER_REACTIVE,
    TRIGGER_PROACTIVE,
)
from core.microservice_dags import MICROSERVICE_DAGS
from core.reward import calculate_microservice_reward, build_servers_info, REWARD_CLIP_MIN
from core.dag_utils import initialize_dag_assignment


def ok(name: str, cond: bool, msg: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {msg}" if msg else ""))
    if not cond:
        raise SystemExit(1)


def main() -> None:
    print("=== 1. physics_utils ===")
    ok("FIBER_SPEED_KM_MS", FIBER_SPEED_KM_MS == 200.0)
    ok("BASE_ROUTER_DELAY_MS", BASE_ROUTER_DELAY_MS == 2.0)
    ok("calc(0)==2", abs(calc_access_latency_ms(0) - 2.0) < 1e-9)
    ok("propagation(0)==0", propagation_latency_ms(0) == 0.0)
    ok("propagation excess", abs(propagation_latency_ms(20.0) - 20.0 / FIBER_SPEED_KM_MS) < 1e-12)
    print(f"  calc_access_latency_ms(15km) = {calc_access_latency_ms(15.0):.6f} ms")

    print("=== 2. context ===")
    ok("USER_SLA matches 0.99*calc(15)", abs(USER_SLA_TOLERANCE_MS - calc_access_latency_ms(15.0) * 0.99) < 1e-9)
    # 用户与网关同点：不应违规
    ok("check_sla same point", not check_sla_violation(0.0, 0.0, 0.0, 0.0))
    # 远超 15km（经纬度粗略拉开）
    ok("check_sla far", check_sla_violation(0.0, 0.0, 10.0, 0.0))
    # get_trigger_type：无前瞻 -> None
    t0 = get_trigger_type(0.0, 0.0, 0.0, 0.0, predicted_locations=None, proactive_enabled=False)
    ok("get_trigger safe no proactive", t0 is None)
    # 前瞻一步远离网关 -> PROACTIVE
    preds = [(1.0, 0.0)]  # 约 111km 量级 — 远大于 PROACTIVE_WARNING_KM
    t1 = get_trigger_type(0.0, 0.0, 0.0, 0.0, predicted_locations=preds, proactive_enabled=True)
    ok("get_trigger proactive", t1 == TRIGGER_PROACTIVE, repr(t1))

    print("=== 3. reward.calculate_microservice_reward ===")
    dag = MICROSERVICE_DAGS["IoT_Lightweight_DAG"]
    servers_df = pd.read_csv("data/edge_server_locations.csv")
    info = build_servers_info(servers_df)
    sid = int(servers_df.iloc[0]["edge_server_id"])
    cur = initialize_dag_assignment("IoT_Lightweight_DAG", sid)
    prev = dict(cur)
    r, d = calculate_microservice_reward(
        1, dag, cur, prev, (40.7128, -74.0060), info, trigger_type=TRIGGER_REACTIVE
    )
    ok("reward == -total_cost or clip", abs(r + d["total_cost_ms"]) < 1e-6 or r == REWARD_CLIP_MIN)
    ok("details total_cost_ms", d["total_cost_ms"] == d["total_cost"])
    ok("future 0 no preds", d["future_penalty"] == 0.0)

    # 强制一次迁移：改一个节点服务器
    prev2 = dict(cur)
    cur2 = dict(cur)
    n0 = list(dag["nodes"].keys())[0]
    cur2[n0] = int(servers_df.iloc[1]["edge_server_id"])
    r2, d2 = calculate_microservice_reward(
        1, dag, cur2, prev2, (40.7128, -74.0060), info, trigger_type=TRIGGER_REACTIVE
    )
    ok("migration > 0 after change", d2["migration_cost"] > 0)

    print("=== 4. algorithms 导入 ===")
    import algorithms.sa  # noqa: F401
    import algorithms.dqn  # noqa: F401
    import algorithms.hybrid_sac  # noqa: F401
    import algorithms.hybrid_sa_dqn  # noqa: F401
    print("  [PASS] import sa, dqn, hybrid_sac, hybrid_sa_dqn")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
