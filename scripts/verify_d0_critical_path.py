"""D0 smoke: compare norm_traffic sum vs critical-path comm on a split FanOut placement."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.microservice_dags import MICROSERVICE_DAGS
from core.reward import (
    build_servers_info,
    calculate_microservice_reward,
    compute_internal_critical_path_ms,
    edge_actual_rpc_calls,
    edge_effective_latency_ms,
)
from core.geo import haversine_distance
from core.physics_utils import BASE_ROUTER_DELAY_MS, FIBER_SPEED_KM_MS
import pandas as pd


def _legacy_comm_sum_ms(dag_info, assignments, servers_info):
    edges_items = list(dag_info["edges"].items())
    if not edges_items:
        return 0.0
    max_traffic = max(dag_info["edges"].values())
    total = 0.0
    for (src, dst), traffic in edges_items:
        if assignments[src] == assignments[dst]:
            continue
        slat, slon = servers_info[assignments[src]]
        dlat, dlon = servers_info[assignments[dst]]
        edge_dist = float(haversine_distance(slat, slon, dlat, dlon))
        norm_t = float(traffic) / float(max_traffic) if max_traffic > 0 else 0.0
        base = (max(0.0, edge_dist) / FIBER_SPEED_KM_MS) + BASE_ROUTER_DELAY_MS
        total += norm_t * base
    return total


def main():
    dag_type = "FanOut_Broadcaster_1"
    dag_info = MICROSERVICE_DAGS[dag_type]
    servers_df = pd.DataFrame(
        [
            {"edge_server_id": 1, "latitude": 40.0, "longitude": 116.0},
            {"edge_server_id": 2, "latitude": 40.45, "longitude": 116.0},
        ]
    )
    servers_info = build_servers_info(servers_df)
    nodes = list(dag_info["nodes"].keys())
    assign_colocated = {n: 1 for n in nodes}
    assign_split = {n: 1 for n in nodes}
    if len(nodes) >= 2:
        assign_split[nodes[1]] = 2

    for label, assign in [("colocated", assign_colocated), ("split", assign_split)]:
        _, details = calculate_microservice_reward(
            "taxi-1",
            dag_info,
            assign,
            assign,
            (40.0, 116.0),
            servers_info,
        )
        legacy = _legacy_comm_sum_ms(dag_info, assign, servers_info)
        cp = compute_internal_critical_path_ms(dag_info, assign, servers_info)
        print(
            f"[{label}] legacy_sum_comm={legacy:.2f} ms | "
            f"D0_internal_path={cp:.2f} ms | details.comm={details['communication_cost']:.2f} ms"
        )

    heavy_edge = max(dag_info["edges"].items(), key=lambda kv: kv[1])
    (src, dst), traffic = heavy_edge
    rpc = edge_actual_rpc_calls(traffic)
    eff = edge_effective_latency_ms(50.0, traffic, True)
    print(
        f"Heavy edge {src}->{dst} traffic={traffic} "
        f"actual_rpc_calls={rpc:.1f} effective_ms@50km={eff:.1f}"
    )


if __name__ == "__main__":
    main()
