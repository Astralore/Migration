"""D1 smoke: compare v2.0 vs v2.1 SLA objective on split FanOut placement."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["REWARD_SCHEME"] = "v2"

import pandas as pd

from core.microservice_dags import MICROSERVICE_DAGS
from core.reward import build_servers_info, calculate_microservice_reward, use_reward_v2_internal_path


def main():
    dag_info = MICROSERVICE_DAGS["FanOut_Broadcaster_1"]
    servers_df = pd.DataFrame(
        [
            {"edge_server_id": 1, "latitude": 40.0, "longitude": 116.0},
            {"edge_server_id": 2, "latitude": 40.45, "longitude": 116.0},
        ]
    )
    servers_info = build_servers_info(servers_df)
    nodes = list(dag_info["nodes"].keys())
    assign = {n: 1 for n in nodes}
    if len(nodes) >= 2:
        assign[nodes[1]] = 2

    for flag, label in [("0", "v2.0 objective"), ("1", "v2.1 objective")]:
        os.environ["REWARD_V2_USE_INTERNAL_PATH"] = flag
        _, details = calculate_microservice_reward(
            "taxi-1",
            dag_info,
            assign,
            assign,
            (40.0, 116.0),
            servers_info,
        )
        scale = float(os.environ.get("REWARD_V2_OBJECTIVE_SCALE_MS", "50000"))
        reward = float(details.get("reward", 0.0))
        print(
            f"[{label}] use_internal={use_reward_v2_internal_path()} "
            f"L_internal={details['internal_critical_path_ms']:.1f} ms "
            f"sla_report={details['sla_penalty_ms']:.1f} ms "
            f"sla_objective={details['sla_penalty_objective_ms']:.1f} ms "
            f"J={details['reward_objective_ms']:.1f} ms r={reward:.4f} (S={scale:.0f})"
        )


if __name__ == "__main__":
    main()
