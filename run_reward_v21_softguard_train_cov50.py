"""
Reward v2.1 + Phase C soft guard: full train + inference (8 MARL epochs).

D0 critical-path comm in total_cost_ms; D1 L_e2e in v2 training P_SLA.
"""

import os

os.environ["REWARD_SCHEME"] = "v2"
os.environ["REWARD_V2_USE_INTERNAL_PATH"] = os.environ.get("REWARD_V2_USE_INTERNAL_PATH", "1")
os.environ["REWARD_V2_OBJECTIVE_SCALE_MS"] = os.environ.get(
    "REWARD_V2_OBJECTIVE_SCALE_MS", "100000"
)
os.environ["PROACTIVE_MAX_MIGRATIONS_PER_DECISION"] = "0"
os.environ["REACTIVE_MAX_MIGRATIONS_PER_DECISION"] = "0"
os.environ["PROACTIVE_MIGRATION_BUDGET_MS"] = "6000"
os.environ["MEDIUM_VALIDATION_STAMP"] = "20260527_reward_v21_softguard_v1"
os.environ.pop("MEDIUM_VALIDATION_INFERENCE_ONLY", None)
os.environ.pop("MEDIUM_VALIDATION_INFERENCE_FORCE", None)
os.environ.pop("INFERENCE_FORCE", None)

from run_medium_validation_cov50 import main


if __name__ == "__main__":
    main()
