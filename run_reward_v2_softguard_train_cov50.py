"""
Reward v2 + Phase C soft guard: full medium-validation train + inference (8+8 MARL epochs).

For a shorter reward-v2 check, use run_reward_v2_medium_test_cov50.py (default 2 epochs).

- REWARD_SCHEME=v2: alpha*E^2 SLA + exp(size/tau) migration + linear -objective/scale reward.
- Guard: MAX<=0 (budget/ROI only), PROACTIVE_MIGRATION_BUDGET_MS=6000.
- Default S=10000 (after future_gain fix + scale5000 STAY collapse).

Env must be set before any import that loads core.reward / algorithms.
"""

import os

os.environ["REWARD_SCHEME"] = "v2"
os.environ["REWARD_V2_OBJECTIVE_SCALE_MS"] = os.environ.get(
    "REWARD_V2_OBJECTIVE_SCALE_MS", "10000"
)
os.environ["PROACTIVE_MAX_MIGRATIONS_PER_DECISION"] = "0"
os.environ["REACTIVE_MAX_MIGRATIONS_PER_DECISION"] = "0"
os.environ["PROACTIVE_MIGRATION_BUDGET_MS"] = "6000"
os.environ["MEDIUM_VALIDATION_STAMP"] = "20260526_reward_v2_softguard_scale10000_v1"
os.environ.pop("MEDIUM_VALIDATION_INFERENCE_ONLY", None)
os.environ.pop("MEDIUM_VALIDATION_INFERENCE_FORCE", None)
os.environ.pop("INFERENCE_FORCE", None)

from run_medium_validation_cov50 import main


if __name__ == "__main__":
    main()
