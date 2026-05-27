"""
Reward v2.1 medium-scale validation (D0+D1+D1.5+D2), reactive-only, S=100000.

- REWARD_SCHEME=v2.1 (v2 + internal path + clip + log-RPC features)
- No proactive / no trajectory predictor (REACTIVE_ONLY)
- Soft guard + 2 MARL epochs

Launch:
  python -u run_reward_v21_medium_test_cov50.py
"""

import os

os.environ["REWARD_SCHEME"] = "v2"
os.environ["REWARD_V2_USE_INTERNAL_PATH"] = os.environ.get("REWARD_V2_USE_INTERNAL_PATH", "1")
os.environ["REWARD_V2_OBJECTIVE_SCALE_MS"] = "100000"
os.environ["MEDIUM_VALIDATION_REACTIVE_ONLY"] = "1"
os.environ["PROACTIVE_MAX_MIGRATIONS_PER_DECISION"] = "0"
os.environ["REACTIVE_MAX_MIGRATIONS_PER_DECISION"] = "0"
os.environ["PROACTIVE_MIGRATION_BUDGET_MS"] = "6000"
os.environ["MEDIUM_VALIDATION_MARL_EPOCHS"] = os.environ.get(
    "MEDIUM_VALIDATION_MARL_EPOCHS", "2"
)
os.environ["MEDIUM_VALIDATION_STAMP"] = "20260527_reward_v21_reactive_s100k_v1"
os.environ.pop("MEDIUM_VALIDATION_QUIET", None)
os.environ.pop("TQDM_DISABLE", None)
os.environ.pop("MEDIUM_VALIDATION_INFERENCE_ONLY", None)
os.environ.pop("MEDIUM_VALIDATION_INFERENCE_FORCE", None)
os.environ.pop("INFERENCE_FORCE", None)

from run_medium_validation_cov50 import main


if __name__ == "__main__":
    main()
