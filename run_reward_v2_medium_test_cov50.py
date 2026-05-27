"""
Reward v2 medium-scale validation (2 MARL epochs by default).

Launch in terminal (unbuffered, no log pipe):
  python -u run_reward_v2_medium_test_cov50.py

After future_gain v2 fix: default S=10000, stamp scale10000_v1.
Progress: per-algorithm >>> / <<< lines + tqdm bars (quiet mode off).
"""

import os

os.environ["REWARD_SCHEME"] = "v2"
os.environ["REWARD_V2_OBJECTIVE_SCALE_MS"] = os.environ.get(
    "REWARD_V2_OBJECTIVE_SCALE_MS", "10000"
)
os.environ["PROACTIVE_MAX_MIGRATIONS_PER_DECISION"] = "0"
os.environ["REACTIVE_MAX_MIGRATIONS_PER_DECISION"] = "0"
os.environ["PROACTIVE_MIGRATION_BUDGET_MS"] = "6000"
os.environ["MEDIUM_VALIDATION_MARL_EPOCHS"] = os.environ.get(
    "MEDIUM_VALIDATION_MARL_EPOCHS", "2"
)
os.environ["MEDIUM_VALIDATION_STAMP"] = "20260526_reward_v2_scale10000_v1"
os.environ.pop("MEDIUM_VALIDATION_QUIET", None)
os.environ.pop("TQDM_DISABLE", None)
os.environ.pop("MEDIUM_VALIDATION_INFERENCE_ONLY", None)
os.environ.pop("MEDIUM_VALIDATION_INFERENCE_FORCE", None)
os.environ.pop("INFERENCE_FORCE", None)

from run_medium_validation_cov50 import main


if __name__ == "__main__":
    main()
