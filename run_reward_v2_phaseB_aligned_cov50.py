"""
Reward v2 + Phase B/B2 guard alignment: full train + inference (8 MARL epochs).

Fair v1 vs v2 comparison: only REWARD_SCHEME differs from Phase B2 defaults.
- MAX=1 per decision (same as phaseB2_reward_align_v1)
- No soft-guard budget override (marl_gat defaults)
- REWARD_SCHEME=v2, S=10000

Launch:
  python -u run_reward_v2_phaseB_aligned_cov50.py
"""

import os

os.environ["REWARD_SCHEME"] = "v2"
os.environ["REWARD_V2_OBJECTIVE_SCALE_MS"] = os.environ.get(
    "REWARD_V2_OBJECTIVE_SCALE_MS", "10000"
)
os.environ["PROACTIVE_MAX_MIGRATIONS_PER_DECISION"] = "1"
os.environ["REACTIVE_MAX_MIGRATIONS_PER_DECISION"] = "1"
os.environ["MEDIUM_VALIDATION_STAMP"] = "20260526_reward_v2_phaseB_aligned_v1"
os.environ.pop("PROACTIVE_MIGRATION_BUDGET_MS", None)
os.environ.pop("MEDIUM_VALIDATION_MARL_EPOCHS", None)
os.environ.pop("MEDIUM_VALIDATION_INFERENCE_ONLY", None)
os.environ.pop("MEDIUM_VALIDATION_INFERENCE_FORCE", None)
os.environ.pop("INFERENCE_FORCE", None)
os.environ.pop("MEDIUM_VALIDATION_QUIET", None)
os.environ.pop("TQDM_DISABLE", None)

from run_medium_validation_cov50 import main


if __name__ == "__main__":
    main()
