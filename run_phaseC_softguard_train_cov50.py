"""
Phase C (scheme A): full train + inference with soft guards only.

- No per-step migration count cap (MAX<=0 → unlimited, subject to budget / ROI).
- Proactive migration budget 6000 ms + counterfactual ROI gates unchanged.
- Reward alignment matches phase B (CORE_MIGRATION_REWARD_WEIGHT=0.75 in reward.py).

Set env before any algorithm import so marl_gat guard constants pick up overrides.
"""

import os

# Force Phase C settings (do not inherit INFERENCE_ONLY / Phase B stamp from shell).
os.environ["PROACTIVE_MAX_MIGRATIONS_PER_DECISION"] = "0"
os.environ["REACTIVE_MAX_MIGRATIONS_PER_DECISION"] = "0"
os.environ["PROACTIVE_MIGRATION_BUDGET_MS"] = "6000"
os.environ["MEDIUM_VALIDATION_STAMP"] = "20260526_phaseC_softguard_v1"
os.environ.pop("MEDIUM_VALIDATION_INFERENCE_ONLY", None)
os.environ.pop("MEDIUM_VALIDATION_INFERENCE_FORCE", None)
os.environ.pop("INFERENCE_FORCE", None)

from run_medium_validation_cov50 import main


if __name__ == "__main__":
    main()
