"""
Reward v2.1 cov50 中等规模实验（唯一推荐入口）。

默认：Reactive-only、S=100000、硬护栏 bypass、P1 动作协调、8 MARL epoch 全量 train+inference。
快筛：MEDIUM_VALIDATION_MARL_EPOCHS=2

示例：
  python -u run_reward_v21_cov50.py
  set MEDIUM_VALIDATION_MARL_EPOCHS=2 && python -u run_reward_v21_cov50.py
  set MEDIUM_VALIDATION_STAMP_TAG=reward_v21_p1_smoke && python -u run_reward_v21_cov50.py
"""

import os

os.environ["REWARD_SCHEME"] = "v2"
os.environ["REWARD_V2_USE_INTERNAL_PATH"] = os.environ.get("REWARD_V2_USE_INTERNAL_PATH", "1")
os.environ["REWARD_V2_OBJECTIVE_SCALE_MS"] = "100000"
os.environ["MEDIUM_VALIDATION_REACTIVE_ONLY"] = "1"
os.environ["MARL_P1"] = os.environ.get("MARL_P1", "1")
os.environ["PROACTIVE_MAX_MIGRATIONS_PER_DECISION"] = "0"
os.environ["REACTIVE_MAX_MIGRATIONS_PER_DECISION"] = "0"
os.environ["PROACTIVE_MIGRATION_BUDGET_MS"] = "6000"

_marl_epochs = os.environ.get("MEDIUM_VALIDATION_MARL_EPOCHS", "8")
os.environ["MEDIUM_VALIDATION_MARL_EPOCHS"] = _marl_epochs
if "MEDIUM_VALIDATION_STAMP_TAG" not in os.environ and "MEDIUM_VALIDATION_STAMP" not in os.environ:
    os.environ["MEDIUM_VALIDATION_STAMP_TAG"] = (
        "reward_v21_p1_2ep" if _marl_epochs == "2" else "reward_v21_p1_8ep"
    )

os.environ.pop("MEDIUM_VALIDATION_QUIET", None)
os.environ.pop("TQDM_DISABLE", None)
os.environ.pop("MEDIUM_VALIDATION_INFERENCE_ONLY", None)
os.environ.pop("MEDIUM_VALIDATION_INFERENCE_FORCE", None)
os.environ.pop("INFERENCE_FORCE", None)

from run_medium_validation_cov50 import main


if __name__ == "__main__":
    main()
