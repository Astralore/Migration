"""
Reward v2.1 P2 cov50 中等规模实验。

在 P1 基础上启用：
  - REWARD_V2_CURRICULUM：S / αβ / λ 退火
  - MARL_SOFT_CF_BIAS：v2 软 counterfactual logit bias
  - MARL_WARMSTART_CHECKPOINT（可选）：同架构 checkpoint 热启动

快筛：MEDIUM_VALIDATION_MARL_EPOCHS=2

示例：
  python -u run_reward_v21_p2_cov50.py
  set MEDIUM_VALIDATION_MARL_EPOCHS=2 && python -u run_reward_v21_p2_cov50.py
  set MARL_WARMSTART_CHECKPOINT=experiments\\...\\marl_gat_reactive.pth && python -u run_reward_v21_p2_cov50.py
"""

import os

os.environ["REWARD_SCHEME"] = "v2"
os.environ["REWARD_V2_USE_INTERNAL_PATH"] = os.environ.get("REWARD_V2_USE_INTERNAL_PATH", "1")
os.environ["REWARD_V2_OBJECTIVE_SCALE_MS"] = os.environ.get("REWARD_V2_OBJECTIVE_SCALE_MS", "100000")
os.environ["REWARD_V2_CURRICULUM"] = os.environ.get("REWARD_V2_CURRICULUM", "1")
os.environ["MEDIUM_VALIDATION_REACTIVE_ONLY"] = "1"
os.environ["MARL_P1"] = os.environ.get("MARL_P1", "1")
os.environ["MARL_SOFT_CF_BIAS"] = os.environ.get("MARL_SOFT_CF_BIAS", "1")
os.environ["PROACTIVE_MAX_MIGRATIONS_PER_DECISION"] = "0"
os.environ["REACTIVE_MAX_MIGRATIONS_PER_DECISION"] = "0"
os.environ["PROACTIVE_MIGRATION_BUDGET_MS"] = "6000"

if "MARL_WARMSTART_CHECKPOINT" not in os.environ:
    _p1_ckpt = os.path.join(
        "experiments",
        "medium_validation_20260605_145403_reward_v21_p1_v1",
        "checkpoints",
        "marl_gat_reactive.pth",
    )
    if os.path.isfile(_p1_ckpt):
        os.environ["MARL_WARMSTART_CHECKPOINT"] = _p1_ckpt

_marl_epochs = os.environ.get("MEDIUM_VALIDATION_MARL_EPOCHS", "8")
os.environ["MEDIUM_VALIDATION_MARL_EPOCHS"] = _marl_epochs
_default_tag = "reward_v21_p2_2ep" if _marl_epochs == "2" else "reward_v21_p2_8ep"
if "MEDIUM_VALIDATION_STAMP" not in os.environ:
    os.environ["MEDIUM_VALIDATION_STAMP_TAG"] = _default_tag

os.environ.pop("MEDIUM_VALIDATION_QUIET", None)
os.environ.pop("TQDM_DISABLE", None)
os.environ.pop("MEDIUM_VALIDATION_INFERENCE_ONLY", None)
os.environ.pop("MEDIUM_VALIDATION_INFERENCE_FORCE", None)
os.environ.pop("INFERENCE_FORCE", None)

from run_medium_validation_cov50 import main


if __name__ == "__main__":
    main()
