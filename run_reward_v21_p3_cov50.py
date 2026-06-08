"""
Reward v2.1 P3 cov50：P1 + B1.1b CF gate + Direction A（训练目标对齐 total_cost_ms）。

Direction A（默认开启）：
  - Critic/Actor 优化 −total_cost_ms / 10000（与 SA 报表 KPI 同量纲）
  - γ=1 固定，关闭 P2/P3 课程与 agent-level λ penalty
  - B11b gate/clip 不变

关闭 A（legacy P3）：MARL_TRAIN_TOTAL_COST=0

默认 8 MARL epoch（reactive-only）。快筛：MEDIUM_VALIDATION_MARL_EPOCHS=2

示例：
  python -u run_reward_v21_p3_cov50.py
  set MEDIUM_VALIDATION_MARL_EPOCHS=2 && python -u run_reward_v21_p3_cov50.py
"""

import os

os.environ["REWARD_SCHEME"] = "v2"
os.environ["REWARD_V2_USE_INTERNAL_PATH"] = os.environ.get("REWARD_V2_USE_INTERNAL_PATH", "1")
os.environ["REWARD_V2_OBJECTIVE_SCALE_MS"] = os.environ.get("REWARD_V2_OBJECTIVE_SCALE_MS", "100000")

_train_total_cost = os.environ.get("MARL_TRAIN_TOTAL_COST", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
if "MARL_TRAIN_TOTAL_COST" not in os.environ:
    os.environ["MARL_TRAIN_TOTAL_COST"] = "1" if _train_total_cost else "0"

if _train_total_cost:
    os.environ["REWARD_V2_CURRICULUM"] = os.environ.get("REWARD_V2_CURRICULUM", "0")
    os.environ["REWARD_V2_INTERNAL_GAMMA"] = os.environ.get("REWARD_V2_INTERNAL_GAMMA", "0")
else:
    os.environ["REWARD_V2_CURRICULUM"] = os.environ.get("REWARD_V2_CURRICULUM", "1")
    os.environ["REWARD_V2_INTERNAL_GAMMA"] = os.environ.get("REWARD_V2_INTERNAL_GAMMA", "1")
    os.environ["REWARD_V2_GAMMA_WARMUP_EPOCHS"] = os.environ.get("REWARD_V2_GAMMA_WARMUP_EPOCHS", "2")
os.environ["MEDIUM_VALIDATION_REACTIVE_ONLY"] = "1"
os.environ["MARL_P1"] = os.environ.get("MARL_P1", "1")
os.environ["MARL_CF_SLA_GAIN_GATE"] = os.environ.get("MARL_CF_SLA_GAIN_GATE", "1")
os.environ["MARL_CF_GATE_MODE"] = os.environ.get("MARL_CF_GATE_MODE", "score")
# B1.1b: hard gate replaces soft CF bias by default
if "MARL_SOFT_CF_BIAS" not in os.environ:
    os.environ["MARL_SOFT_CF_BIAS"] = "0"
os.environ["PROACTIVE_MAX_MIGRATIONS_PER_DECISION"] = "0"
os.environ["REACTIVE_MAX_MIGRATIONS_PER_DECISION"] = "0"
os.environ["PROACTIVE_MIGRATION_BUDGET_MS"] = "6000"

_marl_epochs = os.environ.get("MEDIUM_VALIDATION_MARL_EPOCHS", "8")
os.environ["MEDIUM_VALIDATION_MARL_EPOCHS"] = _marl_epochs
_cf_gate = os.environ.get("MARL_CF_SLA_GAIN_GATE", "1").strip().lower() not in ("0", "false", "no")
_gate_mode = os.environ.get("MARL_CF_GATE_MODE", "score").strip().lower()
if _cf_gate and _gate_mode == "score":
    _guard_suffix = "_b11b"
elif _cf_gate:
    _guard_suffix = "_b11a"
else:
    _guard_suffix = ""
_a_suffix = "_a" if _train_total_cost else ""
_default_tag = (
    f"reward_v21_p3{_guard_suffix}{_a_suffix}_2ep"
    if _marl_epochs == "2"
    else f"reward_v21_p3{_guard_suffix}{_a_suffix}_8ep"
)
if "MEDIUM_VALIDATION_STAMP" not in os.environ:
    os.environ["MEDIUM_VALIDATION_STAMP_TAG"] = _default_tag

os.environ.pop("MEDIUM_VALIDATION_QUIET", None)
os.environ.pop("TQDM_DISABLE", None)

from run_medium_validation_cov50 import main


if __name__ == "__main__":
    main()
