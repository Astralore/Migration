"""
Reward v2.1 P3 cov50：P1 + P2 + L_internal γ 课程 + B1.1b CF gate。

P3 核心：训练目标 L_e2e = L_acc + γ·L_internal，前 2 epoch γ=0，随后渐增至 1.0。
B1.1b：CF gate mode=score（与 _clip_reactive_actions 对齐）+ entry rescue；CF gate 开启时恢复 rule clip。

默认 8 MARL epoch（reactive-only）。快筛：MEDIUM_VALIDATION_MARL_EPOCHS=2

C1 快验（P3 checkpoint + B1.1b 推理）：
  set MEDIUM_VALIDATION_STAMP=20260605_170422_reward_v21_p3_8ep
  set MEDIUM_VALIDATION_INFERENCE_ONLY=1
  set MEDIUM_VALIDATION_INFERENCE_FORCE=1
  python -u run_reward_v21_p3_cov50.py

示例：
  python -u run_reward_v21_p3_cov50.py
  set MEDIUM_VALIDATION_MARL_EPOCHS=2 && python -u run_reward_v21_p3_cov50.py
"""

import os

os.environ["REWARD_SCHEME"] = "v2"
os.environ["REWARD_V2_USE_INTERNAL_PATH"] = os.environ.get("REWARD_V2_USE_INTERNAL_PATH", "1")
os.environ["REWARD_V2_OBJECTIVE_SCALE_MS"] = os.environ.get("REWARD_V2_OBJECTIVE_SCALE_MS", "100000")
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
_default_tag = (
    f"reward_v21_p3{_guard_suffix}_2ep" if _marl_epochs == "2" else f"reward_v21_p3{_guard_suffix}_8ep"
)
if "MEDIUM_VALIDATION_STAMP" not in os.environ:
    os.environ["MEDIUM_VALIDATION_STAMP_TAG"] = _default_tag

os.environ.pop("MEDIUM_VALIDATION_QUIET", None)
os.environ.pop("TQDM_DISABLE", None)

from run_medium_validation_cov50 import main


if __name__ == "__main__":
    main()
