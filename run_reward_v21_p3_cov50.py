"""
Reward v2.1 P3/P4 cov50 launcher.

P3 (legacy): B11b + Direction A + E/B/A'
P4 (COLOCATE): MARL_COLOCATE_REACTIVE=1 — traffic-aware joint migration + Direction A

COLOCATE v4 快验 / 训练：
  set MARL_COLOCATE_REACTIVE=1
  set MARL_CF_SLA_GAIN_GATE=0
  set MEDIUM_VALIDATION_MARL_EPOCHS=2
  python -u run_reward_v21_p3_cov50.py
"""

import os

os.environ["REWARD_SCHEME"] = "v2"
os.environ["REWARD_V2_USE_INTERNAL_PATH"] = os.environ.get("REWARD_V2_USE_INTERNAL_PATH", "1")
os.environ["REWARD_V2_OBJECTIVE_SCALE_MS"] = os.environ.get("REWARD_V2_OBJECTIVE_SCALE_MS", "100000")

_colocate_mode = os.environ.get("MARL_COLOCATE_REACTIVE", "0").strip().lower() not in (
    "0",
    "false",
    "no",
)

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

if _colocate_mode:
    os.environ["MARL_COLOCATE_REACTIVE"] = "1"
    os.environ["MARL_CF_SLA_GAIN_GATE"] = os.environ.get("MARL_CF_SLA_GAIN_GATE", "0")
    os.environ["SA_COLOCATE_MODE_A"] = os.environ.get("SA_COLOCATE_MODE_A", "1")
    os.environ["MARL_CF_PHYSICAL_MIGRATION_COST"] = os.environ.get(
        "MARL_CF_PHYSICAL_MIGRATION_COST", "0"
    )
    os.environ["MARL_REACTIVE_DENSE_BONUS"] = os.environ.get("MARL_REACTIVE_DENSE_BONUS", "0")
    os.environ["MARL_MIGRATION_AMORT_H"] = os.environ.get("MARL_MIGRATION_AMORT_H", "100")
    # Amortization replaces n-step — disable n-step to prevent SLA spike accumulation
    os.environ["MARL_NSTEP_RETURN"] = "0"
else:
    if "MARL_CF_PHYSICAL_MIGRATION_COST" not in os.environ:
        os.environ["MARL_CF_PHYSICAL_MIGRATION_COST"] = "1"
    if "MARL_REACTIVE_DENSE_BONUS" not in os.environ:
        os.environ["MARL_REACTIVE_DENSE_BONUS"] = "1" if _train_total_cost else "0"

os.environ["MEDIUM_VALIDATION_REACTIVE_ONLY"] = "1"
os.environ["MARL_P1"] = os.environ.get("MARL_P1", "1")
if not _colocate_mode:
    os.environ["MARL_CF_SLA_GAIN_GATE"] = os.environ.get("MARL_CF_SLA_GAIN_GATE", "1")
    os.environ["MARL_CF_GATE_MODE"] = os.environ.get("MARL_CF_GATE_MODE", "score")
if "MARL_SOFT_CF_BIAS" not in os.environ:
    os.environ["MARL_SOFT_CF_BIAS"] = "0"
os.environ["PROACTIVE_MAX_MIGRATIONS_PER_DECISION"] = "0"
os.environ["REACTIVE_MAX_MIGRATIONS_PER_DECISION"] = "0"
os.environ["PROACTIVE_MIGRATION_BUDGET_MS"] = "6000"

_marl_epochs = os.environ.get("MEDIUM_VALIDATION_MARL_EPOCHS", "8")
os.environ["MEDIUM_VALIDATION_MARL_EPOCHS"] = _marl_epochs

if _colocate_mode:
    _amort_h = os.environ.get("MARL_MIGRATION_AMORT_H", "100")
    _amort_suffix = f"_aH{_amort_h}" if _amort_h != "0" else ""
    _default_tag = (
        f"reward_v21_p4_colocate{_amort_suffix}_2ep"
        if _marl_epochs == "2"
        else f"reward_v21_p4_colocate{_amort_suffix}_8ep"
    )
else:
    _cf_gate = os.environ.get("MARL_CF_SLA_GAIN_GATE", "1").strip().lower() not in ("0", "false", "no")
    _gate_mode = os.environ.get("MARL_CF_GATE_MODE", "score").strip().lower()
    if _cf_gate and _gate_mode == "score":
        _guard_suffix = "_b11b"
    elif _cf_gate:
        _guard_suffix = "_b11a"
    else:
        _guard_suffix = ""

    _cf_e = os.environ.get("MARL_CF_PHYSICAL_MIGRATION_COST", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    _reactive_bonus = os.environ.get("MARL_REACTIVE_DENSE_BONUS", "0").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    _reactive_suffix = ""
    if _cf_e and _reactive_bonus and _train_total_cost:
        _reactive_suffix = "_eba"
    elif _train_total_cost:
        _reactive_suffix = "_a"
    elif _cf_e:
        _reactive_suffix = "_e"
    _cp_floor = os.environ.get("MARL_CF_TRAIN_SCORE_FLOOR", "").strip()
    if _cp_floor:
        _reactive_suffix += f"_cp{_cp_floor.replace('.', 'p').replace('-', 'm')}"

    _default_tag = (
        f"reward_v21_p3{_guard_suffix}{_reactive_suffix}_2ep"
        if _marl_epochs == "2"
        else f"reward_v21_p3{_guard_suffix}{_reactive_suffix}_8ep"
    )

if "MEDIUM_VALIDATION_STAMP" not in os.environ and "MEDIUM_VALIDATION_STAMP_TAG" not in os.environ:
    os.environ["MEDIUM_VALIDATION_STAMP_TAG"] = _default_tag

os.environ.pop("MEDIUM_VALIDATION_QUIET", None)
os.environ.pop("TQDM_DISABLE", None)

from run_medium_validation_cov50 import main


if __name__ == "__main__":
    main()
