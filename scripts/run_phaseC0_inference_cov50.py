"""
Phase C0: inference-only ablation on Phase B checkpoints.

Relax per-decision migration cap (default MAX=2, proactive budget=6000 ms) without
retraining.  Compare against frozen Phase B inference metrics (MAX=1).
"""

import json
import os

# Must be set before marl_gat is imported (guard constants read at import time).
os.environ.setdefault("PROACTIVE_MAX_MIGRATIONS_PER_DECISION", "2")
os.environ.setdefault("REACTIVE_MAX_MIGRATIONS_PER_DECISION", "2")
os.environ.setdefault("PROACTIVE_MIGRATION_BUDGET_MS", "6000")

import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from algorithms.marl_gat import (
    PROACTIVE_MAX_MIGRATIONS_PER_DECISION,
    PROACTIVE_MIGRATION_BUDGET_MS,
    REACTIVE_MAX_MIGRATIONS_PER_DECISION,
    run_marl_gat_microservice,
)
from core.data_loader import DEFAULT_SERVER_PATH, DEFAULT_TAXI_PATH, load_data
from prediction.simple_predictor import SimpleTrajectoryPredictor
from run_comparison import _avg_total_cost_ms
from run_medium_validation_cov50 import (
    ACTIVE_USERS,
    FORECAST_HORIZON,
    PROCESSED_TAXI_PATH,
    SPLIT_SEED,
    _filter_top_active,
    _migration_efficiency_metrics,
    _migration_efficiency_row,
    _split_by_balanced_exposure,
    _summarize_result,
    _to_jsonable,
)

SOURCE_EXPERIMENT = Path(
    os.environ.get(
        "PHASE_C0_SOURCE",
        "experiments/medium_validation_20260526_phaseB_reward_align_v1",
    )
)
STAMP = os.environ.get(
    "PHASE_C0_STAMP",
    datetime.now().strftime("%Y%m%d_%H%M%S_phaseC0_max2_infer"),
)
OUT_DIR = Path("experiments") / f"phaseC0_inference_{STAMP}"


def _fmt(value, *, percent=False):
    if value is None:
        return "—"
    if percent:
        return f"{100.0 * value:.2f}%"
    return f"{value:.2f}"


def _load_phase_b_baseline():
    path = SOURCE_EXPERIMENT / "results.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("inference")


def _metric_row(label, res, *, proactive=False):
    dc = int(res.get("decision_count") or 0)
    mig = _migration_efficiency_metrics(res)
    avg_sla = (float(res.get("total_sla_penalty_ms") or 0.0) / dc) if dc > 0 else 0.0
    avg_total = _avg_total_cost_ms(res)
    pro = int(res.get("proactive_decisions") or 0)
    mig_dec = int(res.get("migration_decision_count") or 0)
    if proactive:
        return (
            f"| {label} | {res.get('total_migrations', 0)} | {mig_dec} | {res.get('total_violations', 0)} | "
            f"{res.get('severe_sla_violations', 0)} | {res.get('p95_sla_excess_distance_km', 0):.2f} | "
            f"{avg_sla:.2f} | {avg_total:.2f} | {pro} | "
            f"{_fmt(mig['cost_per_migrated_node_ms'])} | {_fmt(mig['cost_per_migration_decision_ms'])} | "
            f"{_fmt(mig['migration_cost_share'], percent=True)} |\n"
        )
    return (
        f"| {label} | {res.get('total_migrations', 0)} | {mig_dec} | {res.get('total_violations', 0)} | "
        f"{res.get('severe_sla_violations', 0)} | {res.get('p95_sla_excess_distance_km', 0):.2f} | "
        f"{avg_sla:.2f} | {avg_total:.2f} | "
        f"{_fmt(mig['cost_per_migrated_node_ms'])} | {_fmt(mig['cost_per_migration_decision_ms'])} | "
        f"{_fmt(mig['migration_cost_share'], percent=True)} |\n"
    )


def _write_report(payload, baseline_inference):
    pro = payload["inference"]["proactive"].get("GAT-MARL")
    rea = payload["inference"]["reactive"].get("GAT-MARL")
    cfg = payload["config"]
    if pro is None or rea is None:
        return

    header = (
        "| Run | Migrations | Mig Decisions | SLA Risk | Severe | P95 (km) | Avg SLA Penalty (ms) | "
        "Avg Total Cost (ms) | Pro Decisions | Cost/Node | Cost/Mig Decision | Migration Share |\n"
        "|-----|------------|---------------|----------|--------|----------|----------------------|"
        "---------------------|---------------|-----------|-------------------|-----------------|\n"
    )

    pro_body = header
    if baseline_inference:
        b = baseline_inference.get("proactive", {}).get("GAT-MARL")
        if b:
            pro_body += _metric_row("Phase B (MAX=1)", b, proactive=True)
    pro_body += _metric_row("Phase C0 (MAX=2)", pro, proactive=True)

    rea_header = (
        "\n| Run | Migrations | Mig Decisions | SLA Risk | Severe | P95 (km) | Avg SLA Penalty (ms) | "
        "Avg Total Cost (ms) | Cost/Node | Cost/Mig Decision | Migration Share |\n"
        "|-----|------------|---------------|----------|--------|----------|----------------------|"
        "---------------------|-----------|-------------------|-----------------|\n"
    )
    rea_body = rea_header
    if baseline_inference:
        b = baseline_inference.get("reactive", {}).get("GAT-MARL")
        if b:
            rea_body += _metric_row("Phase B (MAX=1)", b, proactive=False)
    rea_body += _metric_row("Phase C0 (MAX=2)", rea, proactive=False)

    mig_table = (
        "\n## 迁移效率汇总（GAT-MARL）\n\n"
        "| Run | Pro: Cost/Node | Pro: Cost/Mig Dec | Pro: Share | Rea: Cost/Node | Rea: Cost/Mig Dec | Rea: Share |\n"
        "|-----|----------------|-------------------|------------|----------------|-------------------|------------|\n"
    )
    if baseline_inference:
        bpro = baseline_inference.get("proactive", {}).get("GAT-MARL", {})
        brea = baseline_inference.get("reactive", {}).get("GAT-MARL", {})
        m0 = _migration_efficiency_metrics(bpro)
        m1 = _migration_efficiency_metrics(brea)
        mig_table += (
            f"| Phase B | {_fmt(m0['cost_per_migrated_node_ms'])} | {_fmt(m0['cost_per_migration_decision_ms'])} | "
            f"{_fmt(m0['migration_cost_share'], percent=True)} | "
            f"{_fmt(m1['cost_per_migrated_node_ms'])} | {_fmt(m1['cost_per_migration_decision_ms'])} | "
            f"{_fmt(m1['migration_cost_share'], percent=True)} |\n"
        )
    mc0p = _migration_efficiency_metrics(pro or {})
    mc0r = _migration_efficiency_metrics(rea or {})
    mig_table += (
        f"| Phase C0 | {_fmt(mc0p['cost_per_migrated_node_ms'])} | {_fmt(mc0p['cost_per_migration_decision_ms'])} | "
        f"{_fmt(mc0p['migration_cost_share'], percent=True)} | "
        f"{_fmt(mc0r['cost_per_migrated_node_ms'])} | {_fmt(mc0r['cost_per_migration_decision_ms'])} | "
        f"{_fmt(mc0r['migration_cost_share'], percent=True)} |\n"
    )

    report = f"""# Phase C0 推理摸底：放宽每步迁移上限（不重训）

**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**输出目录**：`{OUT_DIR}`  
**Checkpoint**：`{cfg['source_experiment']}`（阶段 B，未重训）  
**Guard**：Proactive MAX={cfg['proactive_max_migrations']}，Reactive MAX={cfg['reactive_max_migrations']}，Proactive budget={cfg['proactive_migration_budget_ms']:.0f} ms  
**对比基线**：`{SOURCE_EXPERIMENT}/results.json` 中阶段 B 推理（MAX=1，budget=3000）

## 推理 Proactive（GAT-MARL）

{pro_body}

## 推理 Reactive（GAT-MARL）

{rea_body}

{mig_table}

## 说明

- C0 仅改 inference guard；策略权重仍为阶段 B checkpoint。
- **Mig Decisions** = `migration_decision_count`（本步 `migration_cost > 0` 的次数）。
- 若 C0 在 SLA/Severe 上明显优于 B 且迁移可控，再进入阶段 C 重训（train/infer 一致）。
"""
    (OUT_DIR / "result.md").write_text(report, encoding="utf-8")


def main():
    random.seed(SPLIT_SEED)
    np.random.seed(SPLIT_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = SOURCE_EXPERIMENT / "checkpoints"
    for name in ("marl_gat_proactive.pth", "marl_gat_reactive.pth"):
        if not (checkpoint_dir / name).exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_dir / name}")

    baseline_inference = _load_phase_b_baseline()

    start = time.time()
    df_all = load_data(DEFAULT_TAXI_PATH, processed_csv=PROCESSED_TAXI_PATH)
    df = _filter_top_active(df_all, ACTIVE_USERS)
    servers_df = pd.read_csv(DEFAULT_SERVER_PATH)
    train_df, test_df, train_ids, test_ids, split_meta = _split_by_balanced_exposure(df, servers_df)
    predictor = SimpleTrajectoryPredictor(forecast_horizon=FORECAST_HORIZON).fit(train_df)

    payload = {
        "config": {
            "source_experiment": str(SOURCE_EXPERIMENT),
            "active_users": ACTIVE_USERS,
            "split_seed": SPLIT_SEED,
            "proactive_max_migrations": PROACTIVE_MAX_MIGRATIONS_PER_DECISION,
            "reactive_max_migrations": REACTIVE_MAX_MIGRATIONS_PER_DECISION,
            "proactive_migration_budget_ms": PROACTIVE_MIGRATION_BUDGET_MS,
            "started_at": datetime.now().isoformat(),
        },
        "data": {
            "test_rows": int(len(test_df)),
            "test_taxis": int(test_df["taxi_id"].nunique()),
            "split": _to_jsonable(split_meta),
        },
        "inference": {"proactive": {}, "reactive": {}},
    }

    def save(*, write_report=False):
        with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2)
        if write_report:
            _write_report(payload, baseline_inference)

    save()

    for mode, proactive in (("proactive", True), ("reactive", False)):
        print(
            f"\n=== C0 INFERENCE {mode.upper()} "
            f"(max_mig={PROACTIVE_MAX_MIGRATIONS_PER_DECISION if proactive else REACTIVE_MAX_MIGRATIONS_PER_DECISION}) ===",
            flush=True,
        )
        payload["inference"][mode]["GAT-MARL"] = _summarize_result(
            run_marl_gat_microservice(
                test_df,
                servers_df,
                predictor=predictor,
                proactive=proactive,
                inference_mode=True,
                checkpoint_path=str(checkpoint_dir / f"marl_gat_{mode}.pth"),
                collect_dag_proactive_stats=proactive,
            )
        )
        payload["elapsed_seconds_so_far"] = time.time() - start
        save()

    payload["completed_at"] = datetime.now().isoformat()
    payload["elapsed_seconds"] = time.time() - start
    save(write_report=True)
    print(f"\n[DONE] Phase C0 inference saved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
