"""
Unified scoring and ranking utilities for microservice migration experiments.
Supports dual-track metric decoupling: Real Violations (V) vs Proactive Decisions (D).
"""


def compute_score(migrations, violations, weight=0.5):
    """Composite score: M + weight * V."""
    return migrations + weight * violations


def print_ranking(results_dict, weight=0.5):
    """
    Print a ranked comparison table with full metric breakdown.

    Parameters
    ----------
    results_dict : dict
        {"AlgoName": results_dict_from_run, ...}
        Each value must contain:
          - total_migrations (M)
          - total_violations (V) — real SLA breaches
          - proactive_decisions (D) — proactive triggers (optional, defaults to 0)
          - decision_count — total migration triggers
          - total_reward

    Columns
    -------
    M        : total_migrations     — microservice nodes actually relocated
    V(real)  : total_violations     — real SLA breaches (gateway_dist > 15km)
    D(proac) : proactive_decisions  — proactive triggers (future prediction)
    Score    : M + weight * V
    """
    rows = []
    for name, res in results_dict.items():
        m = res['total_migrations']
        v = res['total_violations']
        d_proac = res.get('proactive_decisions', 0)
        d_total = res.get('decision_count', 0)
        s = compute_score(m, v, weight)
        r = res.get('total_reward', 0.0)
        rows.append((name, m, v, d_proac, d_total, s, r))

    rows.sort(key=lambda x: x[5])  # sort by Score

    header = (f"{'Rank':<5} {'Algorithm':<22} {'M':>8} {'V(real)':>10} "
              f"{'D(proac)':>10} {'D(total)':>10} {'Score':>10} {'Reward':>12}")
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for i, (name, m, v, d_proac, d_total, s, r) in enumerate(rows, 1):
        print(f"{i:<5} {name:<22} {m:>8} {v:>10} {d_proac:>10} {d_total:>10} {s:>10.1f} {r:>12.2f}")
    print(sep)

    return rows


def print_proactive_analysis(proactive_results, reactive_results=None):
    """
    Print analysis comparing Proactive vs Reactive modes.

    Parameters
    ----------
    proactive_results : dict
        Results from algorithms run in Proactive mode.
    reactive_results : dict or None
        Results from algorithms run in Reactive mode (if available).
    """
    print("\n" + "=" * 80)
    print("  EXPERIMENTAL ANALYSIS: Proactive vs Reactive Migration")
    print("=" * 80)

    if reactive_results is None:
        print("  (Reactive baseline not provided — showing Proactive-only summary)")
        for name, res in proactive_results.items():
            m = res['total_migrations']
            v = res['total_violations']
            d = res.get('proactive_decisions', 0)
            d_total = res.get('decision_count', 0)
            print(f"\n  [{name}]")
            print(f"    Nodes Moved (M):       {m:>8}")
            print(f"    Real SLA Drops (V):    {v:>8}")
            print(f"    Proactive Decisions:   {d:>8} / {d_total} total")
        return

    print("\n  Proactive mode benefits analysis:")
    for name in proactive_results.keys():
        if name not in reactive_results:
            continue

        pro_res = proactive_results[name]
        rea_res = reactive_results[name]

        pro_v = pro_res['total_violations']
        rea_v = rea_res['total_violations']
        pro_d = pro_res.get('proactive_decisions', 0)
        rea_d = rea_res.get('proactive_decisions', 0)
        delta_d = pro_d - rea_d

        if rea_v > 0:
            drop_rate = (rea_v - pro_v) / rea_v * 100
        else:
            drop_rate = 0.0

        print(f"\n  [{name}]")
        print(f"    Compared to Reactive mode, Proactive executed {delta_d} more")
        print(f"    preemptive decisions, reducing real SLA drops from {rea_v} to {pro_v}.")
        if drop_rate > 0:
            print(f"    => Violation reduction: {drop_rate:.1f}%")
        elif drop_rate < 0:
            print(f"    => Violation increase: {abs(drop_rate):.1f}% (investigate!)")
        else:
            print(f"    => No change in violations.")

    print("\n" + "=" * 80)
