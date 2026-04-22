"""Unified scoring and ranking utilities for microservice migration experiments."""


def compute_score(migrations, violations, weight=0.5):
    """Composite score: M + weight * V."""
    return migrations + weight * violations


def print_ranking(results_dict, weight=0.5):
    """
    Print a ranked comparison table.

    Parameters
    ----------
    results_dict : dict
        {"AlgoName": results_dict_from_run, ...}
        Each value must contain 'total_migrations' and 'total_violations'.
    weight : float
        Violation weight in composite score.

    Columns
    -------
    M   : total_migrations  — microservice nodes actually relocated
    V   : total_violations  — real SLA breaches (gateway_dist > 15km, user-perceived)
    D   : decision_count    — migration triggers (includes proactive decisions)
    Score : M + weight * V
    """
    rows = []
    for name, res in results_dict.items():
        m = res['total_migrations']
        v = res['total_violations']
        s = compute_score(m, v, weight)
        r = res.get('total_reward', 0.0)
        d = res.get('decision_count', 0)
        rows.append((name, m, v, d, s, r))

    rows.sort(key=lambda x: x[4])  # sort by Score

    header = (f"{'Rank':<5} {'Algorithm':<22} {'M':>8} {'V(real)':>10} "
              f"{'D(trigger)':>12} {'Score':>10} {'Reward':>12}")
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for i, (name, m, v, d, s, r) in enumerate(rows, 1):
        print(f"{i:<5} {name:<22} {m:>8} {v:>10} {d:>12} {s:>10.1f} {r:>12.2f}")
    print(sep)
