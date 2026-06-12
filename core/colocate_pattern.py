"""Traffic-aware colocate pattern generation (mode A) for COLOCATE reactive MARL."""

from __future__ import annotations

import statistics
from typing import Dict, Iterable, List, Optional, Set, Tuple

from core.dag_utils import is_external_node


def compute_t_critical(dag_info) -> float:
    """T_critical = median(edge_traffic) × 2 (data-derived, no manual tuning)."""
    traffics = [float(t) for t in dag_info.get("edges", {}).values()]
    if not traffics:
        return 0.0
    return float(statistics.median(traffics)) * 2.0


def _selected_edges_mode_a(dag_info, t_critical: float) -> List[Tuple[str, str]]:
    selected = []
    for edge, traffic in dag_info.get("edges", {}).items():
        if float(traffic) > t_critical:
            selected.append(tuple(edge))
    return selected


def _connectivity_expand(entry_node: str, selected_edges: Iterable[Tuple[str, str]]) -> Set[str]:
    """Propagate from entry through high-traffic edges (§11.2.3 step 3)."""
    pattern: Set[str] = {entry_node}
    changed = True
    while changed:
        changed = False
        for src, dst in selected_edges:
            if src in pattern or dst in pattern:
                if src not in pattern:
                    pattern.add(src)
                    changed = True
                if dst not in pattern:
                    pattern.add(dst)
                    changed = True
    return pattern


def sort_pattern_by_migration_cost(nodes: List[str], dag_info) -> List[str]:
    """Order pattern nodes by transfer size (lighter first, reduces bandwidth contention)."""
    def _transfer_mb(node: str) -> float:
        props = dag_info["nodes"][node]
        return float(props.get("image_mb", 0.0)) + float(props.get("state_mb", 0.0))

    return sorted(nodes, key=_transfer_mb)


def build_colocate_pattern_mode_a(
    entry_node: str,
    target_server,
    assignments: Dict[str, int],
    dag_info,
    *,
    t_critical: Optional[float] = None,
) -> List[str]:
    """
    Build deployable nodes to colocate onto ``target_server`` (mode A).

    ``target_server`` / ``assignments`` are accepted for API parity with SA hooks;
    pattern topology depends only on DAG traffic graph and entry node.
    """
    del target_server, assignments
    if t_critical is None:
        t_critical = compute_t_critical(dag_info)
    selected = _selected_edges_mode_a(dag_info, t_critical)
    pattern_set = _connectivity_expand(entry_node, selected)
    pattern_set = {node for node in pattern_set if not is_external_node(node)}
    if entry_node not in pattern_set and not is_external_node(entry_node):
        pattern_set.add(entry_node)
    return sort_pattern_by_migration_cost(list(pattern_set), dag_info)


def count_high_traffic_colocated_edges(
    dag_info,
    assignments: Dict[str, int],
    *,
    t_critical: Optional[float] = None,
) -> Tuple[int, int]:
    """Return (colocated_count, total_critical_edges) for reporting."""
    if t_critical is None:
        t_critical = compute_t_critical(dag_info)
    colocated = 0
    total = 0
    for (src, dst), traffic in dag_info.get("edges", {}).items():
        if float(traffic) <= t_critical:
            continue
        if is_external_node(src) or is_external_node(dst):
            continue
        total += 1
        if assignments.get(src) == assignments.get(dst):
            colocated += 1
    return colocated, total
