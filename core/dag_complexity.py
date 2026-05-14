"""Utilities for DAG complexity grouping and traceable DAG extraction."""

from collections import defaultdict, deque


def dag_complexity_features(dag_info):
    nodes = dag_info.get("nodes", {})
    edges = dag_info.get("edges", {})
    stateful = sum(1 for props in nodes.values() if props.get("is_stateful"))
    max_traffic = max(edges.values()) if edges else 0
    return {
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "stateful_nodes": stateful,
        "stateful_ratio": (stateful / len(nodes)) if nodes else 0.0,
        "max_traffic": max_traffic,
        "avg_degree": (2.0 * len(edges) / len(nodes)) if nodes else 0.0,
    }


def classify_dag_complexity(dag_info):
    f = dag_complexity_features(dag_info)
    if f["num_nodes"] >= 8 or f["num_edges"] >= 10 or f["avg_degree"] >= 2.5:
        return "complex"
    if f["num_nodes"] >= 5 or f["num_edges"] >= 5:
        return "medium"
    return "simple"


def weak_components_from_edges(edge_rows, src_col="src", dst_col="dst"):
    graph = defaultdict(set)
    nodes = set()
    for row in edge_rows:
        src = str(row[src_col])
        dst = str(row[dst_col])
        nodes.add(src)
        nodes.add(dst)
        graph[src].add(dst)
        graph[dst].add(src)

    seen = set()
    components = []
    for node in sorted(nodes):
        if node in seen:
            continue
        q = deque([node])
        seen.add(node)
        comp = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nb in graph[cur]:
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        components.append(sorted(comp))
    return components


def dagify_component_edges(edge_rows, component_nodes, src_col="src", dst_col="dst", traffic_col="traffic"):
    """
    Convert a raw call subgraph into an acyclic DAG by keeping edges that follow
    a deterministic node order.  Dropped back-edges are returned for provenance.
    """
    comp = set(component_nodes)
    order = {node: i for i, node in enumerate(sorted(comp))}
    kept = {}
    dropped = []
    for row in edge_rows:
        src = str(row[src_col])
        dst = str(row[dst_col])
        if src not in comp or dst not in comp or src == dst:
            continue
        traffic = float(row.get(traffic_col, 1.0))
        if order[src] < order[dst]:
            kept[(src, dst)] = kept.get((src, dst), 0.0) + traffic
        else:
            dropped.append({"src": src, "dst": dst, "traffic": traffic, "reason": "back_edge_or_cycle"})
    return kept, dropped
