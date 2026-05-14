"""
Traceable microservice DAG extraction helper.

This script does not fabricate topology.  It consumes an existing call edge
table and emits candidate DAG definitions plus provenance metadata.

Expected edge columns by default: src,dst,traffic
Optional node attribute columns can be supplied in a separate CSV:
node,image_mb,state_mb,is_stateful
"""

import argparse
import json
import os

import pandas as pd

from core.dag_complexity import (
    classify_dag_complexity,
    dag_complexity_features,
    dagify_component_edges,
    weak_components_from_edges,
)


def _load_node_attrs(path):
    if not path:
        return {}
    df = pd.read_csv(path)
    required = {"node", "image_mb", "state_mb", "is_stateful"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"node attrs missing columns: {sorted(missing)}")
    out = {}
    for _, row in df.iterrows():
        out[str(row["node"])] = {
            "image_mb": float(row["image_mb"]),
            "state_mb": float(row["state_mb"]),
            "is_stateful": bool(row["is_stateful"]),
        }
    return out


def extract_dags(edge_csv, node_attrs_csv=None, min_nodes=4, max_nodes=12):
    edges_df = pd.read_csv(edge_csv)
    rows = edges_df.to_dict("records")
    attrs = _load_node_attrs(node_attrs_csv)
    components = weak_components_from_edges(rows)
    dags = {}
    provenance = []

    for idx, comp in enumerate(components, 1):
        if len(comp) < min_nodes or len(comp) > max_nodes:
            continue
        dag_edges, dropped = dagify_component_edges(rows, comp)
        if not dag_edges:
            continue
        nodes = {
            node: attrs.get(
                node,
                {"image_mb": 100.0, "state_mb": 0.0, "is_stateful": False},
            )
            for node in comp
        }
        dag_info = {"probability": 0.0, "nodes": nodes, "edges": dag_edges}
        name = f"Extracted_DAG_{idx:03d}"
        dags[name] = dag_info
        provenance.append(
            {
                "name": name,
                "source_edge_csv": os.path.abspath(edge_csv),
                "source_node_attrs_csv": os.path.abspath(node_attrs_csv) if node_attrs_csv else None,
                "component_nodes": comp,
                "dropped_edges": dropped,
                "complexity": classify_dag_complexity(dag_info),
                "features": dag_complexity_features(dag_info),
                "note": "Default node attrs are used only when no source node attribute row exists.",
            }
        )
    return dags, provenance


def _jsonable_dags(dags):
    out = {}
    for name, dag in dags.items():
        out[name] = {
            "probability": dag["probability"],
            "nodes": dag["nodes"],
            "edges": {f"{src}->{dst}": traffic for (src, dst), traffic in dag["edges"].items()},
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Extract traceable candidate microservice DAGs")
    parser.add_argument("--edges", required=True, help="CSV with src,dst,traffic columns")
    parser.add_argument("--node-attrs", default=None, help="Optional node attributes CSV")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--min-nodes", type=int, default=4)
    parser.add_argument("--max-nodes", type=int, default=12)
    args = parser.parse_args()

    dags, provenance = extract_dags(
        args.edges,
        node_attrs_csv=args.node_attrs,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
    )
    payload = {"dags": _jsonable_dags(dags), "provenance": provenance}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(dags)} DAG candidates to {args.out}")


if __name__ == "__main__":
    main()
