from collections import deque

import numpy as np

from core.microservice_dags import MICROSERVICE_DAGS


EXTERNAL_NODE_NAMES = {"USER", "UNKNOWN", "UNAVAILABLE"}


def is_external_node(node_name):
    """Return True for nodes that are topology context, not deployable services."""
    return str(node_name).upper() in EXTERNAL_NODE_NAMES


def get_deployable_nodes(dag_info):
    """Return service nodes that can be controlled/migrated."""
    return [node for node in dag_info["nodes"] if not is_external_node(node)]


def get_entry_nodes(dag_info):
    """Return nodes with in-degree == 0 (DAG entry points)."""
    all_nodes = set(dag_info['nodes'].keys())
    nodes_with_incoming = set()
    for (src, dst) in dag_info['edges'].keys():
        nodes_with_incoming.add(dst)
    return list(all_nodes - nodes_with_incoming)


def get_service_entry_nodes(dag_info):
    """
    Return deployable entry services for SLA/access calculations.

    External nodes such as USER/UNKNOWN/UNAVAILABLE are retained in the graph for
    topology context, but they are not deployable service entries.
    """
    deployable = set(get_deployable_nodes(dag_info))
    if not deployable:
        return []

    deployable_with_internal_incoming = set()
    direct_external_targets = set()
    for src, dst in dag_info["edges"]:
        if dst not in deployable:
            continue
        if src in deployable:
            deployable_with_internal_incoming.add(dst)
        elif is_external_node(src):
            direct_external_targets.add(dst)

    service_entries = direct_external_targets or (deployable - deployable_with_internal_incoming)
    if service_entries:
        return list(service_entries)

    topo = topological_sort(dag_info)
    for node in topo:
        if node in deployable:
            return [node]
    return list(deployable)


def topological_sort(dag_info):
    """BFS topological sort (Kahn's algorithm)."""
    all_nodes = list(dag_info['nodes'].keys())
    in_degree = {n: 0 for n in all_nodes}
    adj = {n: [] for n in all_nodes}
    for (src, dst) in dag_info['edges'].keys():
        adj[src].append(dst)
        in_degree[dst] += 1

    queue = deque([n for n in all_nodes if in_degree[n] == 0])
    sorted_nodes = []
    while queue:
        node = queue.popleft()
        sorted_nodes.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    if len(sorted_nodes) < len(all_nodes):
        # Extracted production call graphs may contain small cycles. Keep a
        # stable order for unresolved nodes so every service remains controllable.
        seen = set(sorted_nodes)
        sorted_nodes.extend([node for node in all_nodes if node not in seen])
    return sorted_nodes


def assign_dag_type():
    """Randomly assign a DAG type based on configured probabilities."""
    dag_names = list(MICROSERVICE_DAGS.keys())
    dag_probs = [MICROSERVICE_DAGS[d]['probability'] for d in dag_names]
    return np.random.choice(dag_names, p=dag_probs)


def initialize_dag_assignment(dag_type, nearest_server_id):
    """Deploy all microservice nodes of a DAG onto a single server."""
    dag_info = MICROSERVICE_DAGS[dag_type]
    return {node: nearest_server_id for node in dag_info['nodes']}
