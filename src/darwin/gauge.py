from collections import defaultdict, deque
from enum import Enum
from dataclasses import dataclass
import numpy as np

from darwin.assemble import System
from darwin.topology import Topology


class GaugeKind(Enum):
    """Gauge applied to the edge unknowns after the boundary conditions."""

    NONE = 0
    TREE = 1
    COULOMB = 2


@dataclass(frozen=True)
class GaugeResult:
    system: System
    tree_edges: np.ndarray


def apply_gauge(
    system: System, topology: Topology, kind: GaugeKind = GaugeKind.NONE
) -> GaugeResult:
    """Return the gauged system and the tree edges written by the tree gauge.

    Tree–cotree replaces a spanning set of free edge rows by a_e = 0. It does
    not change nodal unknowns. Coulomb is not implemented.
    """
    if kind == GaugeKind.NONE:
        return GaugeResult(system, np.array([], dtype=int))
    if kind == GaugeKind.TREE:
        system, tree_edges = _apply_tree_cotree_gauge(system, topology)
        return GaugeResult(system, np.asarray(tree_edges, dtype=int))
    if kind == GaugeKind.COULOMB:
        raise NotImplementedError("Coulomb gauge is not implemented")
    raise ValueError(f"unknown gauge kind {kind!r}")


def _is_identity_row(K, i: int) -> bool:
    """True when row i is the statement that this one unknown equals the load."""
    row = K.getrow(i)
    return row.nnz == 1 and int(row.indices[0]) == i and np.isclose(row.data[0], 1.0)


def _fixed_edge_components(
    n_nodes: int, fixed_ends: list[tuple[int, int]]
) -> list[set[int]]:
    """Nodes joined by edges that are already a_e = 0, plus every other node alone."""
    parent = list(range(n_nodes))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for n1, n2 in fixed_ends:
        r1, r2 = find(n1), find(n2)
        if r1 != r2:
            parent[r2] = r1

    groups: dict[int, set[int]] = defaultdict(set)
    for node in range(n_nodes):
        groups[find(node)].add(node)
    return list(groups.values())


def _apply_tree_cotree_gauge(system: System, topology: Topology) -> System:
    """Set one spanning tree of the free edges to zero on every component.

    Identity edge rows are edges already eliminated by a boundary condition.
    Their connected components share one gauge value. Free edges contract
    those components into a graph; every connected piece of that graph is
    gauged on its own, starting from the component that holds the lowest fixed
    node, or from the component that holds the lowest node when the piece has
    no fixed edge. A genuinely disconnected mesh therefore gets one tree per
    connected piece. Each piece still carries one constant, so each piece needs
    its own phi reference. Nodal rows are not roots: a prescribed phi does not
    fix the edge gradient.
    """
    k_gauge = system.k_globals.tolil(copy=True)
    f_gauge = np.array(system.f_globals, dtype=complex, copy=True)
    n_edges, n_nodes = topology.incidence_matrix.shape

    neighbors: dict[int, list[tuple[int, int]]] = defaultdict(list)
    fixed_ends: list[tuple[int, int]] = []
    free_ends: list[tuple[int, int]] = []
    for e in range(n_edges):
        n1, n2 = (int(n) for n in topology.edges[e])
        if _is_identity_row(k_gauge, e):
            fixed_ends.append((n1, n2))
        else:
            free_ends.append((n1, n2))
            neighbors[n1].append((n2, e))
            neighbors[n2].append((n1, e))

    components = _fixed_edge_components(n_nodes, fixed_ends)
    component_of = [0] * n_nodes
    for i_comp, nodes in enumerate(components):
        for node in nodes:
            component_of[node] = i_comp

    # Connected pieces of the contracted graph: one tree is grown per piece.
    parent = list(range(len(components)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for n1, n2 in free_ends:
        r1, r2 = find(component_of[n1]), find(component_of[n2])
        if r1 != r2:
            parent[r2] = r1

    groups: dict[int, list[int]] = defaultdict(list)
    for i_comp in range(len(components)):
        groups[find(i_comp)].append(i_comp)

    fixed_nodes = {node for pair in fixed_ends for node in pair}
    seen_component: set[int] = set()
    visited_nodes: set[int] = set()
    tree_edges: list[int] = []

    for group in groups.values():
        fixed_members = [
            c for c in group if any(node in fixed_nodes for node in components[c])
        ]
        if fixed_members:
            root = min(
                fixed_members,
                key=lambda c: min(
                    node for node in components[c] if node in fixed_nodes
                ),
            )
        else:
            root = min(group, key=lambda c: min(components[c]))

        visited = set(components[root])
        seen_component.add(root)
        queue = deque(visited)
        while queue:
            u = queue.popleft()
            for v, e in neighbors[u]:
                comp = component_of[v]
                if comp in seen_component:
                    continue
                seen_component.add(comp)
                for node in components[comp]:
                    visited.add(node)
                    queue.append(node)
                tree_edges.append(e)
        visited_nodes.update(visited)

    if len(visited_nodes) != n_nodes:
        raise RuntimeError(
            f"{n_nodes - len(visited_nodes)} nodes are not reachable from the root nodes"
        )

    for e in tree_edges:
        k_gauge[e, :] = 0.0
        k_gauge[:, e] = 0.0
        k_gauge[e, e] = 1.0
        f_gauge[e] = 0.0

    return (
        System(
            k_gauge.tocsr(),
            f_gauge,
            system.edge_dofs.copy(),
            system.node_dofs.copy(),
        ),
        tree_edges,
    )
