from collections import defaultdict, deque
from collections.abc import Iterable
from enum import Enum
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sparse

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
    """Return the gauged system.

    Tree–cotree replaces a spanning set of free edge rows by a_e = 0 and lists
    those edges in ``GaugeResult.tree_edges``. Coulomb appends Lagrange
    multipliers on ``System.p_dofs`` so that G_f^T a = 0; ``tree_edges`` is
    empty. Neither gauge changes nodal unknowns. A floating piece still needs
    its own phi reference.
    """
    if kind == GaugeKind.NONE:
        return GaugeResult(system, np.array([], dtype=int))
    if kind == GaugeKind.TREE:
        system, tree_edges = _apply_tree_cotree_gauge(system, topology)
        return GaugeResult(system, np.asarray(tree_edges, dtype=int))
    if kind == GaugeKind.COULOMB:
        return GaugeResult(*_apply_coulomb_gauge(system, topology))
    raise ValueError(f"unknown gauge kind {kind!r}")


def _is_identity_row(K: sparse.csr_matrix, i: int) -> bool:
    """True when row i is the statement that this one unknown equals the load."""
    row = K.getrow(i)
    return row.nnz == 1 and int(row.indices[0]) == i and np.isclose(row.data[0], 1.0)


def _connected_components(n_ids: int, ends: Iterable) -> list[list[int]]:
    """Union-find on 0..n_ids-1. Isolated ids stay singleton components."""
    parent = list(range(n_ids))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for n1, n2 in ends:
        r1, r2 = find(int(n1)), find(int(n2))
        if r1 != r2:
            parent[r2] = r1

    groups: dict[int, list[int]] = defaultdict(list)
    for node in range(n_ids):
        groups[find(node)].append(node)
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

    components = _connected_components(n_nodes, fixed_ends)
    component_of = [0] * n_nodes
    for i_comp, nodes in enumerate(components):
        for node in nodes:
            component_of[node] = i_comp

    # Connected pieces of the contracted graph: one tree is grown per piece.
    groups = _connected_components(
        len(components),
        ((component_of[n1], component_of[n2]) for n1, n2 in free_ends),
    )

    fixed_nodes = {node for pair in fixed_ends for node in pair}
    seen_component: set[int] = set()
    visited_nodes: set[int] = set()
    tree_edges: list[int] = []

    for group in groups:
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
            m_edge_globals=system.m_edge_globals,
        ),
        tree_edges,
    )


# apply coulomb gauge
def _free_dofs(
    k_globals: sparse.csr_matrix, n_edges: int, n_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Get the free degrees of freedom indices of the system. The dofs are separated into edge and node dofs."""
    free_edges = np.array(
        [e for e in range(n_edges) if not _is_identity_row(k_globals, e)], dtype=int
    )
    free_nodes = np.array(
        [n for n in range(n_nodes) if not _is_identity_row(k_globals, n + n_edges)],
        dtype=int,
    )
    return free_edges, free_nodes


def _multiplier_nodes(
    free_nodes: np.ndarray, components: list[list[int]]
) -> np.ndarray:
    """G columns that stay in C: one drop per piece with no Dirichlet node."""
    free = {int(n) for n in free_nodes}
    keep: list[int] = []
    for comp in components:
        free_in = sorted(n for n in comp if n in free)
        if not free_in:
            continue
        if any(n not in free for n in comp):
            keep.extend(free_in)
        else:
            keep.extend(free_in[:-1])
    return np.asarray(keep, dtype=int)


def _apply_coulomb_gauge(
    system: System, topology: Topology
) -> tuple[System, np.ndarray]:
    """Append multipliers that enforce G_f^T M_f a = 0.

    The constraint is the L2 (mass-weighted) Coulomb gauge, not the
    topological G_f^T a = 0. p lives on System.p_dofs. The constraint does
    not pin phi; tree_edges stay empty because no edge unknown is set to zero.
    """
    k = system.k_globals
    n_edges, n_nodes = topology.incidence_matrix.shape
    n_total = n_edges + n_nodes
    empty = np.array([], dtype=int)

    free_edges, free_nodes = _free_dofs(k, n_edges, n_nodes)
    keep = _multiplier_nodes(free_nodes, _connected_components(n_nodes, topology.edges))
    if keep.size == 0:
        return system, empty

    n_p = keep.size
    g_f = topology.incidence_matrix[np.ix_(free_edges, keep)].astype(complex)
    m_edge = system.m_edge_globals
    if m_edge is None:
        raise ValueError("Coulomb gauge requires system.m_edge_globals")

    m_free = m_edge[np.ix_(free_edges, free_edges)]  # (n_free, n_free)

    c = m_free @ g_f  # (n_free, n_p)

    c_full = sparse.lil_matrix((n_total, n_p), dtype=complex)
    c_full[free_edges, :] = c
    c_full = c_full.tocsr()

    z = sparse.csr_matrix((n_p, n_p), dtype=complex)
    k_augmented = sparse.bmat([[k, c_full], [c_full.T, z]], format="csr")

    f_augmented = np.concatenate([system.f_globals, np.zeros(n_p, dtype=complex)])

    p_dofs = np.arange(n_total, n_total + n_p)

    return (
        System(
            k_augmented,
            f_augmented,
            system.edge_dofs.copy(),
            system.node_dofs.copy(),
            p_dofs,
            system.m_edge_globals,
        ),
        np.array([], dtype=int),
    )
