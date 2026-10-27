from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from darwin.mesh_io import Mesh


@dataclass(frozen=True)
class Boundary:
    nodes: np.ndarray  # global node ids on this wall
    edges: np.ndarray  # global edge ids with both endpoints on this wall alone


@dataclass
class Topology:
    edges: np.ndarray  # (n_edges, 2) with n1 < n2
    cell_edges: np.ndarray  # (n_cells, n_edges_per_cell) with edge indices
    incidence_matrix: np.ndarray  # (n_edges, n_nodes)
    skin: Boundary # all boundary edges (n, 2) and nodes (n, )
    boundary: dict[str, Boundary] = field(default_factory=dict) # boundary edges (n, 2) and nodes (n, ) for each boundary tag


def _facet_size(n_nodes: int) -> int:
    """The number of nodes on a facet of a cell with ``n_nodes``."""
    if n_nodes == 3:
        return 2
    if n_nodes == 4:
        return 3
    raise ValueError(f"unsupported cell with {n_nodes} nodes")


def skin_facets(mesh: Mesh) -> list[tuple[int, ...]]:
    """Facets that belong to exactly one cell, as sorted node tuples. 
    For a 2D mesh a facet is a set of 2 nodes and for a 3D mesh a facet is a set of 3 nodes.

    Args:
        mesh: The mesh to get the skin facets from.

    Returns:
        A list of sorted node tuples representing the skin facets.
    """
    # get the number of nodes on a facet for the cell type
    size = _facet_size(mesh.cells.shape[1])
    counts: Counter[tuple[int, ...]] = Counter()
    for cell in mesh.cells:
        for facet in combinations((int(v) for v in cell), size):
            counts[tuple(sorted(facet))] += 1
    return [facet for facet, count in counts.items() if count == 1]


def tag_boundary(
    mesh: Mesh, predicates: dict[str, Callable[[np.ndarray], bool]]
) -> tuple[list[tuple[int, ...]], list[str]]:
    """Assign skin facets to named walls by midpoint predicates.

    Each predicate receives the facet midpoint of shape ``(dim,)`` and returns
    a bool. A predicate that matches no facet raises. A skin facet that matches
    no predicate is omitted from the returned lists and remains only on the skin.
    """
    if not predicates:
        raise ValueError("predicates must not be empty")

    facets = skin_facets(mesh)
    tagged_facets: list[tuple[int, ...]] = []
    tags: list[str] = []
    matched = {name: 0 for name in predicates}

    for facet in facets:
        midpoint = mesh.points[list(facet)].mean(axis=0)
        hits = [name for name, pred in predicates.items() if pred(midpoint)]
        if len(hits) > 1:
            raise ValueError(f"facet {facet} matched multiple tags: {hits}")
        if len(hits) == 1:
            name = hits[0]
            tagged_facets.append(facet)
            tags.append(name)
            matched[name] += 1

    for name, count in matched.items():
        if count == 0:
            raise ValueError(f"boundary tag {name!r} matched no skin facet")

    return tagged_facets, tags


def _edges_of_facets(
    facets: list[tuple[int, ...]], edge_ids: dict[tuple[int, int], int]
) -> set[int]:
    """Build a set of edge ids from facets."""
    edges: set[int] = set()
    for facet in facets:
        for a, b in combinations(facet, 2):
            key = (a, b) if a < b else (b, a)
            eid = edge_ids.get(key)
            if eid is not None:
                edges.add(eid)
    return edges


def _boundary_from_facets(
    facets: list[tuple[int, ...]], edge_ids: dict[tuple[int, int], int]
) -> Boundary:
    """Build a boundary from facets."""
    nodes = np.asarray(sorted({int(n) for facet in facets for n in facet}), dtype=int)
    edges = np.asarray(sorted(_edges_of_facets(facets, edge_ids)), dtype=int)
    return Boundary(nodes, edges)


def _boundaries_from_tags(
    facets: list[tuple[int, ...]],
    tags: list[str],
    edge_ids: dict[tuple[int, int], int],
) -> dict[str, Boundary]:
    """Build boundaries from tagged facets."""
    facets_by_tag: dict[str, list[tuple[int, ...]]] = defaultdict(list)
    for facet, tag in zip(facets, tags):
        facets_by_tag[tag].append(tuple(sorted(int(n) for n in facet)))

    edges_by_tag = {
        tag: _edges_of_facets(tag_facets, edge_ids) for tag, tag_facets in facets_by_tag.items()
    }
    # An edge that lies on two differently tagged walls is in neither set.
    ownership: Counter[int] = Counter()
    for edge_set in edges_by_tag.values():
        ownership.update(edge_set)

    boundary: dict[str, Boundary] = {}
    for tag, tag_facets in facets_by_tag.items():
        nodes = np.asarray(sorted({int(n) for facet in tag_facets for n in facet}), dtype=int)
        edges = np.asarray(
            sorted(eid for eid in edges_by_tag[tag] if ownership[eid] == 1),
            dtype=int,
        )
        boundary[tag] = Boundary(nodes, edges)
    return boundary


def build_topology(
    mesh: Mesh,
    boundary_tags: tuple[list[tuple[int, ...]], list[str]] | None = None,
) -> Topology:
    """Build a topology from a mesh and boundary tags."""
    edges: list[tuple[int, int]] = []
    edge_ids: dict[tuple[int, int], int] = {}
    cell_edges: list[list[int]] = []

    for cell_nodes in mesh.cells:
        edges_in_cell: list[int] = []
        for edge in combinations((int(v) for v in cell_nodes), 2):
            edge = tuple(sorted(edge))
            eid = edge_ids.get(edge)
            if eid is None:
                eid = len(edges)
                edge_ids[edge] = eid
                edges.append(edge)
            edges_in_cell.append(eid)
        cell_edges.append(edges_in_cell)

    edges_array = np.asarray(edges, dtype=int)
    cell_edges_array = np.asarray(cell_edges, dtype=int)

    n_edges = edges_array.shape[0]
    n_nodes = mesh.points.shape[0]
    incidence_matrix = np.zeros((n_edges, n_nodes), dtype=int)
    for e, (i, j) in enumerate(edges_array):
        incidence_matrix[e, i] = -1
        incidence_matrix[e, j] = +1

    skin_list = skin_facets(mesh)
    skin = _boundary_from_facets(skin_list, edge_ids)

    boundary: dict[str, Boundary] = {}
    if boundary_tags is not None:
        facets, tags = boundary_tags
        if len(facets) != len(tags):
            raise ValueError("facets and tags must have the same length")
        skin_set = set(skin_list)
        for facet in facets:
            key = tuple(sorted(int(n) for n in facet))
            if key not in skin_set:
                raise ValueError(f"tagged facet {facet} is not a skin facet")
        boundary = _boundaries_from_tags(facets, tags, edge_ids)

    return Topology(edges_array, cell_edges_array, incidence_matrix, skin, boundary)
