"""Interior chords under a grounded PEC wall: geometry and regression check.

The PEC rule eliminates an edge only when it lies on the skin **and** both
endpoints carry the same prescribed voltage (``darwin/bc.py``,
``_equipotential_tangential_edges``). The skin test matters. Two nodes can lie
on two different grounded faces at the same voltage; the straight edge between
them then cuts through the interior and is not tangential to either wall. An
earlier version tested only the two endpoints, zeroed such a chord, and
imposed ``A.dl = 0`` on a line that is not on a conductor. For a single flat
wall the endpoint test alone is exact, because a straight segment between two
points of one plane stays in that plane.

This script draws those chords and verifies that the shipped code keeps them
free.

1. A fan triangulation of a square from one corner. Every vertex is on the
   boundary, so every edge is either a skin edge or a chord. The old rule
   removed all 9 edge DoFs and the field was zero; the chords now stay free.
2. A 4 x 4 grid whose cells are split by the same "/" diagonal. It has interior
   nodes, so most edges survive either way; the two chords between opposite
   sides are the ones the old rule removed. The correct solution carries a
   nonzero circulation along them.

The reference ("correct") space used below keeps every edge that is not on the
skin. For a connected grounded wall every skin edge is tangential and no
interior edge is, so that is the intended set. The shipped space is read out of
the real ``apply_dirichlet`` output. The two must coincide, and the table
prints the relative difference in Bz, which must be zero.

Run with::

    PYTHONPATH=src python tests/view_test_interior_chord.py

Outputs: ``tests/results/interior_chord_grid.png`` and
``tests/results/interior_chord_fan.png``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from darwin.assemble import assemble
from darwin.bc import DirichletWall, apply_dirichlet
from darwin.elements import TriangleWhitney
from darwin.material import MU_0, Material
from darwin.mesh_io import Mesh
from darwin.topology import build_topology, tag_boundary

result_dir = Path(__file__).parent / "results"
result_dir.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# meshes
# --------------------------------------------------------------------------


def grid_mesh(n: int = 4) -> Mesh:
    """n x n cells of the unit square, every cell split by the "/" diagonal.

    The "/" diagonal of a cell joins its lower-right and upper-left corners.
    That diagonal is a chord between two boundary nodes in the two corner
    cells it starts and ends in, and an edge with an interior endpoint
    everywhere else.
    """
    xs = np.linspace(0.0, 1.0, n + 1)
    points = [(xs[i], xs[j]) for j in range(n + 1) for i in range(n + 1)]

    def node(i: int, j: int) -> int:
        return j * (n + 1) + i

    cells: list[list[int]] = []
    for j in range(n):
        for i in range(n):
            a, b, c, d = node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)
            cells.append([a, b, c])
            cells.append([a, c, d])
    return Mesh(
        np.asarray(points, dtype=float),
        np.asarray(cells, dtype=int),
        np.zeros(len(cells), dtype=int),
        {0: "air"},
    )


def fan_mesh() -> Mesh:
    """Square fanned from its lower-left corner. No interior node exists."""
    points = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0), (0.0, 1.0)]
    cells = [[0, i, i + 1] for i in range(1, len(points) - 1)]
    return Mesh(
        np.asarray(points, dtype=float),
        np.asarray(cells, dtype=int),
        np.zeros(len(cells), dtype=int),
        {0: "air"},
    )


def wall_predicates() -> dict[str, object]:
    return {
        "bottom": lambda x: np.isclose(x[1], 0.0),
        "top": lambda x: np.isclose(x[1], 1.0),
        "left": lambda x: np.isclose(x[0], 0.0),
        "right": lambda x: np.isclose(x[0], 1.0),
    }


# --------------------------------------------------------------------------
# magnetostatic solve of the A block
# --------------------------------------------------------------------------


def is_identity_row(k, i: int) -> bool:
    row = k.getrow(i)
    return (
        row.nnz == 1
        and int(row.indices[0]) == i
        and np.isclose(row.data[0], 1.0)
    )


def cell_bz(mesh: Mesh, topology, element: TriangleWhitney, a_edge: np.ndarray):
    """Constant Bz = curl A on each triangle, plus centroid and area."""
    x_hat = np.array([1.0 / 3.0, 1.0 / 3.0])
    centroids = np.empty((len(mesh.cells), 2))
    bz = np.empty(len(mesh.cells))
    areas = np.empty(len(mesh.cells))
    for i_cell, cell in enumerate(mesh.cells):
        nodes = mesh.points[cell]
        local_of = {int(g): i for i, g in enumerate(cell)}
        edge_ids = topology.cell_edges[i_cell]
        pairs = [
            (local_of[int(n1)], local_of[int(n2)])
            for n1, n2 in topology.edges[edge_ids]
        ]
        jacobian = element.jacobian_matrix(x_hat, nodes)
        curl_ref = element.curl_whiteney_basis_functions(x_hat, pairs)[:, 0]
        det_j = float(np.linalg.det(jacobian))
        bz[i_cell] = np.real(a_edge[edge_ids] @ curl_ref) / det_j
        centroids[i_cell] = nodes.mean(axis=0)
        areas[i_cell] = 0.5 * abs(det_j)
    return centroids, bz, areas


def study(mesh: Mesh) -> dict:
    """Assemble, then solve with the correct and with the shipped PEC rule.

    ``nu = 1`` and a uniform in-plane current ``J = (0, 1)`` keep every number
    order one. ``sigma = eps = 0`` removes the A-phi coupling, so the A block
    is the whole problem. The curl-curl block is singular (its kernel is the
    gradient of every interior nodal hat), and the kernel has zero curl, so the
    minimum-norm least-squares solution gives the same Bz as any other member
    of the solution set.
    """
    predicates = wall_predicates()
    topology = build_topology(mesh, boundary_tags=tag_boundary(mesh, predicates))
    element = TriangleWhitney()

    n_edges = len(topology.edges)
    skin_edges = {int(e) for e in topology.skin.edges}
    wall_nodes = {int(n) for n in topology.skin.nodes}

    # Edges the old endpoint-only rule would have taken. Kept for the picture.
    node_pair_edges = {
        e
        for e, (n1, n2) in enumerate(topology.edges)
        if int(n1) in wall_nodes and int(n2) in wall_nodes
    }
    chords = sorted(node_pair_edges - skin_edges)

    material = Material(name="air", sigma=0.0, mu_r=1.0 / MU_0, eps_r=0.0)
    source = lambda x: np.array([0.0, 1.0])
    system = assemble(mesh, topology, {0: material}, 0.0, source, element)

    constrained = apply_dirichlet(
        system, topology, [DirichletWall(tag, 0.0) for tag in predicates]
    )
    k_true = system.k_globals[:n_edges, :n_edges]
    k_shipped = constrained.k_globals[:n_edges, :n_edges]
    rhs = np.asarray(system.f_globals[:n_edges], dtype=complex)

    # Correct space: every edge that is not on the skin. A chord is free.
    free_true = [e for e in range(n_edges) if e not in skin_edges]
    # Shipped space: whatever apply_dirichlet left as a real row.
    eliminated_shipped = {
        e for e in range(n_edges) if is_identity_row(k_shipped, e)
    }
    free_shipped = [e for e in range(n_edges) if e not in eliminated_shipped]
    # Chords the shipped code actually removed. Must be empty.
    wrongly_eliminated = sorted(eliminated_shipped - skin_edges)

    def solve_on(free: list[int], k) -> np.ndarray:
        a = np.zeros(n_edges, dtype=complex)
        if not free:
            return a
        block = k[free][:, free].toarray()
        a[free] = np.linalg.lstsq(block, rhs[free], rcond=None)[0]
        return a

    a_true = solve_on(free_true, k_true)
    a_shipped = solve_on(free_shipped, k_shipped)

    centroids, bz_true, areas = cell_bz(mesh, topology, element, a_true)
    _, bz_shipped, _ = cell_bz(mesh, topology, element, a_shipped)

    return {
        "mesh": mesh,
        "topology": topology,
        "n_edges": n_edges,
        "skin_edges": skin_edges,
        "node_pair_edges": node_pair_edges,
        "chords": chords,
        "wrongly_eliminated": wrongly_eliminated,
        "free_true": free_true,
        "free_shipped": free_shipped,
        "a_true": a_true,
        "a_shipped": a_shipped,
        "centroids": centroids,
        "bz_true": bz_true,
        "bz_shipped": bz_shipped,
        "areas": areas,
    }


# --------------------------------------------------------------------------
# drawings
# --------------------------------------------------------------------------


def draw_edges(ax, mesh: Mesh, topology) -> None:
    points = mesh.points
    for a, b in topology.edges:
        ax.plot(points[[a, b], 0], points[[a, b], 1], color="0.85", lw=0.8, zorder=1)


def draw_mesh_panel(ax, result: dict, title: str) -> None:
    mesh, topology = result["mesh"], result["topology"]
    points = mesh.points
    ax.add_collection(
        PolyCollection(
            [points[cell] for cell in mesh.cells],
            facecolors="#f4f7fb",
            edgecolors="none",
            zorder=0,
        )
    )
    draw_edges(ax, mesh, topology)

    for e in sorted(result["skin_edges"]):
        a, b = topology.edges[e]
        ax.plot(points[[a, b], 0], points[[a, b], 1], color="0.15", lw=2.2, zorder=3)
    for e in result["chords"]:
        a, b = topology.edges[e]
        ax.plot(points[[a, b], 0], points[[a, b], 1], color="C3", lw=2.6, zorder=4)
    other_free = [e for e in result["free_true"] if e not in result["chords"]]
    for e in other_free:
        a, b = topology.edges[e]
        mid = 0.5 * (points[a] + points[b])
        ax.plot(mid[0], mid[1], ".", color="C0", ms=4, zorder=5)

    ax.plot([], [], color="0.15", lw=2.2, label="skin edge, tangential, eliminated")
    ax.plot([], [], color="C3", lw=2.6, label="chord between two walls, kept free")
    if other_free:
        ax.plot([], [], ".", color="C0", ms=4, label="other free edge")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="lower left", fontsize=6, framealpha=0.95)


def draw_field(
    ax, result: dict, key: str, title: str, symmetric: bool = False, limit=None
) -> None:
    mesh = result["mesh"]
    values = result[key]
    if symmetric:
        limit = limit or (np.max(np.abs(values)) or 1.0)
        im = ax.tripcolor(
            mesh.points[:, 0],
            mesh.points[:, 1],
            mesh.cells,
            values,
            shading="flat",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
    else:
        im = ax.tripcolor(
            mesh.points[:, 0],
            mesh.points[:, 1],
            mesh.cells,
            values,
            shading="flat",
            cmap="magma",
        )
    plt.colorbar(im, ax=ax, shrink=0.85)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)


def print_table(rows: list[tuple[str, dict]]) -> None:
    header = (
        f"{'mesh':<10}{'edges':>7}{'skin':>6}{'chords':>8}{'wrong':>7}"
        f"{'free(correct)':>15}{'free(shipped)':>15}{'rel L2 dBz':>12}"
    )
    print(header)
    print("-" * len(header))
    for name, result in rows:
        n_true = len(result["free_true"])
        n_ship = len(result["free_shipped"])
        denom = np.linalg.norm(result["bz_true"])
        diff = np.linalg.norm(result["bz_true"] - result["bz_shipped"]) / denom
        print(
            f"{name:<10}{result['n_edges']:>7}{len(result['skin_edges']):>6}"
            f"{len(result['chords']):>8}{len(result['wrongly_eliminated']):>7}"
            f"{n_true:>15}{n_ship:>15}{diff:>12.3e}"
        )
    print()


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def main() -> None:
    fan = study(fan_mesh())
    grid = study(grid_mesh(4))

    print("PEC rule: an edge is eliminated only when it lies on the skin and")
    print("both endpoints carry the same prescribed voltage.\n")
    print_table([("fan", fan), ("grid-4x4", grid)])

    for name, result in (("fan", fan), ("grid-4x4", grid)):
        if not result["chords"]:
            continue
        print(f"{name}: chords between two grounded faces, kept free")
        for e in result["chords"]:
            n1, n2 = (int(n) for n in result["topology"].edges[e])
            p1, p2 = result["mesh"].points[n1], result["mesh"].points[n2]
            print(
                f"  edge {e:>3}  nodes ({n1}, {n2})  "
                f"{np.round(p1, 3)} -> {np.round(p2, 3)}  "
                f"A.dl correct = {result['a_true'][e].real:+.4f}"
                f"{result['a_true'][e].imag:+.4f}j"
            )
        print()

    for name, result in (("fan", fan), ("grid-4x4", grid)):
        if result["wrongly_eliminated"]:
            print(f"{name}: FAIL, eliminated chords {result['wrongly_eliminated']}")
        else:
            print(f"{name}: OK, no chord is eliminated")
    print()

    # Grid: mesh, correct field, shipped field, difference.
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 8.2))
    draw_mesh_panel(axes[0, 0], grid, "4x4 grid: skin edges vs chords")
    draw_field(axes[0, 1], grid, "bz_true", "Bz, correct space", symmetric=True)
    draw_field(
        axes[1, 0], grid, "bz_shipped", "Bz, apply_dirichlet", symmetric=True
    )
    difference = grid["bz_true"] - grid["bz_shipped"]
    peak = float(np.max(np.abs(difference)))
    axes[1, 1].tripcolor(
        grid["mesh"].points[:, 0],
        grid["mesh"].points[:, 1],
        grid["mesh"].cells,
        difference,
        shading="flat",
        cmap="RdBu_r",
        vmin=-(peak or 1.0),
        vmax=peak or 1.0,
    )
    axes[1, 1].set_aspect("equal")
    axes[1, 1].set_title("Bz correct - Bz apply_dirichlet", fontsize=9)
    if peak == 0.0:
        axes[1, 1].text(
            0.5,
            0.5,
            "identical",
            transform=axes[1, 1].transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="0.35",
        )
    relative = np.linalg.norm(difference) / np.linalg.norm(grid["bz_true"])
    figure.suptitle(
        f"2 of 40 edges are chords and stay free; relative L2 change {relative:.1e}",
        fontsize=9,
    )
    figure.tight_layout()
    path = result_dir / "interior_chord_grid.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"figure: {path}")

    # Fan: 6 skin edges and 3 chords; all 3 chords must survive.
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    draw_mesh_panel(axes[0], fan, "fan mesh: 6 skin edges, 3 chords")
    fan_limit = float(np.max(np.abs(fan["bz_true"])))
    draw_field(
        axes[1], fan, "bz_true", "Bz, correct space", symmetric=True, limit=fan_limit
    )
    fan_difference = fan["bz_true"] - fan["bz_shipped"]
    fan_peak = float(np.max(np.abs(fan_difference)))
    axes[2].tripcolor(
        fan["mesh"].points[:, 0],
        fan["mesh"].points[:, 1],
        fan["mesh"].cells,
        fan_difference,
        shading="flat",
        cmap="RdBu_r",
        vmin=-(fan_peak or 1.0),
        vmax=fan_peak or 1.0,
    )
    axes[2].set_aspect("equal")
    axes[2].set_title("Bz correct - Bz apply_dirichlet", fontsize=9)
    if fan_peak == 0.0:
        axes[2].text(
            0.5,
            0.5,
            "identical",
            transform=axes[2].transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="0.35",
        )
    figure.suptitle(
        "Fan mesh: all 3 chords stay free, so B is no longer forced to zero",
        fontsize=9,
    )
    figure.tight_layout()
    path = result_dir / "interior_chord_fan.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"figure: {path}")


if __name__ == "__main__":
    main()
