"""Visualise region ids and named boundary walls on a 2D square and a 3D cube.

Run from the package root:
    python tests/view_test_regions_boundaries.py
Figures are written next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from darwin.material import Material, materials_by_region
from darwin.mesh_io import Mesh
from darwin.topology import build_topology, skin_facets, tag_boundary

OUT = Path(__file__).resolve().parent
TAG_COLORS = {
    "bottom": "C3",
    "top": "C0",
    "left": "C2",
    "right": "C1",
    "front": "C4",
    "back": "C5",
    "xmin": "C2",
    "xmax": "C1",
    "ymin": "C3",
    "ymax": "C0",
    "zmin": "C4",
    "zmax": "C5",
}


def square_mesh() -> Mesh:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cells = np.array([[0, 1, 3], [1, 2, 3]])
    return Mesh(points, cells, np.array([0, 0]), region_names={0: "air"})


def cube_mesh() -> Mesh:
    # Corner numbering:
    #   7------6
    #  /|     /|
    # 4------5 |
    # | 3----|-2
    # |/     |/
    # 0------1
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    # Six tets that fill the unit cube and share the space diagonal 0–6.
    cells = np.array(
        [
            [0, 1, 2, 6],
            [0, 2, 3, 6],
            [0, 1, 5, 6],
            [0, 4, 5, 6],
            [0, 3, 7, 6],
            [0, 4, 7, 6],
        ]
    )
    return Mesh(points, cells, np.zeros(6, dtype=int), region_names={0: "air"})


def print_summary(title: str, mesh: Mesh, topology, materials):
    print(f"=== {title} ===")
    print(f"cells = {len(mesh.cells)}, edges = {len(topology.edges)}")
    print(f"region_ids = {mesh.region_ids.tolist()}")
    print(f"region_names = {mesh.region_names}")
    print(f"materials = {{{', '.join(f'{rid}: {mat.name!r}' for rid, mat in materials.items())}}}")
    print(f"skin nodes = {topology.skin.nodes.tolist()}")
    print(f"skin edges = {[tuple(map(int, topology.edges[e])) for e in topology.skin.edges]}")
    for tag, wall in sorted(topology.boundary.items()):
        edge_pairs = [tuple(map(int, topology.edges[e])) for e in wall.edges]
        print(f"  {tag}: nodes={wall.nodes.tolist()}, exclusive edges={edge_pairs}")
    print()


def plot_square(mesh: Mesh, topology, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))

    # Left: region ids on the two triangles.
    ax = axes[0]
    polys = [mesh.points[cell] for cell in mesh.cells]
    faces = PolyCollection(
        polys, facecolors=["#cfe8ff", "#ffe0b3"], edgecolors="black", linewidths=1.2
    )
    ax.add_collection(faces)
    for cell_id, cell in enumerate(mesh.cells):
        centroid = mesh.points[cell].mean(axis=0)
        rid = int(mesh.region_ids[cell_id])
        name = mesh.region_names.get(rid, str(rid))
        ax.text(
            centroid[0],
            centroid[1],
            f"cell {cell_id}\nregion {rid}\n({name})",
            ha="center",
            va="center",
            fontsize=9,
        )
    for nid, xy in enumerate(mesh.points):
        ax.plot(*xy, "ko", markersize=4)
        ax.text(xy[0] + 0.03, xy[1] + 0.03, str(nid), fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.15)
    ax.set_title("2D square: region ids")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Right: named boundary edges.
    ax = axes[1]
    for cell in mesh.cells:
        tri = np.vstack([mesh.points[cell], mesh.points[cell[0]]])
        ax.plot(tri[:, 0], tri[:, 1], color="0.7", linewidth=1.0)
    for tag, wall in topology.boundary.items():
        color = TAG_COLORS.get(tag, "black")
        for eid in wall.edges:
            a, b = topology.edges[eid]
            ax.plot(
                mesh.points[[a, b], 0],
                mesh.points[[a, b], 1],
                color=color,
                linewidth=3.0,
                label=tag,
            )
            mid = mesh.points[[a, b]].mean(axis=0)
            ax.text(mid[0], mid[1], tag, color=color, fontsize=8, ha="center", va="bottom")
        ax.plot(
            mesh.points[wall.nodes, 0],
            mesh.points[wall.nodes, 1],
            "o",
            color=color,
            markersize=7,
        )
    handles = [
        plt.Line2D([0], [0], color=TAG_COLORS[t], lw=3, label=t)
        for t in ("bottom", "top", "left", "right")
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.15)
    ax.set_title("2D square: boundary tags")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"figure: {path}")
    plt.show()


def plot_cube(mesh: Mesh, topology, path: Path):
    fig = plt.figure(figsize=(10.5, 4.6))

    # Left: wireframe of the six tets with region labels at centroids.
    ax = fig.add_subplot(121, projection="3d")
    for cell_id, cell in enumerate(mesh.cells):
        for a, b in combinations_edges(cell):
            ax.plot(*mesh.points[[a, b]].T, color="0.55", linewidth=0.9)
        centroid = mesh.points[cell].mean(axis=0)
        rid = int(mesh.region_ids[cell_id])
        ax.text(*centroid, f"{cell_id}/{rid}", fontsize=7, ha="center")
    for nid, xyz in enumerate(mesh.points):
        ax.scatter(*xyz, color="black", s=18)
        ax.text(xyz[0] + 0.04, xyz[1] + 0.04, xyz[2] + 0.04, str(nid), fontsize=7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("3D cube: six tets, region ids")
    _set_equal_3d(ax)

    # Right: coloured skin faces by boundary tag.
    ax = fig.add_subplot(122, projection="3d")
    for a, b in topology.edges:
        ax.plot(*mesh.points[[a, b]].T, color="0.75", linewidth=0.7)
    # Rebuild facet → tag from the same predicates used in main.
    predicates = cube_predicates()
    facets, tags = tag_boundary(mesh, predicates)
    for facet, tag in zip(facets, tags):
        color = TAG_COLORS.get(tag, "gray")
        tri = mesh.points[list(facet)]
        poly = Poly3DCollection([tri], alpha=0.45, facecolor=color, edgecolor="black")
        ax.add_collection3d(poly)
        mid = tri.mean(axis=0)
        ax.text(*mid, tag, color=color, fontsize=7, ha="center")
    handles = [
        plt.Line2D([0], [0], color=TAG_COLORS[t], lw=6, label=t)
        for t in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("3D cube: boundary face tags")
    _set_equal_3d(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"figure: {path}")
    plt.show()


def combinations_edges(cell):
    from itertools import combinations

    return combinations(cell, 2)


def _set_equal_3d(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_box_aspect((1, 1, 1))


def square_predicates():
    return {
        "bottom": lambda x: np.isclose(x[1], 0.0),
        "top": lambda x: np.isclose(x[1], 1.0),
        "left": lambda x: np.isclose(x[0], 0.0),
        "right": lambda x: np.isclose(x[0], 1.0),
    }


def cube_predicates():
    return {
        "xmin": lambda x: np.isclose(x[0], 0.0),
        "xmax": lambda x: np.isclose(x[0], 1.0),
        "ymin": lambda x: np.isclose(x[1], 0.0),
        "ymax": lambda x: np.isclose(x[1], 1.0),
        "zmin": lambda x: np.isclose(x[2], 0.0),
        "zmax": lambda x: np.isclose(x[2], 1.0),
    }


def main():
    air = Material(name="air", sigma=0.0, mu_r=1.0, eps_r=1.0)
    library = {"air": air}

    mesh2 = square_mesh()
    materials2 = materials_by_region(mesh2, library)
    facets2, tags2 = tag_boundary(mesh2, square_predicates())
    topo2 = build_topology(mesh2, boundary_tags=(facets2, tags2))
    print_summary("2D square (2 triangles)", mesh2, topo2, materials2)
    assert len(skin_facets(mesh2)) == 4
    plot_square(mesh2, topo2, OUT / "regions_boundaries_square.png")

    mesh3 = cube_mesh()
    materials3 = materials_by_region(mesh3, library)
    facets3, tags3 = tag_boundary(mesh3, cube_predicates())
    topo3 = build_topology(mesh3, boundary_tags=(facets3, tags3))
    print_summary("3D cube (6 tetrahedra)", mesh3, topo3, materials3)
    assert len(mesh3.cells) == 6
    assert len(skin_facets(mesh3)) == 12  # 6 faces × 2 triangles
    assert set(topo3.boundary) == {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}
    plot_cube(mesh3, topo3, OUT / "regions_boundaries_cube.png")
    print("region/boundary visuals done")


if __name__ == "__main__":
    main()
