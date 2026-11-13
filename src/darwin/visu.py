"""2D plots of a mesh and the problem configured on it."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from darwin.bc import DirichletWall, FloatingConductor
from darwin.material import Material
from darwin.mesh_io import Mesh
from darwin.topology import Topology

# Dark colors, chosen so a wall stays visible on a tab10 region face.
_WALL_COLORS = ("#c0392b", "#e67e22", "#1e8449", "#6c3483", "#1a5276", "#b7950b")


def plot_configuration(
    mesh: Mesh,
    topology: Topology,
    path: Path,
    materials: dict[int, Material] | None = None,
    walls: list[DirichletWall | FloatingConductor] | None = None,
    source: Callable | None = None,
    title: str = "",
) -> None:
    """Draw regions, named walls, and an optional volume current for a 2D mesh.

    Face color is the region id. A wall label comes from the matching boundary
    condition when one is passed; a tag with no condition is a magnetic wall.
    Wall edges are every skin edge whose two endpoints lie in that tag's node
    set: an interior chord between two walls is not a wall. An edge claimed by
    two tags is drawn once, for the first tag.
    """
    points = np.asarray(mesh.points, dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    region_handles = _draw_regions(ax, mesh, points, materials)
    source_handle = _draw_source(ax, points, mesh.cells, source) if source is not None else None
    wall_handles = _draw_walls(ax, points, topology, walls)

    handles = region_handles + wall_handles
    if source_handle is not None:
        handles.append(source_handle)
    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=0.95)

    span = points.max(axis=0) - points.min(axis=0)
    pad = 0.04 * float(max(span.max(), 1e-12))
    low, high = points.min(axis=0), points.max(axis=0)
    ax.set_xlim(low[0] - pad, high[0] + pad)
    ax.set_ylim(low[1] - pad, high[1] + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or "problem configuration")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _draw_regions(ax, mesh: Mesh, points: np.ndarray, materials: dict[int, Material] | None):
    region_ids = [int(r) for r in dict.fromkeys(int(r) for r in mesh.region_ids)]
    cmap = plt.get_cmap("tab10")
    color_of = {rid: cmap(i % 10) for i, rid in enumerate(region_ids)}
    faces = [points[cell] for cell in mesh.cells]
    colors = [color_of[int(rid)] for rid in mesh.region_ids]
    ax.add_collection(
        PolyCollection(faces, facecolors=colors, edgecolors="0.75", linewidths=0.3)
    )
    return [
        Patch(facecolor=color_of[rid], edgecolor="0.6", label=_region_label(mesh, rid, materials))
        for rid in region_ids
    ]


def _region_label(mesh: Mesh, region_id: int, materials: dict[int, Material] | None) -> str:
    name = mesh.region_names.get(region_id, str(region_id))
    if materials is None or region_id not in materials:
        return name
    material = materials[region_id]
    return f"{name}, σ={material.sigma:g}, μr={material.mu_r:g}, εr={material.eps_r:g}"


def _draw_source(ax, points: np.ndarray, cells: np.ndarray, source: Callable):
    centroids = np.array([points[cell].mean(axis=0) for cell in cells])
    current = np.vstack([np.real(np.asarray(source(x), dtype=complex)).ravel()[:2] for x in centroids])
    carries = np.linalg.norm(current, axis=1) > 0
    if not np.any(carries):
        return None
    step = max(1, int(carries.sum()) // 70)
    chosen = np.flatnonzero(carries)[::step]
    ax.quiver(
        centroids[chosen, 0],
        centroids[chosen, 1],
        current[chosen, 0],
        current[chosen, 1],
        color="0.1",
        angles="xy",
        scale_units="xy",
        scale=8,
        width=0.004,
    )
    return Line2D([0], [0], color="0.1", lw=1.5, label="source current J")


def _draw_walls(ax, points: np.ndarray, topology: Topology, walls: list | None):
    by_tag = {wall.tag: wall for wall in walls or []}
    claimed: set[int] = set()
    handles = []
    on_skin = {int(e) for e in topology.skin.edges}
    for i_tag, (tag, wall) in enumerate(topology.boundary.items()):
        nodes = {int(n) for n in wall.nodes}
        edge_ids = [
            eid
            for eid, (n1, n2) in enumerate(topology.edges)
            if eid in on_skin
            and int(n1) in nodes
            and int(n2) in nodes
            and eid not in claimed
        ]
        claimed.update(edge_ids)
        if not edge_ids:
            continue
        condition = by_tag.get(tag)
        color = _WALL_COLORS[i_tag % len(_WALL_COLORS)]
        width = 1.2 if condition is None else 2.5
        for eid in edge_ids:
            n1, n2 = topology.edges[eid]
            ax.plot(points[[n1, n2], 0], points[[n1, n2], 1], color=color, lw=width)
        handles.append(Line2D([0], [0], color=color, lw=width, label=_wall_label(tag, condition)))
    return handles


def _wall_label(tag: str, condition) -> str:
    if isinstance(condition, DirichletWall):
        return f"{tag}, φ={_fmt(condition.voltage)}"
    if isinstance(condition, FloatingConductor):
        return f"{tag}, floating, I={_fmt(condition.current)}"
    return f"{tag}, magnetic wall"


def _fmt(value) -> str:
    number = complex(value)
    if abs(number.imag) < 1e-15 * max(1.0, abs(number.real)):
        return f"{number.real:g}"
    return f"{number.real:g}{number.imag:+g}j"
