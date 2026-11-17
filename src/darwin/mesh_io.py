"""Mesh container and a gmsh ASCII reader.

Solver meshes are normally built in Python.  :func:`read_msh` reads the ASCII
``.msh`` files that gmsh writes (versions 2.2 and 4.1), so a mesh can come from
gmsh instead.  Physical groups become the mesh's regions (dimension ``d``
groups) and boundary tags (dimension ``d - 1`` groups), which is what
:func:`darwin.topology.build_topology` consumes.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Mesh:
    points: np.ndarray  # (n_points, n_dims)
    cells: np.ndarray  # (n_cells, n_nodes)
    region_ids: np.ndarray  # (n_cells,), cell i has region region_ids[i]
    region_names: dict[int, str] = field(default_factory=dict)
    # sorted tuple of global node ids of a boundary facet -> tag name, e.g.
    # from a gmsh physical group. build_topology uses it when no explicit
    # boundary tags are passed.
    facet_tags: dict[tuple[int, ...], str] = field(default_factory=dict)


# gmsh element type -> number of nodes, for the types this reader accepts.
_ELEMENT_NODES = {1: 2, 2: 3, 4: 4, 15: 1}
_ELEMENT_DIM = {1: 1, 2: 2, 4: 3, 15: 0}


def _sections(lines: list[str]) -> dict[str, list[str]]:
    """Split a gmsh file into its ``$Section`` bodies."""
    sections: dict[str, list[str]] = {}
    name: str | None = None
    body: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("$"):
            if name is not None:
                sections[name] = body
            name = None if line.startswith("$End") else line[1:]
            body = []
        elif name is not None:
            body.append(line)
    return sections


def _read_format(lines: list[str]) -> float:
    if not lines:
        raise ValueError("empty $MeshFormat section")
    try:
        return float(lines[0].split()[0])
    except (IndexError, ValueError) as error:
        raise ValueError(f"malformed $MeshFormat: {lines[0]!r}") from error


def _read_physical_names(lines: list[str]) -> dict[tuple[int, int], str]:
    """Map ``(dimension, tag) -> name`` from a ``$PhysicalNames`` section."""
    names: dict[tuple[int, int], str] = {}
    if not lines:
        return names
    count = int(lines[0])
    for line in lines[1 : count + 1]:
        parts = line.split(maxsplit=2)
        if len(parts) != 3:
            raise ValueError(f"malformed $PhysicalNames line: {line!r}")
        dim, tag = int(parts[0]), int(parts[1])
        names[(dim, tag)] = parts[2].strip().strip('"')
    return names


def _read_entities(lines: list[str]) -> dict[tuple[int, int], int | None]:
    """Map ``(dimension, entity tag) -> physical tag`` from 4.1 ``$Entities``."""
    physical: dict[tuple[int, int], int | None] = {}
    if not lines:
        return physical
    counts = [int(v) for v in lines[0].split()][:4]
    index = 1
    for dim, count in enumerate(counts):
        for _ in range(count):
            parts = lines[index].split()
            index += 1
            entity = int(parts[0])
            # A point entity has 3 coordinates, the others a bounding box.
            n_tags_index = 4 if dim == 0 else 7
            n_tags = int(parts[n_tags_index])
            tags = parts[n_tags_index + 1 : n_tags_index + 1 + n_tags]
            physical[(dim, entity)] = int(tags[0]) if n_tags else None
    return physical


def _read_nodes(lines: list[str], version: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(node tags, coordinates)``, the latter with 3 columns."""
    if not lines:
        raise ValueError("empty $Nodes section")

    if version < 4.0:
        count = int(lines[0])
        node_tags = np.empty(count, dtype=int)
        coords = np.empty((count, 3))
        for i, line in enumerate(lines[1 : count + 1]):
            parts = line.split()
            node_tags[i] = int(parts[0])
            coords[i] = [float(v) for v in parts[1:4]]
        return node_tags, coords

    header = lines[0].split()
    n_blocks = int(header[0])
    node_tags: list[int] = []
    coords: list[list[float]] = []
    index = 1
    for _ in range(n_blocks):
        _, _, parametric, n_block = (int(v) for v in lines[index].split())
        index += 1
        block_tags = [int(v) for v in lines[index : index + n_block]]
        index += n_block
        if parametric:
            raise NotImplementedError("parametric gmsh nodes are not supported")
        block_coords = [
            [float(v) for v in lines[index + k].split()[:3]] for k in range(n_block)
        ]
        index += n_block
        node_tags.extend(block_tags)
        coords.extend(block_coords)
    return np.asarray(node_tags, dtype=int), np.asarray(coords)


def _read_elements(
    lines: list[str], version: float, entities: dict[tuple[int, int], int | None]
) -> list[tuple[int, int, tuple[int, ...], int | None]]:
    """Return ``(dimension, element type, node tags, physical tag)`` records."""
    if not lines:
        raise ValueError("empty $Elements section")

    records: list[tuple[int, int, tuple[int, ...], int | None]] = []
    if version < 4.0:
        count = int(lines[0])
        for line in lines[1 : count + 1]:
            parts = [int(v) for v in line.split()]
            element_type = parts[1]
            if element_type not in _ELEMENT_NODES:
                raise ValueError(f"unsupported gmsh element type {element_type}")
            n_tags = parts[2]
            tags = parts[3 : 3 + n_tags]
            n_nodes = _ELEMENT_NODES[element_type]
            nodes = tuple(parts[3 + n_tags : 3 + n_tags + n_nodes])
            records.append(
                (_ELEMENT_DIM[element_type], element_type, nodes, tags[0] if tags else None)
            )
        return records

    n_blocks = int(lines[0].split()[0])
    index = 1
    for _ in range(n_blocks):
        dim, entity, element_type, n_block = (int(v) for v in lines[index].split())
        index += 1
        if element_type not in _ELEMENT_NODES:
            raise ValueError(f"unsupported gmsh element type {element_type}")
        n_nodes = _ELEMENT_NODES[element_type]
        physical = entities.get((dim, entity))
        for _ in range(n_block):
            parts = [int(v) for v in lines[index].split()]
            index += 1
            records.append((dim, element_type, tuple(parts[1 : 1 + n_nodes]), physical))
    return records


def read_msh(path: str | Path) -> Mesh:
    """Read an ASCII gmsh ``.msh`` file into a :class:`Mesh`.

    Cells are the highest-dimensional elements (triangles or tetrahedra).
    Physical groups on dimension ``d`` become regions; physical groups on
    dimension ``d - 1`` become entries of :attr:`Mesh.facet_tags`.  The 2D
    case assumes the mesh is planar (the ``z`` coordinate is dropped).
    """
    path = Path(path)
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    sections = _sections(lines)
    if "MeshFormat" not in sections:
        raise ValueError(f"{path} is not a gmsh ASCII file: no $MeshFormat section")
    version = _read_format(sections["MeshFormat"])
    if version not in (2.2, 4.1):
        raise NotImplementedError(
            f"gmsh {version} is not supported; write the file as ASCII 2.2 or 4.1"
        )

    names = _read_physical_names(sections.get("PhysicalNames", []))
    entities = _read_entities(sections.get("Entities", [])) if version >= 4.0 else {}
    node_tags, coords = _read_nodes(sections["Nodes"], version)
    records = _read_elements(sections["Elements"], version, entities)

    cell_dim = max(record[0] for record in records)
    if cell_dim == 3:
        cell_type = 4
    elif cell_dim == 2:
        cell_type = 2
    else:
        raise ValueError(f"no 2D or 3D cells found in {path}")
    cell_records = [r for r in records if r[0] == cell_dim and r[1] == cell_type]
    if not cell_records:
        raise ValueError(f"no {cell_dim}D cells found in {path}")

    index_of = {int(tag): i for i, tag in enumerate(node_tags)}
    cells = np.asarray(
        [[index_of[node] for node in record[2]] for record in cell_records], dtype=int
    )
    region_ids = np.asarray(
        [record[3] if record[3] is not None else 0 for record in cell_records], dtype=int
    )
    region_names = {
        int(tag): names.get((cell_dim, int(tag)), f"region{int(tag)}")
        for tag in np.unique(region_ids)
    }

    facet_dim = cell_dim - 1
    expected_type = 2 if facet_dim == 2 else 1
    facet_tags: dict[tuple[int, ...], str] = {}
    for dim, element_type, nodes, physical in records:
        if dim != facet_dim or element_type != expected_type or physical is None:
            continue
        key = tuple(sorted(index_of[node] for node in nodes))
        name = names.get((facet_dim, physical), f"boundary{physical}")
        if key in facet_tags and facet_tags[key] != name:
            raise ValueError(
                f"facet {key} is tagged both {facet_tags[key]!r} and {name!r}"
            )
        facet_tags[key] = name

    return Mesh(
        points=coords[:, :cell_dim],
        cells=cells,
        region_ids=region_ids,
        region_names=region_names,
        facet_tags=facet_tags,
    )
