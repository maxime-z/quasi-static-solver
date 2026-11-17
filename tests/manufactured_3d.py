"""Manufactured solution for the three dimensional Darwin formulation.

The exact fields are polynomials, so the element can represent them (and the
right hand sides) up to quadrature error. Four refinements drive the relative
L2 error of E and B down; the observed order is the one promised in the plan.

The exact solution is

    A = ((1-y^2)(1-z^2), (1-x^2)(1-z^2), (1-x^2)(1-y^2))
    phi = (1-x^2)(1-y^2)(1-z^2)

so every tangential component of A vanishes on the six walls and phi = 0 on
them. ``div A = 0``, and the sources follow from

    J_s = curl nu curl A + j omega sigma A + (sigma + j omega eps) grad phi
    rho = -eps laplacian phi
"""

import sys
import time

import numpy as np

from darwin.elements import GaussianQuadrature, TetrahedronWhitney
from darwin.formulation import Formulation
from darwin.assemble import assemble
from darwin.bc import DirichletWall, apply_dirichlet
from darwin.material import EPSILON_0, MU_0, Material
from darwin.mesh_io import Mesh
from darwin.post import cell_fields, relative_l2
from darwin.solve import solve_direct
from darwin.topology import build_topology

FREQUENCY = 2.0e6
SIGMA = 0.01
EPS_R = 80.0
MU_R = 1.0
PHI_0 = 1.0
CHI_0 = 1.0e-3

OMEGA = 2 * np.pi * FREQUENCY
EPS = EPS_R * EPSILON_0
NU = 1.0 / (MU_R * MU_0)
ALPHA = -SIGMA / (1j * OMEGA * (SIGMA + 1j * OMEGA * EPS))

# One cube split into six tetrahedra, in the local corner numbering below.
_CUBE_CELLS = np.array(
    [
        [0, 1, 3, 4],
        [4, 5, 7, 1],
        [1, 3, 4, 7],
        [0, 2, 3, 4],
        [4, 7, 6, 2],
        [2, 4, 3, 7],
    ],
    dtype=np.int64,
)


def _vector_potential(points: np.ndarray) -> np.ndarray:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    solenoidal = CHI_0 * np.stack(
        (
            (1 - y**2) * (1 - z**2),
            (1 - x**2) * (1 - z**2),
            (1 - x**2) * (1 - y**2),
        ),
        axis=1,
    )
    return solenoidal + ALPHA * _gradient_potential(points)


def _magnetic_field(points: np.ndarray) -> np.ndarray:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return (
        2
        * CHI_0
        * np.stack(
            ((1 - x**2) * (z - y), (1 - y**2) * (x - z), (1 - z**2) * (y - x)),
            axis=1,
        )
    )


def _scalar_potential(points: np.ndarray) -> np.ndarray:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return PHI_0 * (1 - x**2) * (1 - y**2) * (1 - z**2)


def _gradient_potential(points: np.ndarray) -> np.ndarray:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return (
        -2
        * PHI_0
        * np.stack(
            (
                x * (1 - y**2) * (1 - z**2),
                (1 - x**2) * y * (1 - z**2),
                (1 - x**2) * (1 - y**2) * z,
            ),
            axis=1,
        )
    )


def _electric_field(points: np.ndarray) -> np.ndarray:
    return -1j * OMEGA * _vector_potential(points) - _gradient_potential(points)


def source_current(point: np.ndarray) -> np.ndarray:
    return _source_current(np.asarray(point)[None, :])[0]


def source_charge(point: np.ndarray) -> complex:
    return complex(_source_charge(np.asarray(point)[None, :])[0])


def _reshape(points: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    shape = points.shape[:-1]
    return np.asarray(points).reshape(-1, 3), shape


def exact_e(points: np.ndarray) -> np.ndarray:
    flat, shape = _reshape(points)
    return _electric_field(flat).reshape(shape + (3,))


def exact_b(points: np.ndarray) -> np.ndarray:
    flat, shape = _reshape(points)
    return _magnetic_field(flat).reshape(shape + (3,))


def _source_current(points: np.ndarray) -> np.ndarray:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    curl_b = (
        2
        * CHI_0
        * np.stack(
            (
                2 - y**2 - z**2,
                2 - z**2 - x**2,
                2 - x**2 - y**2,
            ),
            axis=1,
        )
    )
    return (
        NU * curl_b
        + 1j * OMEGA * SIGMA * _vector_potential(points)
        + (SIGMA + 1j * OMEGA * EPS) * _gradient_potential(points)
    )


def _source_charge(points: np.ndarray) -> np.ndarray:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    laplacian = (
        -2
        * PHI_0
        * ((1 - y**2) * (1 - z**2) + (1 - x**2) * (1 - z**2) + (1 - x**2) * (1 - y**2))
    )
    return -1j * OMEGA * EPS**2 * laplacian / (SIGMA + 1j * OMEGA * EPS)


def cube_mesh(n: int) -> Mesh:
    """An ``n x n x n`` cube of cells, each split into six tetrahedra.

    The split is the Kuhn (Freudenthal) triangulation: the six tetrahedra of
    a cell are the paths from the cell corner to its opposite corner, one per
    ordering of the three axes.  That is the split that stays conforming
    across cells - any fixed six-tuple of corner indices does not.
    """

    grid = np.linspace(-1.0, 1.0, n + 1)
    points = np.array([[x, y, z] for z in grid for y in grid for x in grid])

    def node(i: int, j: int, k: int) -> int:
        return (k * (n + 1) + j) * (n + 1) + i

    offset = {0: 1, 1: n + 1, 2: (n + 1) ** 2}
    cells = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                base = node(i, j, k)
                for first, second, third in (
                    (0, 1, 2),
                    (0, 2, 1),
                    (1, 0, 2),
                    (1, 2, 0),
                    (2, 0, 1),
                    (2, 1, 0),
                ):
                    corners = [
                        base,
                        base + offset[third],
                        base + offset[third] + offset[second],
                        base + offset[0] + offset[1] + offset[2],
                    ]
                    corners = np.array(corners)
                    q = points[corners]
                    det = np.linalg.det(
                        np.stack([q[1] - q[0], q[2] - q[0], q[3] - q[0]])
                    )
                    if det < 0.0:
                        corners[2], corners[3] = corners[3], corners[2]
                    cells.append(corners)
    cells = np.array(cells, dtype=int)

    tags = {
        "xmin": lambda p: np.isclose(p[..., 0], -1.0),
        "xmax": lambda p: np.isclose(p[..., 0], 1.0),
        "ymin": lambda p: np.isclose(p[..., 1], -1.0),
        "ymax": lambda p: np.isclose(p[..., 1], 1.0),
        "zmin": lambda p: np.isclose(p[..., 2], -1.0),
        "zmax": lambda p: np.isclose(p[..., 2], 1.0),
    }
    from darwin.topology import tag_boundary

    facets, names = tag_boundary(
        Mesh(points, cells, np.zeros(len(cells), dtype=int)), tags
    )
    facet_tags = {facet: name for facet, name in zip(facets, names)}
    return Mesh(points, cells, np.zeros(len(cells), dtype=int), facet_tags=facet_tags)


def solve_case(n: int, quadrature_degree: int = 6):
    """Assemble, constrain and solve the manufactured problem on the n^3 cube."""
    mesh = cube_mesh(n)
    topology = build_topology(mesh)
    material = Material("body", sigma=SIGMA, mu_r=MU_R, eps_r=EPS_R)

    time_start = time.time()
    system = assemble(
        Formulation.DARWIN,
        mesh,
        topology,
        {0: material},
        FREQUENCY,
        _source_current,
        TetrahedronWhitney(),
        charge=_source_charge,
        quadrature_degree=quadrature_degree,
    )
    walls = [DirichletWall(tag, 0.0) for tag in sorted(set(mesh.facet_tags.values()))]
    system = apply_dirichlet(system, topology, walls)
    time_end_assembly = time.time()
    print(
        f"Assembly and Dirichlet constraints time taken: {time_end_assembly - time_start} seconds"
    )
    x = solve_direct(system)
    time_end_solve = time.time()
    print(f"Solve time taken: {time_end_solve - time_end_assembly} seconds")

    return mesh, topology, x


def errors(n: int) -> tuple[float, float]:
    """Relative L2 errors ``(E, B)`` on the n^3 cube of cells."""
    mesh, topology, x = solve_case(n)
    rule = GaussianQuadrature(3, 4)
    e_samples, b_samples = cell_fields(
        mesh, topology, TetrahedronWhitney(), x, FREQUENCY, rule.points, rule.weights
    )
    return (
        relative_l2(e_samples, exact_e(e_samples.point)),
        relative_l2(b_samples, exact_b(b_samples.point)),
    )


if __name__ == "__main__":
    previous = None
    for n in (2, 4, 8, 16):
        current = errors(n)
        order = ""
        if previous is not None:
            order = (
                f"  order E {np.log2(previous[0] / current[0]):.2f}"
                f"  B {np.log2(previous[1] / current[1]):.2f}"
            )
        print(f"n={n:2d}  E={current[0]:.4e}  B={current[1]:.4e}{order}")
        previous = current
