"""The manufactured Darwin case used by milestone M5.

Geometry: the square ``[-1, 1]^2``, every wall grounded and PEC.
Material: ``sigma = 0.01 S/m``, ``eps_r = 80``, ``mu_r = 1``, ``f = 2 MHz``.

Exact fields::

    phi = cos(pi x / 2) cos(pi y / 2)
    chi = chi0 (cos(pi x / 2) cos(pi y / 2))^2
    psi = alpha phi,   alpha = -(sigma + j omega eps) / (j omega sigma)
    A   = grad(psi) + curl(chi e_z)
    E   = -j omega A - grad(phi),   B = curl A = -laplacian(chi) e_z

with the source

    J_s = curl nu curl A + j omega sigma A + (sigma + j omega eps) grad phi.

Why this is a Darwin problem and not just a random smooth pair: ``div J_s = 0``
because ``div A = laplacian(psi) = alpha laplacian(phi)`` and
``j omega sigma alpha + (sigma + j omega eps) = 0``. The tangential component of
``A`` vanishes on every wall: ``grad(psi)`` is normal there and ``chi`` has a
vanishing normal derivative, so ``curl(chi e_z)`` has no normal component.
``phi`` vanishes on every wall by construction. The exact solution therefore
satisfies the boundary conditions of the solver, and both blocks and their
coupling carry a nonzero field.
"""

import numpy as np

from darwin.assemble import assemble
from darwin.bc import DirichletWall, apply_dirichlet
from darwin.elements import TriangleWhitney
from darwin.formulation import Formulation
from darwin.material import EPSILON_0, MU_0, Material
from darwin.mesh_io import Mesh
from darwin.post import cell_fields, relative_l2
from darwin.solve import solve_direct
from darwin.topology import build_topology, tag_boundary

FREQUENCY = 2.0e6
SIGMA = 0.01
EPS_R = 80.0
MU_R = 1.0
PHI0 = 1.0
CHI0 = 1.0e-3

_OMEGA = 2.0 * np.pi * FREQUENCY
_EPS = EPS_R * EPSILON_0
_NU = 1.0 / (MU_R * MU_0)
_COEFF_PHI = SIGMA + 1j * _OMEGA * _EPS
_ALPHA = -_COEFF_PHI / (1j * _OMEGA * SIGMA)
_HALF_PI = 0.5 * np.pi
_PI_CUBED = np.pi**3


def _cos_half(value):
    return np.cos(_HALF_PI * value)


def _sin_half(value):
    return np.sin(_HALF_PI * value)


def exact_phi(points):
    points = np.atleast_2d(points)
    return PHI0 * _cos_half(points[:, 0]) * _cos_half(points[:, 1])


def exact_grad_phi(points):
    points = np.atleast_2d(points)
    x, y = points[:, 0], points[:, 1]
    return np.stack(
        [
            -PHI0 * _HALF_PI * _sin_half(x) * _cos_half(y),
            -PHI0 * _HALF_PI * _cos_half(x) * _sin_half(y),
        ],
        axis=1,
    )


def exact_chi(points):
    points = np.atleast_2d(points)
    return CHI0 * (_cos_half(points[:, 0]) * _cos_half(points[:, 1])) ** 2


def _grad_chi(points):
    x, y = points[:, 0], points[:, 1]
    return np.stack(
        [
            -CHI0 * _HALF_PI * np.sin(np.pi * x) * _cos_half(y) ** 2,
            -CHI0 * _HALF_PI * np.sin(np.pi * y) * _cos_half(x) ** 2,
        ],
        axis=1,
    )


def _laplacian_chi(points):
    x, y = points[:, 0], points[:, 1]
    return -(_HALF_PI**2) * CHI0 * (
        np.cos(np.pi * x) * _cos_half(y) ** 2 + np.cos(np.pi * y) * _cos_half(x) ** 2
    )


def exact_a(points):
    points = np.atleast_2d(points)
    grad_phi = exact_grad_phi(points)
    grad_chi = _grad_chi(points)
    return np.stack(
        [
            _ALPHA * grad_phi[:, 0] + grad_chi[:, 1],
            _ALPHA * grad_phi[:, 1] - grad_chi[:, 0],
        ],
        axis=1,
    )


def exact_e(points):
    return -1j * _OMEGA * exact_a(points) - exact_grad_phi(points)


def exact_b(points):
    """``Bz = -laplacian(chi)``, shape ``(n_points, 1)``."""
    return (-_laplacian_chi(np.atleast_2d(points)))[:, None]


def _source_array(points):
    points = np.atleast_2d(points)
    x, y = points[:, 0], points[:, 1]
    d_lap_dx = (
        _PI_CUBED / 4.0 * CHI0 * np.sin(np.pi * x) * (1.0 + 2.0 * np.cos(np.pi * y))
    )
    d_lap_dy = (
        _PI_CUBED / 4.0 * CHI0 * np.sin(np.pi * y) * (1.0 + 2.0 * np.cos(np.pi * x))
    )
    curl_curl = np.stack([-d_lap_dy, d_lap_dx], axis=1)
    return (
        _NU * curl_curl
        + 1j * _OMEGA * SIGMA * exact_a(points)
        + _COEFF_PHI * exact_grad_phi(points)
    )


def source(point) -> np.ndarray:
    """Current density ``J_s`` at one physical point, shape ``(dim,)``."""
    return _source_array(np.asarray(point, dtype=float))[0]


def material() -> Material:
    return Material(name="lossy", sigma=SIGMA, mu_r=MU_R, eps_r=EPS_R)


def square_mesh(n: int) -> Mesh:
    """``n x n`` square cells on ``[-1, 1]^2``, counter-clockwise triangles."""
    coordinates = np.linspace(-1.0, 1.0, n + 1)
    points = np.array(
        [
            [coordinates[i], coordinates[j]]
            for j in range(n + 1)
            for i in range(n + 1)
        ]
    )
    cells = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            b = a + 1
            c = a + n + 2
            d = a + n + 1
            cells.append([a, b, c])
            cells.append([a, c, d])
    return Mesh(
        points=points,
        cells=np.array(cells, dtype=int),
        region_ids=np.zeros(len(cells), dtype=int),
        region_names={0: "lossy"},
    )


def solve_case(n: int, quadrature_degree: int = 6):
    """Assemble, constrain and solve the manufactured problem on the n x n mesh."""
    mesh = square_mesh(n)
    tagged, names = tag_boundary(mesh, {"wall": lambda point: True})
    topology = build_topology(mesh, boundary_tags=(tagged, names))
    element = TriangleWhitney()
    system = assemble(
        Formulation.DARWIN,
        mesh,
        topology,
        {0: material()},
        FREQUENCY,
        source,
        element,
        quadrature_degree=quadrature_degree,
    )
    system = apply_dirichlet(system, topology, [DirichletWall(tag="wall", voltage=0.0)])
    return mesh, topology, solve_direct(system)


def errors(n: int) -> tuple[float, float]:
    """Relative L2 errors ``(E, B)`` of the solution on the n x n mesh."""
    mesh, topology, x = solve_case(n)
    e_samples, b_samples = cell_fields(mesh, topology, TriangleWhitney(), x, FREQUENCY)
    return (
        relative_l2(e_samples, exact_e(e_samples.centroid)),
        relative_l2(b_samples, exact_b(b_samples.centroid)),
    )
