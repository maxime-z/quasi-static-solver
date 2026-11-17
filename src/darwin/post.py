"""Sample the solved Darwin fields on each cell.

The solve returns one complex coefficient per edge and per node:

    A(x)   = sum_e a_e w_e(x),        B = curl A,
    phi(x) = sum_n phi_n lambda_n(x), E = -j*omega*A - grad(phi).

Both quantities are evaluated at the cell centroid, which is where the
lowest-order element represents them exactly: E is linear and B is constant
per cell in 2D. Comparing those samples with an exact solution therefore
measures the discretization error and not a postprocessing error.

Mesh export, interpolation between meshes and impedance extraction are not
here on purpose; the convergence tests only need the cell integrals below.
"""

from dataclasses import dataclass

import numpy as np

from darwin.elements import Element
from darwin.mesh_io import Mesh
from darwin.topology import Topology


@dataclass(frozen=True)
class CellSamples:
    """A field sampled at the centroid of every cell."""

    point: np.ndarray  # (n_cell, n_q, dim)
    measure: np.ndarray  # (n_cell, n_q)
    value: np.ndarray  # (n_cell, n_q, n_component), complex


def cell_fields(
    mesh: Mesh,
    topology: Topology,
    element: Element,
    x: np.ndarray,
    frequency: float,
    x_hat: np.ndarray,
    weights: np.ndarray,
) -> tuple[CellSamples, CellSamples]:
    """Return ``(E, B)`` at the quadrature points of every cell.

    Args:
        mesh: The mesh to sample the fields on.
        topology: The topology of the mesh.
        element: The element to use for the sampling.
        x: The full unknown vector ``[a_edge | phi_node]`` in the order used
            by :func:`darwin.assemble.assemble`.
        frequency: The frequency of the fields.
        x_hat: The local coordinates of the quadrature points ``(n_q, dim)``.
        weights: The weights of the quadrature points ``(n_q,)``.

    Returns:
        A tuple of :class:`CellSamples` objects for the electric and magnetic fields.
    """
    n_edges = len(topology.edges)
    a = np.asarray(x[:n_edges])
    phi = np.asarray(x[n_edges:])
    omega = 2 * np.pi * frequency
    dim = mesh.points.shape[1]
    n_cells = len(mesh.cells)
    n_q = x_hat.shape[0]
    x_hat = np.asarray(x_hat, dtype=float)
    weights = np.asarray(weights, dtype=float)

    points = np.empty((n_cells, n_q, dim))
    measure = np.empty((n_cells, n_q))
    e_value = np.empty((n_cells, n_q, dim), dtype=complex)
    b_value = np.empty((n_cells, n_q, 1 if dim == 2 else 3), dtype=complex)

    for i, cell in enumerate(mesh.cells):
        nodes = mesh.points[cell]
        local_of = {int(g): k for k, g in enumerate(cell)}
        edge_ids = topology.cell_edges[i]
        pairs = [
            (local_of[int(n1)], local_of[int(n2)])
            for n1, n2 in topology.edges[edge_ids]
        ]

        J = element.jacobian_matrix(x_hat, nodes)  # (n_q, dim, dim)
        det_j = np.linalg.det(J)  # (n_q,)
        inv_J = np.linalg.inv(J)  # (n_q, dim, dim)
        lam = element.nodal_basis_functions(x_hat)[..., 0]  # (n_q, n_nodes)
        points[i] = lam @ nodes  # (n_q, dim)

        w = np.einsum(
            "qed,qdf->qef",
            element.whiteney_basis_functions(x_hat, pairs),
            inv_J,
        )
        grad = np.einsum(
            "qnd,qdf->qnf",
            element.gradient_nodal_basis_functions(x_hat),
            inv_J,
        )
        curl = element.curl_whiteney_basis_functions(x_hat, pairs)

        a_cell = a[edge_ids]
        phi_cell = phi[cell]
        e_value[i] = -1j * omega * np.einsum("e,qed->qd", a_cell, w) - np.einsum(
            "n,qnd->qd", phi_cell, grad
        )

        if dim == 2:
            b_value[i, :, 0] = np.einsum("e,qe->q", a_cell, curl[..., 0]) / det_j
        else:
            curl_a = np.einsum("e,qed->qd", a_cell, curl)
            b_value[i] = np.einsum("qdi,qi->qd", J, curl_a) / det_j[:, None]

        measure[i] = weights * np.abs(det_j)

    return (
        CellSamples(points, measure, e_value),
        CellSamples(points, measure, b_value),
    )


def relative_l2(samples: CellSamples, exact: np.ndarray) -> float:
    """``||samples - exact||_L2 / ||exact||_L2`` with the cell areas as weights."""
    exact = np.asarray(exact, dtype=complex)
    if exact.shape != samples.value.shape:
        raise ValueError(f"shapes differ: {exact.shape} and {samples.value.shape}")
    difference = samples.value - exact
    numerator = np.sum(samples.measure[..., None] * np.abs(difference) ** 2)
    denominator = np.sum(samples.measure[..., None] * np.abs(exact) ** 2)
    return float(np.sqrt(numerator / denominator))
