from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod


import numpy as np
from darwin.material import Material, MU_0, EPSILON_0
from darwin.formulation import Formulation, coefficients, NodalEquation


@dataclass(frozen=True)
class LocalBlocks:
    K_AA: np.ndarray  # (n_edges, n_edges)
    K_Aphi: np.ndarray  # (n_edges, n_nodes)
    K_phiA: np.ndarray  # (n_nodes, n_edges)
    K_phiphi: np.ndarray  # (n_nodes, n_nodes)
    mass_edge: np.ndarray  # (n_edges, n_edges) integral(w_phys @ w_phys.T)
    rhs_edge: np.ndarray  # (n_edges,)
    rhs_node: np.ndarray  # (n_nodes,)


# |det J| / (product of column lengths) is dimensionless. A shape-regular
# simplex is of order 1. 1e-12 rejects columns that are linearly dependent
# at roundoff and does not depend on the mesh length unit.
_JACOBIAN_RELATIVE_TOLERANCE = 1e-12


def checked_jacobian(J: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(det J, |det J|)`` after rejecting a degenerate map.

    The columns of ``J`` are the tangent vectors ``dx/dξ``. Their lengths
    scale the determinant so the test is a shape measure, not a volume in
    metres or millimetres. On a straight order-1 element ``J`` is constant,
    so every quadrature point repeats the same test. On a curved element
    ``J`` depends on the reference coordinate and the test has to run at
    each point.
    """
    det_J = np.linalg.det(J)
    scale = np.prod(np.linalg.norm(J, axis=-2), axis=-1)
    abs_det_J = abs(det_J)
    if np.any(scale == 0.0) or np.any(
        abs_det_J <= _JACOBIAN_RELATIVE_TOLERANCE * scale
    ):
        raise ValueError("element Jacobian is singular")

    return det_J, abs_det_J


class Element(ABC):
    @abstractmethod
    def nodal_basis_functions(self, x_hat: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_nodal_basis_functions(self, x_hat: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def jacobian_matrix(self, x_hat: np.ndarray, x_nodes: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def whiteney_basis_functions(
        self, x_hat: np.ndarray, edges: list[tuple[int, int]]
    ) -> np.ndarray:
        pass

    @abstractmethod
    def curl_whiteney_basis_functions(
        self, x_hat: np.ndarray, edges: list[tuple[int, int]]
    ) -> np.ndarray:
        pass

    def darwin_local_forms(
        self,
        formulation: Formulation,
        x_nodes: np.ndarray,
        edges: list[tuple[int, int]],
        frequency: float,
        material: Material,
        gauss_points: GaussianQuadrature,
        source=None,  # current source when formulation is MQS, J_s (electric current density)
        charge=None,  # charge when formulation is Darwin, rho (electric charge density)
    ) -> LocalBlocks:
        """Darwin element matrices for one triangle or one tetrahedron.

        The basis, the Jacobian, and the reference curl come from the subclass.
        This method supplies only the Darwin coefficients: the edge mass is
        j*omega*sigma, and there is no -omega**2*epsilon term. A full-wave
        formulation is a second method on this class, using the same basis.

        Quadrature is evaluated in one batch over all Gauss points.
        """
        n_edges = len(edges)
        n_nodes = x_nodes.shape[0]
        n_dim = x_nodes.shape[1]

        omega = 2 * np.pi * frequency
        sigma = material.sigma
        nu = 1.0 / (material.mu_r * MU_0)
        coeff_A = 1j * omega * sigma

        coeffs = coefficients(formulation, material, frequency)
        if charge is not None and coeffs.nodal != NodalEquation.GAUSS:
            raise ValueError(
                "charge rho is only used by the Darwin Gauss row; "
                "use Formulation.DARWIN or pass charge=None"
            )

        weights = np.asarray(gauss_points.weights, dtype=float)
        points_ref = np.asarray(gauss_points.points, dtype=float)
        J = self.jacobian_matrix(points_ref, x_nodes)
        _, abs_det_J = checked_jacobian(J)
        inv_J = np.linalg.inv(J)

        lam = self.nodal_basis_functions(points_ref)  # (n_q, n_nodes, 1)
        w_phys = np.einsum(
            "qjk,qkl->qjl", self.whiteney_basis_functions(points_ref, edges), inv_J
        )
        curl_ref = self.curl_whiteney_basis_functions(points_ref, edges)
        grad_phys = np.einsum(
            "qjk,qkl->qjl", self.gradient_nodal_basis_functions(points_ref), inv_J
        )
        measure = weights * abs_det_J

        mass_edge = np.einsum("q,qid,qjd->ij", measure, w_phys, w_phys)
        K_AA = coeff_A * mass_edge + nu * _physical_curl_gram(
            curl_ref, J, abs_det_J, weights
        )
        K_Aphi = np.einsum("q,qid,qjd->ij", measure * coeffs.a_phi, w_phys, grad_phys)
        K_phiA = np.einsum("q,qid,qjd->ij", measure * coeffs.phi_a, grad_phys, w_phys)
        K_phiphi = np.einsum(
            "q,qid,qjd->ij", measure * coeffs.phi_phi, grad_phys, grad_phys
        )

        rhs_edge = np.zeros(n_edges, dtype=complex)
        rhs_node = np.zeros(n_nodes, dtype=complex)
        x_phys = np.einsum("qnl,nd->qd", lam, x_nodes)

        if source is not None:
            current = _eval_source_on_points(source, x_phys, n_dim)
            rhs_edge += np.einsum("q,qed,qd->e", measure, w_phys, current)
            if coeffs.nodal == NodalEquation.CONTINUITY:
                rhs_node += np.einsum("q,qnd,qd->n", measure, grad_phys, current)

        if charge is not None and coeffs.nodal == NodalEquation.GAUSS:
            rho = _eval_charge_on_points(charge, x_phys)
            rhs_node += np.einsum("q,qn,q->n", measure, lam[..., 0], rho)

        return LocalBlocks(
            K_AA, K_Aphi, K_phiA, K_phiphi, mass_edge, rhs_edge, rhs_node
        )


def _eval_source_on_points(source, x_phys: np.ndarray, n_dim: int) -> np.ndarray:
    """Evaluate a pointwise or batched current density on quadrature points.

    Callables may take one point ``(n_dim,)`` or a batch ``(n_q, n_dim)``.
    A failed or mis-shaped batch call falls back to one evaluation per point.
    """
    n_q = x_phys.shape[0]
    try:
        raw = np.asarray(source(x_phys), dtype=complex)
        if raw.shape == (n_q, n_dim):
            return raw
        if raw.shape == (n_dim,) and n_q == 1:
            return raw.reshape(1, n_dim)
    except (TypeError, ValueError, IndexError):
        pass

    current = np.empty((n_q, n_dim), dtype=complex)
    for q in range(n_q):
        value = np.asarray(source(x_phys[q]), dtype=complex).reshape(-1)
        if value.shape != (n_dim,):
            raise ValueError(
                f"source must return a length-{n_dim} vector, got shape {value.shape}"
            )
        current[q] = value
    return current


def _eval_charge_on_points(charge, x_phys: np.ndarray) -> np.ndarray:
    """Evaluate a pointwise or batched charge density on quadrature points.

    A size-1 return (constant rho) is broadcast to every quadrature point.
    A failed or mis-shaped batch call falls back to one evaluation per point.
    """
    n_q = x_phys.shape[0]
    try:
        raw = np.asarray(charge(x_phys), dtype=complex).reshape(-1)
        if raw.shape == (n_q,):
            return raw
        if raw.size == 1:
            return np.full(n_q, complex(raw.reshape(-1)[0]), dtype=complex)
    except (TypeError, ValueError, IndexError):
        pass

    rho = np.empty(n_q, dtype=complex)
    for q in range(n_q):
        value = np.asarray(charge(x_phys[q]), dtype=complex).reshape(-1)
        if value.size != 1:
            raise ValueError(f"charge must return a scalar rho, got shape {value.shape}")
        rho[q] = value[0]
    return rho


def _physical_curl_gram(
    curl_ref: np.ndarray, J: np.ndarray, abs_det_J: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Assemble Σ_q w_q (curl_x · curl_x) |det J| over the quadrature batch.

    A 2D curl is the scalar curl_ξ / det J. A 3D curl is J curl_ξ / det J.
    Both leave a factor 1/|det J| in front of a reference inner product.
    In 3D that inner product is curl_ξ · (J^T J curl_ξ), not curl_ξ · curl_ξ.

    Args:
        curl_ref: ``(n_q, n_edges, 1)`` in 2D or ``(n_q, n_edges, 3)`` in 3D
        J: ``(n_q, dim, dim)``
        abs_det_J: ``(n_q,)``
        weights: ``(n_q,)``
    Returns:
        Physical curl gram matrix of shape ``(n_edges, n_edges)``.
    """
    if curl_ref.shape[-1] == 1:
        return np.einsum(
            "q,qi,qj->ij",
            weights / abs_det_J,
            curl_ref[:, :, 0],
            curl_ref[:, :, 0],
        )
    return np.einsum(
        "q,qia,qka,qkb,qjb->ij",
        weights / abs_det_J,
        curl_ref,
        J,
        J,
        curl_ref,
    )


def _as_quadrature_batch(x_hat: np.ndarray, dim: int) -> tuple[np.ndarray, bool]:
    """Normalize reference coordinates to ``(n_q, dim)``.

    Returns the batch and whether the caller passed a single point ``(dim,)``.
    """
    x_hat = np.asarray(x_hat, dtype=float)
    single = x_hat.ndim == 1
    if single:
        x_hat = x_hat[None, :]
    if x_hat.ndim != 2 or x_hat.shape[-1] != dim:
        raise ValueError(f"x_hat must have shape (n_q, {dim}) or ({dim},), got {x_hat.shape}")
    return x_hat, single


def _maybe_squeeze_qp(values: np.ndarray, single: bool) -> np.ndarray:
    return values[0] if single else values


class TetrahedronWhitney(Element):
    """Tetrahedron Whitney element of order 1"""

    def __init__(self, order: int = 1):
        assert order == 1, "Only order 1 is supported for now"
        self.order = 1

    def nodal_basis_functions(self, x_hat: np.ndarray) -> np.ndarray:
        """Nodal basis values of shape ``(n_q, 4, 1)`` or ``(4, 1)`` for one point."""
        x_hat, single = _as_quadrature_batch(x_hat, 3)
        xi, eta, zeta = x_hat[:, 0], x_hat[:, 1], x_hat[:, 2]
        lam = np.stack([1.0 - xi - eta - zeta, xi, eta, zeta], axis=-1)[..., None]
        return _maybe_squeeze_qp(lam, single)

    def gradient_nodal_basis_functions(self, x_hat: np.ndarray) -> np.ndarray:
        """Constant gradients of shape ``(n_q, 4, 3)`` or ``(4, 3)`` for one point."""
        x_hat, single = _as_quadrature_batch(x_hat, 3)
        grad = np.array(
            [[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        out = np.tile(grad, (x_hat.shape[0], 1, 1))
        return _maybe_squeeze_qp(out, single)

    def jacobian_matrix(self, x_hat: np.ndarray, x_nodes: np.ndarray) -> np.ndarray:
        """Jacobian of shape ``(n_q, 3, 3)`` or ``(3, 3)`` for one point."""
        x_hat, single = _as_quadrature_batch(x_hat, 3)
        grad_lambda = self.gradient_nodal_basis_functions(x_hat)  # (n_q, 4, 3)
        J = np.einsum("kj,qkl->qjl", x_nodes, grad_lambda)  # (n_q, 3, 3)
        return _maybe_squeeze_qp(J, single)

    def whiteney_basis_functions(
        self, x_hat: np.ndarray, edges: list[tuple[int, int]]
    ) -> np.ndarray:
        """Whitney edge basis of shape ``(n_q, n_edges, 3)`` or ``(n_edges, 3)``."""
        x_hat, single = _as_quadrature_batch(x_hat, 3)
        n_edges = len(edges)
        n_q = x_hat.shape[0]
        lambdas = self.nodal_basis_functions(x_hat)  # (n_q, 4, 1)
        grad_lambdas = self.gradient_nodal_basis_functions(x_hat)  # (n_q, 4, 3)
        w = np.empty((n_q, n_edges, 3))
        for e, (i, j) in enumerate(edges):
            w[:, e, :] = (
                lambdas[:, i, :] * grad_lambdas[:, j, :]
                - lambdas[:, j, :] * grad_lambdas[:, i, :]
            )
        return _maybe_squeeze_qp(w, single)

    def curl_whiteney_basis_functions(
        self, x_hat: np.ndarray, edges: list[tuple[int, int]]
    ) -> np.ndarray:
        """Reference curl of shape ``(n_q, n_edges, 3)`` or ``(n_edges, 3)``."""
        x_hat, single = _as_quadrature_batch(x_hat, 3)
        n_edges = len(edges)
        n_q = x_hat.shape[0]
        grad = self.gradient_nodal_basis_functions(x_hat)  # (n_q, 4, 3)
        curl = np.empty((n_q, n_edges, 3))
        for e, (i, j) in enumerate(edges):
            curl[:, e, :] = 2.0 * np.cross(grad[:, i, :], grad[:, j, :])
        return _maybe_squeeze_qp(curl, single)


class TriangleWhitney(Element):
    """Triangle Whitney element of order 1"""

    def __init__(self, order: int = 1):
        assert order == 1, "Only order 1 is supported for now"
        self.order = 1

    def nodal_basis_functions(self, x_hat: np.ndarray) -> np.ndarray:
        """Nodal basis values of shape ``(n_q, 3, 1)`` or ``(3, 1)`` for one point."""
        x_hat, single = _as_quadrature_batch(x_hat, 2)
        xi, eta = x_hat[:, 0], x_hat[:, 1]
        lam = np.stack([1.0 - xi - eta, xi, eta], axis=-1)[..., None]
        return _maybe_squeeze_qp(lam, single)

    def gradient_nodal_basis_functions(self, x_hat: np.ndarray) -> np.ndarray:
        """Constant gradients of shape ``(n_q, 3, 2)`` or ``(3, 2)`` for one point."""
        x_hat, single = _as_quadrature_batch(x_hat, 2)
        grad = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
        out = np.tile(grad, (x_hat.shape[0], 1, 1))
        return _maybe_squeeze_qp(out, single)

    def jacobian_matrix(self, x_hat: np.ndarray, x_nodes: np.ndarray) -> np.ndarray:
        """Jacobian of shape ``(n_q, 2, 2)`` or ``(2, 2)`` for one point."""
        x_hat, single = _as_quadrature_batch(x_hat, 2)
        grad_lambda = self.gradient_nodal_basis_functions(x_hat)  # (n_q, 3, 2)
        J = np.einsum("kj,qkl->qjl", x_nodes, grad_lambda)  # (n_q, 2, 2)
        return _maybe_squeeze_qp(J, single)

    def whiteney_basis_functions(
        self, x_hat: np.ndarray, edges: list[tuple[int, int]]
    ) -> np.ndarray:
        """Whitney edge basis of shape ``(n_q, n_edges, 2)`` or ``(n_edges, 2)``."""
        x_hat, single = _as_quadrature_batch(x_hat, 2)
        n_edges = len(edges)
        n_q = x_hat.shape[0]
        lambdas = self.nodal_basis_functions(x_hat)  # (n_q, 3, 1)
        grad_lambdas = self.gradient_nodal_basis_functions(x_hat)  # (n_q, 3, 2)
        w = np.empty((n_q, n_edges, 2))
        for e, (i, j) in enumerate(edges):
            w[:, e, :] = (
                lambdas[:, i, :] * grad_lambdas[:, j, :]
                - lambdas[:, j, :] * grad_lambdas[:, i, :]
            )
        return _maybe_squeeze_qp(w, single)

    def curl_whiteney_basis_functions(
        self, x_hat: np.ndarray, edges: list[tuple[int, int]]
    ) -> np.ndarray:
        """Reference curl of shape ``(n_q, n_edges, 1)`` or ``(n_edges, 1)``.

        On a plane the curl is a scalar along the normal.
        """
        x_hat, single = _as_quadrature_batch(x_hat, 2)
        n_edges = len(edges)
        n_q = x_hat.shape[0]
        grad = self.gradient_nodal_basis_functions(x_hat)  # (n_q, 3, 2)
        curl = np.empty((n_q, n_edges, 1))
        for e, (i, j) in enumerate(edges):
            gi = grad[:, i, :]
            gj = grad[:, j, :]
            curl[:, e, 0] = 2.0 * (gi[:, 0] * gj[:, 1] - gi[:, 1] * gj[:, 0])
        return _maybe_squeeze_qp(curl, single)


def _duffy_triangle_rule(degree: int):
    """Gauss rule on the reference triangle through the Duffy map.

    x = u, y = (1 - u) v turns the triangle into the unit square with
    jacobian (1 - u), so n Gauss points per direction integrate every
    polynomial of total degree <= 2n - 2 exactly.
    """
    n = (degree + 3) // 2
    nodes, weights = np.polynomial.legendre.leggauss(n)
    u = 0.5 * (nodes + 1.0)
    wu = 0.5 * weights
    points = np.empty((n * n, 2))
    quad_weights = np.empty(n * n)
    for i in range(n):
        for j in range(n):
            k = i * n + j
            points[k] = (u[i], (1.0 - u[i]) * u[j])
            quad_weights[k] = wu[i] * wu[j] * (1.0 - u[i])
    return points, quad_weights


def _duffy_tetrahedron_rule(degree: int):
    """Gauss rule on the reference tetrahedron through the Duffy map.

    (xi, eta, zeta) = (u, (1 - u) v, (1 - u) (1 - v) w) maps the unit cube
    onto the tetrahedron with jacobian (1 - u)^2 (1 - v). n Gauss points per
    direction integrate every polynomial of total degree <= 2 n - 4 exactly:
    the mapped integrand has degree ``degree + 3`` and each direction is
    exact up to degree ``2 n - 1``.
    """
    n = (degree + 5) // 2
    nodes, weights = np.polynomial.legendre.leggauss(n)
    u = 0.5 * (nodes + 1.0)
    wu = 0.5 * weights
    points = np.empty((n**3, 3))
    quad_weights = np.empty(n**3)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                m = (i * n + j) * n + k
                points[m] = (
                    u[i],
                    (1.0 - u[i]) * u[j],
                    (1.0 - u[i]) * (1.0 - u[j]) * u[k],
                )
                quad_weights[m] = (
                    wu[i] * wu[j] * wu[k] * (1.0 - u[i]) ** 2 * (1.0 - u[j])
                )
    return points, quad_weights


class GaussianQuadrature:
    """Reference-cell rules. degree is the exact polynomial degree."""

    a = 0.5854101966249685
    b = (1 - a) / 3
    _RULES = {
        # (dim, degree): (points of shape (n_q, dim), weights of shape (n_q,))
        (2, 1): (
            np.array([[1.0 / 3.0, 1.0 / 3.0]]),
            np.array([1.0 / 2.0]),
        ),
        (2, 2): (
            np.array(
                [
                    [1.0 / 6.0, 1.0 / 6.0],
                    [2.0 / 3.0, 1.0 / 6.0],
                    [1.0 / 6.0, 2.0 / 3.0],
                ]
            ),
            np.full(3, 1.0 / 6.0),
        ),
        (3, 1): (
            np.array([[0.25, 0.25, 0.25]]),
            np.array([1.0 / 6.0]),
        ),
        (3, 2): (
            np.array(
                [
                    [b, b, b],
                    [a, b, b],
                    [b, a, b],
                    [b, b, a],
                ]
            ),
            np.full(4, 1.0 / 24.0),
        ),
    }

    def __init__(self, dim: int, degree: int):
        rule = self._RULES.get((dim, degree))
        if rule is None and dim == 2:
            rule = _duffy_triangle_rule(degree)
        if rule is None and dim == 3:
            rule = _duffy_tetrahedron_rule(degree)
        if rule is None:
            raise NotImplementedError(f"no rule for dim={dim}, degree={degree}")
        self.dim = dim
        self.degree = degree
        self.points, self.weights = rule

    def __iter__(self):

        for i_gp, (weight, point) in enumerate(zip(self.weights, self.points)):
            yield i_gp, float(weight), point

    def __len__(self):
        return len(self.weights)
