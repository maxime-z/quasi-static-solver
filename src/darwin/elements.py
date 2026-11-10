from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod


import numpy as np
from darwin.material import Material, MU_0, EPSILON_0


@dataclass(frozen=True)
class LocalBlocks:
    K_AA: np.ndarray  # (n_edges, n_edges)
    K_Aphi: np.ndarray  # (n_edges, n_nodes)
    K_phiA: np.ndarray  # (n_nodes, n_edges)
    K_phiphi: np.ndarray  # (n_nodes, n_nodes)
    rhs_edge: np.ndarray  # (n_edges,)
    rhs_node: np.ndarray  # (n_nodes,)


# |det J| / (product of column lengths) is dimensionless. A shape-regular
# simplex is of order 1. 1e-12 rejects columns that are linearly dependent
# at roundoff and does not depend on the mesh length unit.
_JACOBIAN_RELATIVE_TOLERANCE = 1e-12


def checked_jacobian(J: np.ndarray) -> tuple[float, float]:
    """Return ``(det J, |det J|)`` after rejecting a degenerate map.

    The columns of ``J`` are the tangent vectors ``dx/dξ``. Their lengths
    scale the determinant so the test is a shape measure, not a volume in
    metres or millimetres. On a straight order-1 element ``J`` is constant,
    so every quadrature point repeats the same test. On a curved element
    ``J`` depends on the reference coordinate and the test has to run at
    each point.
    """
    det_J = float(np.linalg.det(J))
    scale = float(np.prod(np.linalg.norm(J, axis=0)))
    abs_det_J = abs(det_J)
    if scale == 0.0 or abs_det_J <= _JACOBIAN_RELATIVE_TOLERANCE * scale:
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
        x_nodes: np.ndarray,
        edges: list[tuple[int, int]],
        frequency: float,
        material: Material,
        gauss_points: GaussianQuadrature,
        source=None,
    ) -> LocalBlocks:
        """Darwin element matrices for one triangle or one tetrahedron.

        The basis, the Jacobian, and the reference curl come from the subclass.
        This method supplies only the Darwin coefficients: the edge mass is
        j*omega*sigma, and there is no -omega**2*epsilon term. A full-wave
        formulation is a second method on this class, using the same basis.
        """
        if material is None:
            raise ValueError(f"{type(self).__name__} requires a material")

        n_edges = len(edges)
        n_nodes = x_nodes.shape[0]
        n_dim = x_nodes.shape[1]

        omega = 2 * np.pi * frequency
        sigma = material.sigma
        epsilon = material.eps_r * EPSILON_0
        nu = 1.0 / (material.mu_r * MU_0)
        coeff_phi = sigma + 1j * omega * epsilon
        coeff_A = 1j * omega * sigma

        K_AA = np.zeros((n_edges, n_edges), dtype=complex)
        K_Aphi = np.zeros((n_edges, n_nodes), dtype=complex)
        K_phiA = np.zeros((n_nodes, n_edges), dtype=complex)
        K_phiphi = np.zeros((n_nodes, n_nodes), dtype=complex)
        rhs_edge = np.zeros(n_edges, dtype=complex)
        rhs_node = np.zeros(n_nodes, dtype=complex)

        for _, wi, x_hat in gauss_points:
            x_hat = np.asarray(x_hat, dtype=float)
            J = self.jacobian_matrix(x_hat, x_nodes)
            _, abs_det_J = checked_jacobian(J)
            inv_J = np.linalg.inv(J)

            lam = self.nodal_basis_functions(x_hat).ravel()
            w_phys = self.whiteney_basis_functions(x_hat, edges) @ inv_J
            curl_ref = self.curl_whiteney_basis_functions(x_hat, edges)
            grad_phys = self.gradient_nodal_basis_functions(x_hat) @ inv_J
            measure = wi * abs_det_J

            K_AA += wi * nu * _physical_curl_gram(curl_ref, J, abs_det_J)
            K_AA += measure * coeff_A * (w_phys @ w_phys.T)
            K_Aphi += measure * coeff_phi * (w_phys @ grad_phys.T)
            K_phiA += measure * coeff_A * (grad_phys @ w_phys.T)
            K_phiphi += measure * coeff_phi * (grad_phys @ grad_phys.T)

            if source is not None:
                raw = np.asarray(source(lam @ x_nodes), dtype=complex)
                current = raw.reshape(-1)
                if current.shape != (n_dim,):
                    raise ValueError(
                        f"source must return a length-{n_dim} vector, got shape {raw.shape}"
                    )
                rhs_edge += measure * (w_phys @ current)
                rhs_node += measure * (grad_phys @ current)

        return LocalBlocks(K_AA, K_Aphi, K_phiA, K_phiphi, rhs_edge, rhs_node)


def _physical_curl_gram(curl_ref: np.ndarray, J: np.ndarray, abs_det_J: float) -> np.ndarray:
    """Integrand of curl_x · curl_x, already multiplied by |det J|.

    A 2D curl is the scalar curl_ξ / det J. A 3D curl is J curl_ξ / det J.
    Both leave a factor 1/|det J| in front of a reference inner product.
    In 3D that inner product is curl_ξ · (J^T J curl_ξ), not curl_ξ · curl_ξ.
    """
    if curl_ref.shape[1] == 1:
        reference_inner = curl_ref @ curl_ref.T
    else:
        metric = J.T @ J
        reference_inner = curl_ref @ metric @ curl_ref.T
    return reference_inner / abs_det_J


class TetrahedronWhitney(Element):
    """Tetrahedron Whitney element of order 1"""

    def __init__(self, order: int = 1):
        assert order == 1, "Only order 1 is supported for now"
        self.order = 1

    def nodal_basis_functions(self, x_hat: np.ndarray) -> np.ndarray:
        """
        Args:
            x_hat: coordinates in the reference element (xi, eta, zeta)

        Returns:
            np.ndarray: nodal basis functions of shape (n_nodes, 1)
        """
        assert x_hat.shape == (3,), "x_hat must have 3 components"
        return np.array(
            [
                [1 - x_hat[0] - x_hat[1] - x_hat[2]],
                [x_hat[0]],
                [x_hat[1]],
                [x_hat[2]],
            ]
        )

    def gradient_nodal_basis_functions(self, x_hat: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the lambdas, which are constant for order

        Args:
            x_hat: coordinates in the reference element (xi, eta, zeta)

        Returns:
            np.ndarray: gradient of the nodal basis functions of shape (n_nodes, 3)
        """
        assert x_hat.shape == (3,), "x_hat must have 3 components"
        return np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def jacobian_matrix(self, x_hat: np.ndarray, x_nodes: np.ndarray) -> np.ndarray:
        """
        Computes the Jacobian matrix of the mapping from the reference element to the physical element.
        Args:
            x_hat: coordinates in the reference element (xi, eta, zeta)
            x_nodes: coordinates of the nodes of the tetrahedron (n_nodes, 3)
        Returns:
            np.ndarray: Jacobian matrix of shape (3, 3)
        """
        X = x_nodes  # shape (n_nodes, 3)
        grad_lambda = self.gradient_nodal_basis_functions(x_hat)  # shape (n_nodes, 3)
        # Jacobian is X^T @ grad_lambda, with shape (3, 3)
        J = X.T @ grad_lambda  # shape (3, 3)
        return J

    def whiteney_basis_functions(
        self, x_hat: np.ndarray, edges: list[tuple[int, int]]
    ) -> np.ndarray:
        """
        Computes the Whitney basis functions in reference element of shape (n_edges, 3)
        """
        n_edges = len(edges)
        lambdas = self.nodal_basis_functions(x_hat)
        grad_lambdas = self.gradient_nodal_basis_functions(x_hat)
        w = np.empty((n_edges, 3))
        for e, (i, j) in enumerate(edges):
            w[e] = lambdas[i] * grad_lambdas[j] - lambdas[j] * grad_lambdas[i]
        return w

    def curl_whiteney_basis_functions(
        self, x_hat: np.ndarray, edges: list[tuple[int, int]]
    ) -> np.ndarray:
        """
        Computes the curl of the Whitney basis functions of shape (n_edges, 3).
        """
        n_edges = len(edges)
        grad = self.gradient_nodal_basis_functions(x_hat)
        curl = np.empty((n_edges, 3))
        for e, (i, j) in enumerate(edges):
            gi = grad[i]
            gj = grad[j]
            curl[e] = 2.0 * np.cross(gi, gj)

        return curl


class TriangleWhitney(Element):
    """Triangle Whitney element of order 1"""

    def __init__(self, order: int = 1):
        assert order == 1, "Only order 1 is supported for now"
        self.order = 1

    def nodal_basis_functions(self, x_hat: np.ndarray) -> np.ndarray:
        """

        Computes the lambdas: nodal basis functions of shape (n_nodes, 1)

        Args:
            x_hat: coordinates in the reference element (xi, eta)

        Returns:
            np.ndarray: nodal basis functions of shape (n_nodes, 1)
        """
        assert x_hat.shape == (2,), "x_hat must have 2 components"
        return np.array(
            [
                [1 - x_hat[0] - x_hat[1]],
                [x_hat[0]],
                [x_hat[1]],
            ]
        )

    def gradient_nodal_basis_functions(self, x_hat: np.ndarray) -> np.ndarray:
        """

        Computes the gradient of the lambdas: nodal basis functions of shape (n_nodes, 2)

        Args:
        x_hat: coordinates in the reference element (xi, eta)

        Returns:
            np.ndarray: gradient of the nodal basis functions of shape (n_nodes, 2)
        """
        assert x_hat.shape == (2,), "x_hat must have 2 components"
        return np.array([[-1, -1], [1, 0], [0, 1]])

    def jacobian_matrix(self, x_hat: np.ndarray, x_nodes: np.ndarray) -> np.ndarray:
        """
        Computes the Jacobian matrix of the mapping from the reference element to the physical element.
        Args:
            x_hat: coordinates in the reference element (xi, eta)
            x_nodes: coordinates of the nodes of the triangle (n_nodes, 2)
        Returns:
            np.ndarray: Jacobian matrix of shape (2, 2)
        """
        # The Jacobian of the mapping from (xi, eta) to (x, y) is a 2x2 matrix whose columns are the derivatives
        # of the physical coordinates with respect to xi and eta.
        # Let x_hat = (xi, eta). The mapping is x = sum_i lambda_i(x_hat) * X_i where X_i are the node coordinates.
        # So, J = [dx/dxi, dx/deta; dy/dxi, dy/deta] computed as X^T @ grad_lambda, where X: (n_nodes, 2)
        X = x_nodes  # shape (n_nodes, 2)
        grad_lambda = self.gradient_nodal_basis_functions(x_hat)  # shape (n_nodes, 2)
        # Jacobian is X^T @ grad_lambda, with shape (2, 2)
        J = X.T @ grad_lambda  # shape (2,2)
        return J

    def whiteney_basis_functions(
        self, x_hat: np.ndarray, edges: list[tuple[int, int]]
    ) -> np.ndarray:
        """
        Computes the Whitney basis functions in reference element of shape (n_edges, 2)
        """
        n_edges = len(edges)
        lambdas = self.nodal_basis_functions(x_hat)
        grad_lambdas = self.gradient_nodal_basis_functions(x_hat)
        w = np.empty((n_edges, 2))
        for e, (i, j) in enumerate(edges):
            w[e] = lambdas[i] * grad_lambdas[j] - lambdas[j] * grad_lambdas[i]
        return w

    def curl_whiteney_basis_functions(
        self, x_hat: np.ndarray, edges: list[tuple[int, int]]
    ) -> np.ndarray:
        """
        Computes the curl of the Whitney basis functions of shape (n_edges, 1).
        Because the curl of a vector field on a planeis a scalar field along the normal direction,
        the result is a scalar field.
        """
        n_edges = len(edges)
        grad = self.gradient_nodal_basis_functions(x_hat)
        curl = np.empty((n_edges, 1))
        for e, (i, j) in enumerate(edges):
            gi = grad[i]
            gj = grad[j]
            curl[e, 0] = 2.0 * (gi[0] * gj[1] - gi[1] * gj[0])
        return curl


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
