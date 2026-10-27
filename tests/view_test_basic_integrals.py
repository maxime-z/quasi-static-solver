"""M2: one right triangle, plus a reference-tetrahedron smoke test.

Run from anywhere:
    python tests/view_test_m2.py
Figures are written next to this file and shown if a display is available.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from darwin.elements import GaussianQuadrature, TetrahedronWhitney, TriangleWhitney
from darwin.material import MU_0, Material


# Vertices (0,0), (1,0), (0,1). det J = 1, area = 1/2.
# mu_r = 1/mu0 makes nu = 1/(mu_r mu0) = 1. sigma = eps = 0.
NODES = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
EDGES = [(0, 1), (0, 2), (1, 2)]
REFERENCE = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


def hand_curl_curl():
    """∫ curl_e curl_f dA on this triangle. Curls are constant and det J = 1."""
    curl_xi = np.array([2.0, -2.0, 2.0])
    area = 0.5
    return area * np.outer(curl_xi, curl_xi)


def circulations(element, nodes, edges, reference):
    """∫_edge_f w_e · τ dl. Rows are basis functions, columns are edges."""
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(4)
    t = 0.5 * (gauss_nodes + 1.0)
    weight = 0.5 * gauss_weights
    n_edges = len(edges)
    table = np.zeros((n_edges, n_edges))
    for f, (i, j) in enumerate(edges):
        tangent = nodes[j] - nodes[i]
        for tk, wk in zip(t, weight):
            x_hat = (1.0 - tk) * reference[i] + tk * reference[j]
            J = element.jacobian_matrix(x_hat, nodes)
            w_phys = element.whiteney_basis_functions(x_hat, edges) @ np.linalg.inv(J)
            table[:, f] += wk * (w_phys @ tangent)
    return table


def gradient_identity_error(element, samples, q, nodes, edges):
    """max |sum_e (q_j - q_i) w_e - sum_k q_k ∇λ_k| over reference points."""
    worst = 0.0
    for x_hat in samples:
        x_hat = np.asarray(x_hat, dtype=float)
        J = element.jacobian_matrix(x_hat, nodes)
        inv_J = np.linalg.inv(J)
        w_phys = element.whiteney_basis_functions(x_hat, edges) @ inv_J
        grad = element.gradient_nodal_basis_functions(x_hat) @ inv_J
        from_edges = np.zeros(x_hat.shape[0])
        for e, (i, j) in enumerate(edges):
            from_edges += (q[j] - q[i]) * w_phys[e]
        from_nodes = q @ grad
        worst = max(worst, np.linalg.norm(from_edges - from_nodes))
    return worst


def plot_edge_functions(element, nodes, edges, path):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharex=True, sharey=True)
    xs = np.linspace(0.02, 0.96, 8)
    samples = [(xi, eta) for xi in xs for eta in xs if xi + eta < 0.98]
    triangle = np.vstack([nodes, nodes[0]])
    for e, ax in enumerate(axes):
        points = []
        vectors = []
        for xi, eta in samples:
            x_hat = np.array([xi, eta])
            J = element.jacobian_matrix(x_hat, nodes)
            w_xy = element.whiteney_basis_functions(x_hat, edges) @ np.linalg.inv(J)
            points.append(x_hat @ J.T + nodes[0])
            vectors.append(w_xy[e])
        points = np.array(points)
        vectors = np.array(vectors)
        ax.plot(triangle[:, 0], triangle[:, 1], color="black", linewidth=1.2)
        ax.quiver(
            points[:, 0],
            points[:, 1],
            vectors[:, 0],
            vectors[:, 1],
            angles="xy",
            scale_units="xy",
            scale=8,
            color="C0",
        )
        i, j = edges[e]
        ax.annotate(
            "",
            xy=nodes[j],
            xytext=nodes[i],
            arrowprops=dict(arrowstyle="->", color="C3", lw=1.5),
        )
        ax.set_aspect("equal")
        ax.set_title(f"w on edge {i}→{j}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.suptitle("Whitney edge functions on the unit right triangle")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"figure: {path}")
    plt.show()


def main():
    material = Material(name="unit", sigma=0.0, mu_r=1.0 / MU_0, eps_r=0.0)
    element = TriangleWhitney()
    nu = 1.0 / (material.mu_r * MU_0)
    blocks = element.darwin_local_forms(
        NODES,
        EDGES,
        frequency=0.0,
        material=material,
        gauss_points=GaussianQuadrature(dim=2, degree=2),
    )
    expected = hand_curl_curl()

    print(f"nu = {nu:.6e}, sigma = {material.sigma}, eps_r = {material.eps_r}")
    print("K_AA")
    print(np.array2string(blocks.K_AA.real, precision=6, suppress_small=True))
    print("hand integral of curl curl")
    print(np.array2string(expected, precision=6, suppress_small=True))
    aa_error = np.max(np.abs(blocks.K_AA - expected))
    print(f"max |K_AA - hand| = {aa_error:.3e}")

    circulation = circulations(element, NODES, EDGES, REFERENCE)
    print("circulations, entry (basis, edge), expect I")
    print(np.array2string(circulation, precision=6, suppress_small=True))
    circulation_error = np.max(np.abs(circulation - np.eye(3)))
    print(f"max |circulation - I| = {circulation_error:.3e}")

    triangle_samples = [(1.0 / 3.0, 1.0 / 3.0), (0.2, 0.2), (0.5, 0.2), (0.1, 0.6)]
    grad_error = gradient_identity_error(
        element, triangle_samples, np.array([1.0, 3.0, -2.0]), NODES, EDGES
    )
    print(f"max |sum (q_j-q_i) w_e - grad q| = {grad_error:.3e}")

    print(f"max |K_Aphi| = {np.max(np.abs(blocks.K_Aphi)):.3e}")
    print(f"max |K_phiA| = {np.max(np.abs(blocks.K_phiA)):.3e}")
    print(f"max |K_phiphi| = {np.max(np.abs(blocks.K_phiphi)):.3e}")
    print(
        f"max |rhs| = {max(np.max(np.abs(blocks.rhs_edge)), np.max(np.abs(blocks.rhs_node))):.3e}"
    )

    figure_path = Path(__file__).resolve().parent / "m2_right_triangle.png"
    plot_edge_functions(element, NODES, EDGES, figure_path)

    if aa_error > 1e-12 or circulation_error > 1e-12 or grad_error > 1e-12:
        raise SystemExit("M2 triangle check failed")
    print("M2 triangle check passed")
    tet_smoke(material)


# Reference tet (0,0,0), (1,0,0), (0,1,0), (0,0,1). det J = 1, volume = 1/6.
TET_NODES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
TET_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
TET_REFERENCE = TET_NODES.copy()


def hand_tet_curl_curl():
    """∫ curl_e · curl_f dV. J = I, so the physical curl is the reference curl."""
    grads = np.array([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    curls = np.vstack([2.0 * np.cross(grads[i], grads[j]) for i, j in TET_EDGES])
    return (1.0 / 6.0) * (curls @ curls.T)


def plot_tet(path):
    fig = plt.figure(figsize=(5.2, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    for i, j in TET_EDGES:
        ax.plot(*TET_NODES[[i, j]].T, color="black", linewidth=1.2)
        ax.quiver(
            *TET_NODES[i],
            *(TET_NODES[j] - TET_NODES[i]),
            color="C3",
            arrow_length_ratio=0.15,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Reference tetrahedron, edges low index to high")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"figure: {path}")
    plt.show()


def tet_smoke(material):
    element = TetrahedronWhitney()
    blocks = element.darwin_local_forms(
        TET_NODES,
        TET_EDGES,
        frequency=0.0,
        material=material,
        gauss_points=GaussianQuadrature(dim=3, degree=2),
    )
    expected = hand_tet_curl_curl()

    print("tet shapes", blocks.K_AA.shape, blocks.K_Aphi.shape, blocks.K_phiA.shape, blocks.K_phiphi.shape)
    print("K_AA")
    print(np.array2string(blocks.K_AA.real, precision=6, suppress_small=True))
    print("hand integral of curl · curl")
    print(np.array2string(expected, precision=6, suppress_small=True))
    aa_error = np.max(np.abs(blocks.K_AA - expected))
    print(f"max |K_AA - hand| = {aa_error:.3e}")

    circulation = circulations(element, TET_NODES, TET_EDGES, TET_REFERENCE)
    circulation_error = np.max(np.abs(circulation - np.eye(6)))
    print(f"max |circulation - I| = {circulation_error:.3e}")

    tet_samples = [(0.25, 0.25, 0.25), (0.2, 0.2, 0.2), (0.5, 0.2, 0.1), (0.1, 0.2, 0.3)]
    grad_error = gradient_identity_error(
        element, tet_samples, np.array([1.0, 3.0, -2.0, 0.5]), TET_NODES, TET_EDGES
    )
    print(f"max |sum (q_j-q_i) w_e - grad q| = {grad_error:.3e}")
    print(f"max |K_Aphi| = {np.max(np.abs(blocks.K_Aphi)):.3e}")
    print(f"max |K_phiA| = {np.max(np.abs(blocks.K_phiA)):.3e}")
    print(f"max |K_phiphi| = {np.max(np.abs(blocks.K_phiphi)):.3e}")

    figure_path = Path(__file__).resolve().parent / "m2_reference_tet.png"
    plot_tet(figure_path)

    if aa_error > 1e-12 or circulation_error > 1e-12 or grad_error > 1e-12:
        raise SystemExit("M2 tetrahedron smoke test failed")
    print("M2 tetrahedron smoke test passed")


if __name__ == "__main__":
    main()
