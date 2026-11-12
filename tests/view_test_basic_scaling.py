"""Example to show the necessarity of scaling the matrix by block.
- K_aa, K_phiphi, K_phia, K_aphi
- Schur scaling

块缩放：A 块乘一个常数、φ 块乘一个常数（比如按块范数），只缩 A/φ，不动 C 和 p；
Schur 缩放：p 块按 diag(S)^{1/2} 缩，S 是消去 u 后的 C^T K^{-1} C。

"""

import numpy as np
import scipy.sparse as sp
from darwin.mesh_io import Mesh
from darwin.topology import tag_boundary, build_topology, Topology
from darwin.material import Material
from darwin.assemble import assemble
from darwin.elements import TriangleWhitney
from darwin.bc import DirichletWall, apply_dirichlet
from darwin.gauge import apply_gauge, GaugeKind
from darwin.solve import prepare, solve_direct


def _square(n: int = 4, size: float = 1.0) -> tuple[Mesh, Topology]:
    """Generate a square mesh."""
    h = size / n
    points = np.array(
        [[i * h, j * h] for i in range(n + 1) for j in range(n + 1)], dtype=float
    )
    cells = []
    for j in range(n):
        for i in range(n):
            n0 = j * (n + 1) + i
            n1 = n0 + 1
            n2 = n0 + (n + 1)
            n3 = n2 + 1
            cells.append([n0, n1, n3])
            cells.append([n0, n3, n2])

    mesh = Mesh(
        points, np.array(cells, dtype=int), np.zeros(2 * n * n, dtype=int), {0: "body"}
    )

    tags = tag_boundary(mesh, {"wall": lambda p: True})
    return mesh, build_topology(mesh, tags)

# def condition_number_sparse(a, k=1):
#     s_max = sp.linalg.svds(a, k=1, which='LM', return_singular_vectors=False)
#     s_min = sp.linalg.svds(a, k=1, which='SM', return_singular_vectors=False)
#     return float(np.max(s_max) / np.min(s_min))
def condition_number(a) -> float:
    return float(np.linalg.cond(np.asarray(a.toarray() if sp.issparse(a) else a)))


def _peak(block: sp.csr_matrix) -> float:
    """Compute the peak value of a sparse matrix."""
    if sp.issparse(block):
        if block.nnz == 0:
            return 0.0
        return float(np.max(np.abs(block.data)))
    else:
        block = np.asarray(block)
        return 0.0 if block.size == 0 else float(np.max(np.abs(block)))


def _report(name, block):
    """Report the peak value of a sparse matrix."""
    print(f"{name:16} {str(block.shape):16} max|.| = {_peak(block):.3e}")


def main():
    mesh, topology = _square()
    n_edges, n_nodes = topology.incidence_matrix.shape
    air = Material("air", sigma=0.0, mu_r=1.0, eps_r=1.0)
    materials = {0: air}
    system = assemble(
        mesh,
        topology,
        materials,
        frequency=2.0e6,  # 1MHz
        source=None,
        element=TriangleWhitney(),
    )

    # apply grounded wall
    system = apply_dirichlet(system, topology, [DirichletWall("wall", 0.0)])
    gauged = apply_gauge(system, topology, GaugeKind.COULOMB).system

    k = gauged.k_globals
    edge = slice(0, n_edges)
    node = slice(n_edges, n_edges + n_nodes)
    p = gauged.p_dofs

    print(f"number of edges: {n_edges}")
    print(f"number of nodes: {n_nodes}")
    print(f"number of p_dofs: {len(p)}")

    _report("K_aa", k[edge, edge])
    _report("K_phiphi", k[node, node])
    _report("K_phia", k[edge, node])
    _report("K_aphi", k[node, edge])
    _report("C", k[:, p])
    _report("C^T", k[p, :])
    _report("Z", k[p][:, p])

    print(f"condition number of k: {condition_number(k)}")

    # Block scaling, then Schur scaling of the multipliers.
    prepared = prepare(gauged)
    k_scaled = prepared.system.k_globals
    _report("K_aa_scaled", k_scaled[edge, edge])
    _report("K_phiphi_scaled", k_scaled[node, node])
    _report("K_phia_scaled", k_scaled[edge, node])
    _report("K_aphi_scaled", k_scaled[node, edge])
    _report("C_scaled", k_scaled[:, p])
    _report("C_T_scaled", k_scaled[p, :])
    _report("Z_scaled", k_scaled[p][:, p])

    print(f"condition number of k_scaled: {condition_number(k_scaled)}")
    scale = prepared.unknown_scale
    print(f"edge scale   min {scale[edge].min():.3e}  max {scale[edge].max():.3e}")
    print(f"node scale   min {scale[node].min():.3e}  max {scale[node].max():.3e}")
    print(f"multiplier   min {scale[p].min():.3e}  max {scale[p].max():.3e}")

    # solve_direct returns unknown_scale * y, the physical coefficient.
    x = solve_direct(gauged)
    residual = k @ x - gauged.f_globals
    load = np.linalg.norm(gauged.f_globals)
    relative = np.linalg.norm(residual) / load if load else np.linalg.norm(residual)
    print(f"recovered residual: {relative:.3e}")


if __name__ == "__main__":
    main()
