from pathlib import Path
import gmsh
import meshio
import numpy as np
from darwin.mesh_io import Mesh
from darwin.topology import build_topology
from darwin.topology import skin_facets
from darwin.topology import Topology
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from darwin.material import MU_0, Material
from darwin.assemble import assemble
from darwin.elements import TriangleWhitney
from darwin.bc import DirichletWall, apply_dirichlet
import scipy.sparse
from collections import defaultdict, deque


def linear_system_analysis(k):
    """
    Analyze a linear system by checking the singular value and condition number of the system and rank
    if k is a sparse matrix, convert it to a dense matrix
    if k is a dense matrix, use the built-in functions
    """
    if scipy.sparse.issparse(k):
        k = k.toarray()

    singular_values = np.linalg.svd(k, compute_uv=False)
    condition_number_form_singular_values = singular_values[0] / singular_values[-1]
    singular_values = np.linalg.svd(k, compute_uv=False)
    condition_number_form_singular_values = singular_values[0] / singular_values[-1]
    print(
        f"condition number from singular values: {condition_number_form_singular_values}"
    )
    condition_number_from_scipy = np.linalg.cond(k)
    print(f"condition number from scipy: {condition_number_from_scipy}")
    matrix_rank = np.linalg.matrix_rank(k)
    print(f"matrix rank: {matrix_rank}")
    return (
        condition_number_form_singular_values,
        condition_number_from_scipy,
        matrix_rank,
    )


def plot_domain(mesh, topology: Topology | None = None):

    points = mesh.points
    cells = mesh.cells
    fig, ax = plt.subplots()
    triangle_cell_list = [points[c] for c in cells]
    ax.add_collection(
        PolyCollection(
            triangle_cell_list, facecolors="#cfe8ff", edgecolors="0.3", linewidths=0.4
        )
    )
    if topology is not None:
        skin_edges = topology.edges[topology.skin.edges]
        for a, b in skin_edges:
            xs = points[[a, b], 0]
            ys = points[[a, b], 1]
            ax.plot(xs, ys, "C3", lw=2)
    ax.set_aspect("equal")
    ax.set_title("mesh and skin")
    fig.tight_layout()
    fig.savefig(result_dir / "square_mesh.png", dpi=140)
    # plt.show()


result_dir = Path(__file__).parent / "results"
mesh_file = result_dir / "square.msh"

if not mesh_file.exists():
    print(f"Mesh file {mesh_file} does not exist. Generating...")
    gmsh.initialize()
    gmsh.model.add("plates")
    outer = gmsh.model.occ.addRectangle(-1, -1, 0, 2, 2)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.1)
    gmsh.model.mesh.generate(dim=2)
    gmsh.write(str(mesh_file))
    gmsh.finalize()
else:
    print(f"Mesh file {mesh_file} exists. Using existing mesh.")

mesh_raw = meshio.read(mesh_file)

print(f"number of points: {len(mesh_raw.points)}")
print(f"number of cells: {len(mesh_raw.cells)}")

# build mesh and boundary

points = mesh_raw.points[:, :2]
cells = mesh_raw.cells_dict["triangle"]
region_ids = np.zeros(len(cells), dtype=int)
region_names = {0: "air"}
darwin_mesh = Mesh(points, cells, region_ids, region_names)
boundary_edges = skin_facets(darwin_mesh)
tagged_skin = ["wall" for _ in boundary_edges]

topology = build_topology(darwin_mesh, boundary_tags=(boundary_edges, tagged_skin))
plot_domain(darwin_mesh, topology)


# define the current sources
J0 = 1.0
inner, outer = 0.4, 0.7


def loop_current(x):
    x, y = float(x[0]), float(x[1])
    if max(abs(x), abs(y)) > outer or max(abs(x), abs(y)) < inner:
        return np.zeros(2)
    if abs(y) >= abs(x):
        return np.array([-J0 * np.sign(y), 0.0])
    return np.array([0.0, J0 * np.sign(x)])


def plot_problem(mesh: Mesh, topology: Topology, source, path: Path):
    """PEC wall and the circulating source current, before the solve."""
    points = mesh.points
    cells = mesh.cells
    centroids = np.array([points[cell].mean(axis=0) for cell in cells])
    current = np.vstack([np.asarray(source(x), dtype=float) for x in centroids])
    carries = np.linalg.norm(current, axis=1) > 0
    colors = ["#f6c38a" if on else "#e8eef5" for on in carries]

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.add_collection(
        PolyCollection(
            [points[cell] for cell in cells],
            facecolors=colors,
            edgecolors="0.75",
            linewidths=0.3,
        )
    )
    wall_edges = topology.edges[topology.boundary["wall"].edges]
    for i_edge, (a, b) in enumerate(wall_edges):
        ax.plot(
            points[[a, b], 0],
            points[[a, b], 1],
            color="C3",
            lw=2.5,
            label="PEC wall, n×A = 0" if i_edge == 0 else None,
        )
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
        label="source current J",
    )
    ax.set_aspect("equal")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("problem: PEC wall and source current")
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"figure: {path}")


plot_problem(darwin_mesh, topology, loop_current, result_dir / "magnetostatic_setup.png")


# assemble the system
air = Material(name="air", sigma=0.0, mu_r=1.0, eps_r=1.0)
materials = {0: air}
fake_frequency = 0.0

element = TriangleWhitney()
system = assemble(
    darwin_mesh, topology, materials, fake_frequency, loop_current, element
)

# apply the boundary conditions
dirichlet_walls = [DirichletWall(tag="wall", voltage=0.0)]
system = apply_dirichlet(system, topology, dirichlet_walls)


# analyze the system before applying the Tree-Cotree gauge
n_edges = len(topology.edges)
rhs_edge = system.f_globals[:n_edges]
print(
    f"# ===> rhs_edge max and min amplitude: {np.max(np.abs(rhs_edge))}, {np.min(np.abs(rhs_edge))}"
)
K_aa = system.k_globals[:n_edges, :n_edges]
print(f"# ===> K_aa shape: {K_aa.shape}")
linear_system_analysis(K_aa)


# apply the Tree-Cotree gauge
def _is_identity_row(K, i) -> bool:
    """
    Check if the i-th row of the matrix K is the identity row
    K: scipy.sparse.csr_matrix
    i: int
    return: bool
    """
    row = K.getrow(i)
    return row.nnz == 1 and int(row.indices[0]) == i and np.isclose(row.data[0], 1.0)


def apply_tree_cotree_gauge(K_aa, rhs_edge, topology, root_nodes):
    K = K_aa.tolil(copy=True)
    rhs = np.array(rhs_edge, dtype=complex, copy=True)

    # 1. Free edges
    neighors = defaultdict(list)
    for e in range(K.shape[0]):
        if _is_identity_row(K, e):
            continue
        else:
            n1, n2 = (int(n) for n in topology.edges[e])
            neighors[n1].append((n2, e))
            neighors[n2].append((n1, e))

    # 2. Every PEC node is a root. No tree edge is associated to a root
    visited = {int(n) for n in root_nodes}
    queue = deque(visited)
    tree_edges = []

    # 3. First free edge that reachs an unvisited node becomes a tree edge.
    # An edge whoese far end i already visited closes a loop, so it stays cotree.
    while queue:
        u = queue.popleft()
        for v, e in neighors[u]:
            if v in visited:
                continue
            visited.add(v)
            tree_edges.append(e)
            queue.append(v)

    # 4. The free edges must reach every node which is a spanning tree.
    n_nodes = topology.incidence_matrix.shape[1]
    if len(visited) != n_nodes:
        raise RuntimeError(
            f"{n_nodes - len(visited)} nodes are not reachable from the root nodes"
        )

    # 5. same identity row steps as a homogeneous Dirichlet edge.
    for e in tree_edges:
        K[e, :] = 0.0
        K[:, e] = 0.0
        K[e, e] = 1.0
        rhs[e] = 0.0

    return K.tocsr(), rhs, tree_edges


K_aa_gauge, rhs_edge_gauge, tree_edges = apply_tree_cotree_gauge(
    K_aa, rhs_edge, topology, topology.boundary["wall"].nodes
)

# analyze the system after applying the Tree-Cotree gauge
linear_system_analysis(K_aa_gauge)

n_interior = len(points) - len(topology.boundary["wall"].nodes)
print(f"# ===> tree edges: {len(tree_edges)}, interior nodes: {n_interior}")

a = scipy.sparse.linalg.spsolve(K_aa_gauge, rhs_edge_gauge)
residual = K_aa_gauge @ a - rhs_edge_gauge
print(f"relative residual: {np.linalg.norm(residual) / np.linalg.norm(rhs_edge_gauge):.3e}")
print(f"max |Im a|: {np.max(np.abs(np.imag(a))):.3e}")


def cell_bz(mesh: Mesh, topology: Topology, element: TriangleWhitney, a_edge: np.ndarray):
    """Constant Bz = curl A on each triangle."""
    centroids = np.empty((len(mesh.cells), 2))
    bz = np.empty(len(mesh.cells))
    areas = np.empty(len(mesh.cells))
    x_hat = np.array([1.0 / 3.0, 1.0 / 3.0])
    for i_cell, cell in enumerate(mesh.cells):
        nodes = mesh.points[cell]
        local_of = {int(g): i for i, g in enumerate(cell)}
        edge_ids = topology.cell_edges[i_cell]
        pairs = [
            (local_of[int(n1)], local_of[int(n2)])
            for n1, n2 in topology.edges[edge_ids]
        ]
        jacobian = element.jacobian_matrix(x_hat, nodes)
        curl_ref = element.curl_whiteney_basis_functions(x_hat, pairs)[:, 0]
        det_j = float(np.linalg.det(jacobian))
        bz[i_cell] = np.real(a_edge[edge_ids] @ curl_ref) / det_j
        centroids[i_cell] = nodes.mean(axis=0)
        areas[i_cell] = 0.5 * abs(det_j)
    return centroids, bz, areas


centroids, bz, areas = cell_bz(darwin_mesh, topology, element, a)
print(f"net flux ∫ Bz dA: {np.sum(bz * areas):.3e}")
print(f"Bz range: {bz.min():.3e} .. {bz.max():.3e}")

window = np.linalg.norm(centroids, axis=1) < 0.15
outside = (centroids[:, 0] > 0.75) & (np.abs(centroids[:, 1]) < 0.15)
hz_jump = bz[window].mean() / MU_0 - bz[outside].mean() / MU_0
print(f"Hz(window) - Hz(outside the right leg): {hz_jump:.4f}")
print(f"current across the right leg, J0*(outer-inner): {J0 * (outer - inner):.4f}")


def plot_bz(mesh: Mesh, bz_cell: np.ndarray, path: Path):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    triangles = ax.tripcolor(
        mesh.points[:, 0],
        mesh.points[:, 1],
        mesh.cells,
        bz_cell,
        shading="flat",
        cmap="coolwarm",
    )
    fig.colorbar(triangles, ax=ax, label="Bz")
    ax.set_aspect("equal")
    ax.set_title("Bz = curl A")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"figure: {path}")


plot_bz(darwin_mesh, bz, result_dir / "magnetostatic.png")
