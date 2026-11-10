"""Reduce the Darwin model to the electrostatic case.

with w (frequency) = 0, and J = 0, the Darwin equation reduces to the electrostatic equation:

∇²φ = -ρ / ε₀

where φ is the electric potential, ρ is the charge density, and ε₀ is the vacuum permittivity.

The Darwin equation is:

∇²φ = -jωμ₀H - σE

where H is the magnetic field, σ is the conductivity, and E is the electric field.

"""

import numpy as np
import meshio
import gmsh
from darwin.mesh_io import Mesh
from darwin.topology import (
    build_topology,
    skin_facets,
    tag_boundary,
    Boundary,
    Topology,
)
from darwin.material import Material, EPSILON_0
from darwin.elements import TriangleWhitney
from darwin.assemble import assemble
from darwin.bc import DirichletWall, apply_dirichlet
import scipy
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


from pathlib import Path

# result directory
result_dir = Path(__file__).parent / "results"
mesh_file = result_dir / "square.msh"


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


def cell_electric_field(mesh: Mesh, element: TriangleWhitney, phi: np.ndarray):
    """Constant E = -∇φ on each triangle, from the physical gradients of λ."""
    centroids = np.empty((len(mesh.cells), 2))
    fields = np.empty((len(mesh.cells), 2))
    x_hat = np.array([1.0 / 3.0, 1.0 / 3.0])
    for i_cell, cell in enumerate(mesh.cells):
        nodes = mesh.points[cell]
        grad = element.gradient_nodal_basis_functions(x_hat) @ np.linalg.inv(
            element.jacobian_matrix(x_hat, nodes)
        )
        fields[i_cell] = -(phi[cell] @ grad).real
        centroids[i_cell] = nodes.mean(axis=0)
    return centroids, fields


def plot_potential_and_field(mesh: Mesh, phi: np.ndarray, path: Path):
    centroids, e_field = cell_electric_field(mesh, TriangleWhitney(), phi)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    ax = axes[0]
    triangles = ax.tripcolor(
        mesh.points[:, 0],
        mesh.points[:, 1],
        mesh.cells,
        phi,
        shading="gouraud",
        cmap="coolwarm",
    )
    fig.colorbar(triangles, ax=ax, label="φ")
    ax.set_aspect("equal")
    ax.set_title("potential")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax = axes[1]
    ax.tripcolor(
        mesh.points[:, 0],
        mesh.points[:, 1],
        mesh.cells,
        phi,
        shading="gouraud",
        cmap="coolwarm",
        alpha=0.85,
    )
    step = max(1, len(centroids) // 350)
    ax.quiver(
        centroids[::step, 0],
        centroids[::step, 1],
        e_field[::step, 0],
        e_field[::step, 1],
        color="black",
        angles="xy",
        scale_units="xy",
        scale=40,
    )
    ax.set_aspect("equal")
    ax.set_title("E = −∇φ")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"figure: {path}")
    plt.show()


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

# get all boundary edges
boundary_edges = skin_facets(darwin_mesh)

predicates = {
    "bottom": lambda x: np.isclose(x[1], -1) and x[0] > -0.5 and x[0] < 0.5,
    "top": lambda x: np.isclose(x[1], 1) and x[0] > -0.5 and x[0] < 0.5,
    "left": lambda x: np.isclose(x[0], -1),
    "right": lambda x: np.isclose(x[0], 1),
}

tagged_facets, tags = tag_boundary(darwin_mesh, predicates)

# build topology
topology = build_topology(darwin_mesh, boundary_tags=(tagged_facets, tags))

plot_domain(darwin_mesh, topology)


# assemble the system
air = Material(name="air", sigma=0.0, mu_r=1.0, eps_r=2.0)
materials = {0: air}
fake_frequency = 1 / (2 * np.pi)

element = TriangleWhitney()
system = assemble(darwin_mesh, topology, materials, fake_frequency, None, element)
# n_edges = topology.edges.shape[0]
# k_phi_phi = (
#     np.imag(system.k_globals[n_edges:, n_edges:]) / EPSILON_0
# )  # eps_r*int(grad(phi).grad(phi))

linear_system_analysis(system.k_globals)

#
n_edges = len(topology.edges)
K_aa = system.k_globals[:n_edges, :n_edges]
print(f"# ===> K_aa shape: {K_aa.shape}")
linear_system_analysis(K_aa)

K_phi_phi = system.k_globals[n_edges:, n_edges:]
print(f"# ===> K_phi_phi shape: {K_phi_phi.shape}")
linear_system_analysis(K_phi_phi)


# boundary conditions
bottom_voltage_BC = DirichletWall(tag="bottom", voltage=-10.0)
top_voltage_BC = DirichletWall(tag="top", voltage=10.0)


contrainted_system = apply_dirichlet(
    system, topology, [bottom_voltage_BC, top_voltage_BC]
)

# The electrode rows are the real statements φ = V. Taking the imaginary part
# after apply_dirichlet turns those diagonals into 0, so the complex block is
# the system to solve. jω cancels in the free rows when ρ = 0.
n_edges = len(topology.edges)
k_phi_phi = contrainted_system.k_globals[n_edges:, n_edges:] / EPSILON_0
rhs_phi = contrainted_system.f_globals[n_edges:] / EPSILON_0
x = scipy.sparse.linalg.spsolve(k_phi_phi.tocsr(), rhs_phi)
phi = np.real(x)

x_imag = np.imag(x)
print(f"norm x_real vs x_imag: {np.linalg.norm(phi)} vs {np.linalg.norm(x_imag)}")
print(f"max x_real vs x_imag: {np.max(phi)} vs {np.max(x_imag)}")

x_iterative = scipy.sparse.linalg.lsqr(k_phi_phi.tocsr(), rhs_phi)[0]

# check the singular value and condition number of the system and rank
k_phi_phi_dense = k_phi_phi.toarray()
singular_values = np.linalg.svd(k_phi_phi_dense, compute_uv=False)
condition_number_form_singular_values = singular_values[0] / singular_values[-1]
# print(f"singular values: {singular_values}")
print(f"condition number from singular values: {condition_number_form_singular_values}")
condition_number_from_scipy = np.linalg.cond(k_phi_phi_dense)
print(f"condition number from scipy: {condition_number_from_scipy}")

matrix_rank = np.linalg.matrix_rank(k_phi_phi_dense)
print(f"matrix rank: {matrix_rank}")


residual = k_phi_phi @ x - rhs_phi
print(f"original rhs norm: {np.linalg.norm(rhs_phi)}")
print(f"residual norm: {np.linalg.norm(residual)}")
print(f"φ real range: {phi.min():.3f} .. {phi.max():.3f}")
print(f"max |Im φ|: {np.max(np.abs(np.imag(x))):.3e}")

plot_potential_and_field(darwin_mesh, phi, result_dir / "electrostatic.png")
