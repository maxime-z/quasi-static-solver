from dataclasses import dataclass

from scipy.constants import mu_0, epsilon_0

from darwin.mesh_io import Mesh

MU_0 = mu_0
EPSILON_0 = epsilon_0


@dataclass
class Material:
    name: str
    sigma: float  # isotropic conductivity
    mu_r: float  # isotropic relative permeability real part
    eps_r: float  # isotropic relative permittivity real part


def materials_by_region(mesh: Mesh, library: dict[str, Material]) -> dict[int, Material]:
    """Map every region id that appears on the mesh to a Material.

    ``mesh.region_names`` turns an integer id into a string. ``library`` turns
    that string into a Material. Both lookups must succeed for every id used
    by at least one cell.
    """
    materials: dict[int, Material] = {}
    for region_id in sorted(set(int(r) for r in mesh.region_ids)):
        if region_id not in mesh.region_names:
            raise KeyError(f"region id {region_id} has no name in mesh.region_names")
        name = mesh.region_names[region_id]
        if name not in library:
            raise KeyError(f"region name {name!r} is missing from the material library")
        materials[region_id] = library[name]
    return materials
