from dataclasses import dataclass, field

import numpy as np


@dataclass
class Mesh:
    points: np.ndarray  # (n_points, n_dims)
    cells: np.ndarray  # (n_cells, n_nodes)
    region_ids: np.ndarray  # (n_cells,), cell i has region region_ids[i]
    region_names: dict[int, str] = field(default_factory=dict)
