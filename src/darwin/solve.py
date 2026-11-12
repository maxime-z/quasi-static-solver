"""Scale the finished saddle and solve it.

Block scaling rescales the free edge and nodal unknowns. Schur scaling then
rescales the Coulomb multipliers from the block-scaled field matrix. The
scaled solve returns ``y``; the physical coefficient is ``unknown_scale * y``.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.sparse.linalg import splu, spsolve

from darwin.assemble import System

# A column whose block-scaled solve misses this relative residual has no Schur
# scale. The field block is singular when the Coulomb columns are what remove
# its kernel, and a failed column keeps scale 1.
_SCHUR_RESIDUAL_TOLERANCE = 1e-6


class BlockScalingType(Enum):
    MEAN = 0
    MAX = 1


@dataclass(frozen=True)
class Prepared:
    """Scaled system and the map back to the physical coefficients.

    ``spsolve(system.k_globals, system.f_globals)`` returns the scaled unknown
    ``y``. The physical coefficient, in the original units of ``A``, ``phi``
    and the multiplier, is ``unknown_scale * y``.
    """

    system: System
    unknown_scale: np.ndarray


def prepare(
    system: System, scale_type: BlockScalingType = BlockScalingType.MAX
) -> Prepared:
    """Block-scale the field unknowns.

    Schur scaling of the multipliers is disabled at execution. In the Coulomb
    formulation the field block is singular: the multiplier rows are exactly
    what remove the gauge kernel, so ``C^T K^{-1} C`` does not exist and the
    field-block factorization has no meaningful residual. The function
    ``schur_scaling`` is kept for an invertible field block and as a reference;
    a genuine multiplier scaling must factor the full saddle, not the field
    block.
    """
    return block_scaling(system, scale_type)


def block_scaling(
    system: System, scale_type: BlockScalingType = BlockScalingType.MAX
) -> Prepared:
    """Rescale free edges by ``sqrt(s_A)`` and free nodes by ``sqrt(s_phi)``.

    Constraint rows and multiplier columns stay at scale 1. ``s_A`` and
    ``s_phi`` are the max or the mean absolute entry of the corresponding
    free block, zeros included.
    """
    k = system.k_globals.tocsr()
    n_edge = len(system.edge_dofs)
    n_node = len(system.node_dofs)
    n_p = 0 if system.p_dofs is None else len(system.p_dofs)
    n_total = n_edge + n_node
    k_field = k[:n_total, :n_total]

    free_edges = np.array(
        [i for i in range(n_edge) if not _is_constraint_row(k_field, i)],
        dtype=int,
    )
    free_nodes = np.array(
        [
            i
            for i in range(n_edge, n_edge + n_node)
            if not _is_constraint_row(k_field, i)
        ],
        dtype=int,
    )

    unknown_scale = np.ones(n_total + n_p)
    unknown_scale[free_edges] = 1.0 / _block_level(k_field, free_edges, scale_type)
    unknown_scale[free_nodes] = 1.0 / _block_level(k_field, free_nodes, scale_type)
    k_scaled, f_scaled = _congruence(k, system.f_globals, unknown_scale)
    return Prepared(_replace_matrix(system, k_scaled, f_scaled), unknown_scale)


def schur_scaling(prepared: Prepared) -> Prepared:
    """Rescale each multiplier by ``sqrt(|S_jj|)`` of the block-scaled field block.

    ``S = C'^T (K')^{-1} C'`` uses the matrix already returned by
    ``block_scaling``. A column that the factorization cannot solve keeps
    scale 1.
    """
    system = prepared.system
    p_dofs = system.p_dofs
    if p_dofs is None or len(p_dofs) == 0:
        return prepared

    k = system.k_globals.tocsr()
    n_total = len(system.edge_dofs) + len(system.node_dofs)
    k_field = k[:n_total, :n_total].tocsc()
    c = k[:n_total, p_dofs].tocsc()
    try:
        factors = splu(k_field)
    except RuntimeError:
        return prepared

    column_scale = np.ones(len(p_dofs))
    for j in range(len(p_dofs)):
        column = np.asarray(c[:, j].todense()).ravel()
        column_norm = np.linalg.norm(column)
        if column_norm == 0.0:
            continue
        solved = factors.solve(column)
        residual = np.linalg.norm(k_field @ solved - column) / column_norm
        if residual > _SCHUR_RESIDUAL_TOLERANCE:
            continue
        pivot = abs(np.dot(column, solved))
        if pivot > 0.0:
            column_scale[j] = 1.0 / np.sqrt(pivot)

    unknown_scale = prepared.unknown_scale.copy()
    extra = np.ones_like(unknown_scale)
    extra[p_dofs] = column_scale
    unknown_scale[p_dofs] *= column_scale
    k_scaled, f_scaled = _congruence(k, system.f_globals, extra)
    return Prepared(_replace_matrix(system, k_scaled, f_scaled), unknown_scale)


def solve_direct(system: System) -> np.ndarray:
    """Return the physical coefficients of the block- and Schur-scaled system."""
    prepared = prepare(system)
    scaled_solution = spsolve(prepared.system.k_globals, prepared.system.f_globals)
    return prepared.unknown_scale * np.asarray(scaled_solution)


def _block_level(k_field, dofs: np.ndarray, scale_type: BlockScalingType) -> float:
    """Absolute level of one free diagonal block. An empty or zero block stays 1."""
    if dofs.size == 0:
        return 1.0
    selected = np.zeros(k_field.shape[0], dtype=bool)
    selected[dofs] = True
    coo = k_field.tocoo()
    inside = selected[coo.row] & selected[coo.col]
    magnitudes = np.abs(coo.data[inside])
    if scale_type is BlockScalingType.MEAN:
        total = magnitudes.sum() if magnitudes.size else 0.0
        level = total / dofs.size**2
    else:
        level = magnitudes.max() if magnitudes.size else 0.0
    if level == 0.0:
        return 1.0
    return float(np.sqrt(level))


def _congruence(k, f, scale):
    """Return ``D K D`` and ``D f`` for a real diagonal ``scale``."""
    coo = k.tocoo(copy=True)
    coo.data = np.asarray(coo.data) * scale[coo.row] * scale[coo.col]
    return coo.tocsr(), np.asarray(f, dtype=complex) * scale


def _replace_matrix(system: System, k, f) -> System:
    return System(
        k.tocsr(),
        np.asarray(f, dtype=complex),
        system.edge_dofs.copy(),
        system.node_dofs.copy(),
        None if system.p_dofs is None else system.p_dofs.copy(),
        system.m_edge_globals,
    )


def _is_constraint_row(k, i: int) -> bool:
    """True for an identity row and for a floating slave row ``phi_s - phi_m = 0``."""
    row = k.getrow(i)
    cols = row.indices
    vals = row.data
    if row.nnz == 1 and int(cols[0]) == i and np.isclose(vals[0], 1.0):
        return True
    if row.nnz == 2:
        diag = vals[cols == i]
        off = vals[cols != i]
        if (
            diag.size == 1
            and off.size == 1
            and np.isclose(diag[0], 1.0)
            and np.isclose(off[0], -1.0)
        ):
            return True
    return False
