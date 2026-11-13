"""Formulation module to allow choosing between different MQS and Darwin"""

from enum import Enum
from dataclasses import dataclass

import numpy as np
from darwin.material import Material, EPSILON_0


class Formulation(Enum):
    """Formulation of the MQS and Darwin equations."""

    DARWIN = "darwin"  # weak Gauss row, no gauge
    MQS = "mqs"  # eps = 0


class NodalEquation(Enum):
    """2nd equation for the load."""

    CONTINUITY = "continuity"  # integral grad(lambda). J_s
    GAUSS = "gauss"  # integral lambda*rho


@dataclass(frozen=True)
class Coefficients:
    """Coefficient for local/elemental block matrix for different formulations."""

    a_phi: complex
    phi_a: complex
    phi_phi: complex
    nodal: NodalEquation


def coefficients(
    formulation: Formulation, material: Material, frequency: float
) -> Coefficients:
    """Return the coefficients for the local/elemental block matrix for different formulations."""
    omega = 2 * np.pi * frequency
    sigma = material.sigma
    eps = material.eps_r * EPSILON_0
    if formulation == Formulation.MQS:
        return Coefficients(sigma, 1j * omega * sigma, sigma, NodalEquation.CONTINUITY)
    elif formulation == Formulation.DARWIN:
        if eps <= 0:
            raise ValueError(
                "Darwin needs eps_r > 0 in every region: the Gauss row vanishes "
                "where eps = 0. Use Formulation.MQS for a zero-permittivity region."
            )
        return Coefficients(
            sigma + 1j * omega * eps, 1j * omega * eps, eps, NodalEquation.GAUSS
        )
