import unittest
from math import factorial

import numpy as np

from darwin.elements import GaussianQuadrature


def _tetrahedron_integral(exponents: tuple[int, int, int]) -> float:
    """Exact ``integral of x**a y**b z**c`` over the reference tetrahedron."""
    a, b, c = exponents
    return factorial(a) * factorial(b) * factorial(c) / factorial(a + b + c + 3)


class TestReferenceQuadrature(unittest.TestCase):
    def test_builtin_tetrahedron_rules_have_the_right_mass(self):
        for degree in (1, 2):
            quadrature = GaussianQuadrature(3, degree)
            self.assertAlmostEqual(float(np.sum(quadrature.weights)), 1.0 / 6.0)

    def test_duffy_tetrahedron_rule_integrates_monomials(self):
        quadrature = GaussianQuadrature(3, 5)

        for exponents in (
            (0, 0, 0),
            (1, 0, 0),
            (0, 2, 0),
            (2, 1, 0),
            (4, 0, 0),
            (2, 2, 0),
            (2, 1, 1),
            (1, 1, 1),
        ):
            powers = np.prod(quadrature.points ** np.asarray(exponents), axis=1)
            value = float(np.sum(quadrature.weights * powers))
            self.assertAlmostEqual(
                value, _tetrahedron_integral(exponents), places=12, msg=str(exponents)
            )

    def test_duffy_triangle_rule_integrates_monomials(self):
        quadrature = GaussianQuadrature(2, 4)

        for exponents in ((0, 0), (1, 0), (2, 0), (3, 0), (2, 2)):
            a, b = exponents
            exact = factorial(a) * factorial(b) / factorial(a + b + 2)
            powers = np.prod(quadrature.points ** np.asarray(exponents), axis=1)
            value = float(np.sum(quadrature.weights * powers))
            self.assertAlmostEqual(value, exact, places=12, msg=str(exponents))

    def test_duffy_tetrahedron_rule_points_lie_in_the_reference_cell(self):
        quadrature = GaussianQuadrature(3, 3)
        points = quadrature.points

        self.assertTrue(np.all(points >= -1e-15))
        self.assertTrue(np.all(points.sum(axis=1) <= 1.0 + 1e-15))


if __name__ == "__main__":
    unittest.main()
