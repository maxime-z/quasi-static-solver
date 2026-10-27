from darwin.elements import GaussianQuadrature


gauss_points_2d_order_2 = GaussianQuadrature(2, 2)
for i_gp, wi, (xi, eta) in gauss_points_2d_order_2:
    print(f"Gauss point {i_gp}: {xi}, {eta}, weight: {wi}")



gauss_points_3d_order_2 = GaussianQuadrature(3, 2)
for i_gp, wi, (xi, eta, zeta) in gauss_points_3d_order_2:
    print(f"Gauss point {i_gp}: {xi}, {eta}, {zeta}, weight: {wi}")