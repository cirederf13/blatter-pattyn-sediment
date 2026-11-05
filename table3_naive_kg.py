"""K_g inferred without the coupled model (observed export / (rho_R area), divided by mean velocity^l) against the K_g fitted with the full model."""
import numpy as np, blatter_flow as bf
G = [('Haut Glacier d\'Arolla', 'load_arolla', bf.load_arolla, bf.CS_AROLLA, bf.SHAPE_AROLLA, bf.NX_AROLLA, 5e7, 6.3e6, {1.0: 3.69e-4, 2.0: 3.14e-5, 3.0: 2.47e-6}),
     ('Franz Josef Glacier', 'load_franzjosef', bf.load_franzjosef, bf.CS_FJ, bf.SHAPE_FJ, bf.NX_FJ, 7.7e8, 35e6, {1.0: 1.02e-4, 2.0: 5.93e-7, 3.0: 2.285e-9}),
     ('Gornergletscher', 'load_gorner', bf.load_gorner, bf.CS_GORNER, bf.SHAPE_GORNER, bf.NX_GORNER, 1e8, 41e6, {1.0: 1.66e-5, 2.0: 1.13e-7, 3.0: 7.54e-10})]
for n, k, ld, Cs, f, nx, Q, A, Kfull in G:
    x, _, dx, _, b, _, hb, _, dbdx, _, _, _, U, _, _ = bf.bp_solution(k, ld, c_slide=Cs, shape=f, nx=nx); ice = hb > 1
    ub, us = np.abs(U[ice, 0]), np.abs(U[ice, -1]); e = Q / (bf.RHO_R * A)
    print(f'{n}: erosion from export/area {e*1e3:.2f} mm/yr, mean basal {ub.mean():.0f} and surface {us.mean():.0f} m/yr')
    for l in (1.0, 2.0, 3.0):
        print(f'  l={l:.0f}: K from export, area and mean basal velocity {e/ub.mean()**l:.2e} ({e/ub.mean()**l/Kfull[l]:.2f} of the full-model {Kfull[l]:.2e}); from mean surface velocity {e/us.mean()**l:.2e} ({e/us.mean()**l/Kfull[l]:.2f})')
