"""Table 2: eroded rock split between a fraction f_fine at 63 um and the rest at 10 mm.  Prints the fluvial export as a share of the
erosion, f_fine s_fine + (1 - f_fine) s_coarse, and the factor by which K_g must change to give the same suspended-sediment discharge
(shielding is neglected)."""
import numpy as np
import blatter_flow as bf

GLACIERS = [("Haut Glacier d'Arolla", 'load_arolla', bf.load_arolla, bf.CS_AROLLA, bf.SHAPE_AROLLA, bf.NX_AROLLA, bf.QW_REF, bf.W_AROLLA),
            ('Franz Josef Glacier', 'load_franzjosef', bf.load_franzjosef, bf.CS_FJ, bf.SHAPE_FJ, bf.NX_FJ, bf.QW_FJ, bf.W_FJ),
            ('Gornergletscher', 'load_gorner', bf.load_gorner, bf.CS_GORNER, bf.SHAPE_GORNER, bf.NX_GORNER, bf.QW_GORNER, bf.W_GORNER)]
FRACTIONS = (1.0, 0.7, 0.5)

print(f'{"":24s}' + ''.join(f'  share f={f:<4}' for f in FRACTIONS) + ''.join(f'  K_g ratio f={f:<4}' for f in FRACTIONS))
for name, key, loader, Cs, f, nx, Qw, W in GLACIERS:
    x, _, dx, _, b, _, hb, _, dbdx, _, _, _, U, _, _ = bf.bp_solution(key, loader, c_slide=Cs, shape=f, nx=nx)
    bf.use_glacier(key)
    s = {}
    for label, D in (('fine', bf.D_FINE), ('coarse', bf.D_GRAIN)):
        bud = bf.budget_from_routing(bf.steady_routing(np.abs(U[:, 0]), Qw, dbdx, dx, x, hb, b, d_grain=D, W=W, c_slide=Cs), dx)
        s[label] = bud['Q_fl'] / bud['Q_ero']
    share = [fr * s['fine'] + (1 - fr) * s['coarse'] for fr in FRACTIONS]
    print(f'{name:24s}' + ''.join(f'  {v:12.2f}' for v in share) + ''.join(f'  {share[0] / v:15.2f}' for v in share))
