"""Section 5.3: transfer ratio of each glacier when its discharge is multiplied by 0.03 to 30, at 63 um and 10 mm, with the largest
relative residual of the sediment budget."""
import numpy as np
import blatter_flow as bf

GLACIERS = [("Haut Glacier d'Arolla", 'load_arolla', bf.load_arolla, bf.CS_AROLLA, bf.SHAPE_AROLLA, bf.NX_AROLLA, bf.QW_REF, bf.W_AROLLA),
            ('Franz Josef Glacier', 'load_franzjosef', bf.load_franzjosef, bf.CS_FJ, bf.SHAPE_FJ, bf.NX_FJ, bf.QW_FJ, bf.W_FJ),
            ('Gornergletscher', 'load_gorner', bf.load_gorner, bf.CS_GORNER, bf.SHAPE_GORNER, bf.NX_GORNER, bf.QW_GORNER, bf.W_GORNER)]
FACTORS = (0.03, 0.1, 0.3, 1, 3, 10, 30)

worst = worst_ref = 0.0
print('discharge factor     ' + ''.join(f'{fac:7g}' for fac in FACTORS))
for name, key, loader, Cs, f, nx, Qw, W in GLACIERS:
    x, _, dx, _, b, _, hb, _, dbdx, _, _, _, U, _, _ = bf.bp_solution(key, loader, c_slide=Cs, shape=f, nx=nx)
    bf.use_glacier(key)
    for label, D in (('63 um', bf.D_FINE), ('10 mm', bf.D_GRAIN)):
        T = []
        for fac in FACTORS:
            r = bf.steady_routing(np.abs(U[:, 0]), Qw * fac, dbdx, dx, x, hb, b, d_grain=D, W=W, c_slide=Cs)
            bud = bf.budget_from_routing(r, dx)
            T.append(bud['T'])
            res = bud['Q_fl'] + bud['Q_def'] + bud['Q_fr'] + bud['Q_disp'] + bud['Q_acc'] - bud['Q_ero']
            worst = max(worst, abs(res) / bud['Q_ero'])
            if fac == 1:
                worst_ref = max(worst_ref, abs(res) / bud['Q_ero'])
        print(f'{name[:12]:12s} {label:6s} T  ' + ''.join(f'{v:7.2f}' for v in T), flush=True)
print(f'largest relative budget residual: {worst_ref:.1e} at the observed discharge, {worst:.1e} over the whole sweep')
