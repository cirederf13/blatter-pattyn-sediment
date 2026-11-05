"""Calibrate K_g (m^(1-l) yr^(l-1)) of each glacier for l = 1, 2 and 3, so that the steady fluvial export at 63 um equals the observed
suspended-sediment discharge: Haut Glacier d'Arolla 5e7 kg/yr (Swift et al. 2002), Franz Josef Glacier 7.7e8 (Herman et al. 2015),
Gornergletscher 1e8 (De Doncker and Herman 2026).  The l = 2 values are K_G_BY_LOADER in blatter_flow.py."""
import numpy as np
from scipy.optimize import brentq
import blatter_flow as bf

D = bf.DATA_DIR / 'franzjosef' / 'herman2015'
w = np.loadtxt(D / 'glacierwidth.txt', delimiter=','); dl = np.loadtxt(D / 'glacierwidthLENGTH.txt', delimiter=',')
ln = np.loadtxt(D / 'ln_edot.txt', delimiter=','); sg = np.loadtxt(D / 'sigma_ln_edot.txt', delimiter=',')
edges = np.arange(0, 10.5e3 + 1, 700.0)
wb = np.array([w[(dl >= edges[i]) & (dl < edges[i + 1])].mean() for i in range(15)])
ok = ~np.isnan(ln) & (wb > 1)
FJ_EXPORT = bf.RHO_R * np.sum((np.exp(ln + sg**2 / 2) * wb * 700.0)[ok])      # integral of the mapped erosion profile

GLACIERS = [("Haut Glacier d'Arolla", 'load_arolla', bf.load_arolla, bf.CS_AROLLA, bf.SHAPE_AROLLA, bf.NX_AROLLA, bf.QW_REF, bf.W_AROLLA, 5e7),
            ('Franz Josef Glacier', 'load_franzjosef', bf.load_franzjosef, bf.CS_FJ, bf.SHAPE_FJ, bf.NX_FJ, bf.QW_FJ, bf.W_FJ, FJ_EXPORT),
            ('Gornergletscher', 'load_gorner', bf.load_gorner, bf.CS_GORNER, bf.SHAPE_GORNER, bf.NX_GORNER, bf.QW_GORNER, bf.W_GORNER, 1e8)]

for name, key, loader, Cs, f, nx, Qw, W, obs in GLACIERS:
    x, _, dx, _, b, _, hb, _, dbdx, _, _, _, U, _, _ = bf.bp_solution(key, loader, c_slide=Cs, shape=f, nx=nx)
    ub = np.abs(U[:, 0])

    def misfit(logK, l, W):
        bf.use_glacier(key); bf.K_G, bf.L_EXP = np.exp(logK), l
        r = bf.steady_routing(ub, Qw, dbdx, dx, x, hb, b, d_grain=bf.D_FINE, W=W, c_slide=Cs)
        return bf.budget_from_routing(r, dx)['Q_fl'] - obs

    for l in (1.0, 2.0, 3.0):
        if key == 'load_franzjosef':        # erosion-weighted width of the measured width profile
            e = ub**l; W = np.sum(e * np.interp(1.2e3 + (x[-1] - x), dl, w)) / np.sum(e)
        K = np.exp(brentq(misfit, np.log(1e-14), np.log(1e-1), args=(l, W), xtol=1e-4))
        print(f'{name}: l = {l:.0f}, K_g = {K:.3e}', flush=True)
