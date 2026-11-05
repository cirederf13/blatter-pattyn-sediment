"""Franz Josef Glacier: invert C_s(x) and f(x) from the observed 2013 surface speeds (Herman et al. 2015), then regress the mapped
erosion rates on the modelled basal speed.  Writes output/franzjosef_Cs_f.npy; the coefficients used in the paper are in data/inverted/."""
import numpy as np, blatter_flow as bf
od, ous = np.loadtxt(bf.DATA_DIR / 'franzjosef' / 'surface_velocity_2013.csv', delimiter=',', skiprows=1).T    # distance to the sampling station, speed
xx, bb, hh = bf.load_franzjosef(); xend = xx[-1]
x_obs = xend - (od - 1.2e3)                                  # the station is 1.2 km up-glacier of the terminus
order = np.argsort(x_obs)
(Cs, f), mis, sol = bf.invert_sliding(bf.load_franzjosef, x_obs[order], ous[order], nx=bf.NX_FJ, shape=1.0, c0=10.8, n_iter=14, f_min=0.6, verbose=False, hold_outside=True)
print('final misfit factor %.2f; C_s %.1f-%.1f, f %.2f-%.2f' % (mis, Cs.min(), Cs.max(), f.min(), f.max()))
np.save(bf.OUT_DIR / 'franzjosef_Cs_f.npy', np.array([Cs, f]))
x = sol[0]; U = sol[12]; ub = np.abs(U[:, 0]); us = np.abs(U[:, -1])
print('model surface at observed positions:', np.round(np.interp(x_obs[order], x, us)), 'observed:', ous[order])

# regression of the mapped erosion (700 m bins, 2-9 km from the station) on the observed and on the modelled basal speed
D = bf.DATA_DIR / 'franzjosef' / 'herman2015'; ln = np.loadtxt(D / 'ln_edot.txt', delimiter=','); sg = np.loadtxt(D / 'sigma_ln_edot.txt', delimiter=',')
edges = np.arange(0, 10.5e3 + 1, 700.0); mid = 0.5 * (edges[:-1] + edges[1:]); d = 1.2e3 + (xend - x)
ub_bin = np.array([ub[(d >= edges[i]) & (d < edges[i + 1])].mean() if ((d >= edges[i]) & (d < edges[i + 1])).any() else np.nan for i in range(15)])
us_bin = np.interp(mid, od, ous)
for label, speed in (('observed surface speed', us_bin), ('modelled basal speed', ub_bin)):
    ok = ~np.isnan(ln) & ~np.isnan(speed) & (speed > 5) & (mid >= 2e3) & (mid <= 9e3)
    X = np.log(speed[ok]); Y = ln[ok]; w = 1 / np.maximum(sg[ok], 0.05)**2
    A = np.vstack([X, np.ones_like(X)]).T; cov = np.linalg.inv(A.T @ np.diag(w) @ A); be = cov @ A.T @ np.diag(w) @ Y
    print('%s, %d bins: l = %.2f +/- %.2f, K = %.2e' % (label, ok.sum(), be[0], np.sqrt(cov[0, 0]), np.exp(be[1])))
    for l in (1.0, 2.0, 3.0):
        K = np.exp(np.average(Y - l * X, weights=w)); print('  fixed l=%.0f: K=%.2e, rms log misfit %.2f' % (l, K, np.sqrt(np.average((Y - np.log(K) - l * X)**2, weights=w))))
