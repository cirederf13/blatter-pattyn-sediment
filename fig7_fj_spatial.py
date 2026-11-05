"""Franz Josef Glacier: spatial comparison along the flowline of the mapped erosion rate (Herman et al. 2015, 700 m bins of distance to the
sampling station), the observed 2013 surface speed, and the modelled speed and erosion (model with C_s(x), f(x) inverted from the observed
speed, l = 2), with the modelled trapping nodes."""
import numpy as np, matplotlib.pyplot as plt, blatter_flow as bf
od, ous = np.loadtxt(bf.DATA_DIR / 'franzjosef' / 'surface_velocity_2013.csv', delimiter=',', skiprows=1).T
xx, bb, hh = bf.load_franzjosef(); xend = xx[-1]
D = bf.DATA_DIR / 'franzjosef' / 'herman2015'
ln = np.loadtxt(D / 'ln_edot.txt', delimiter=','); sg = np.loadtxt(D / 'sigma_ln_edot.txt', delimiter=',')
edges = np.arange(0, 10.5e3 + 1, 700.0); mid = 0.5 * (edges[:-1] + edges[1:])            # distance to the station (m)
bf.use_glacier('load_franzjosef')
def run(Cs, f, l, K):
    bf.use_glacier('load_franzjosef'); bf.L_EXP, bf.K_G = l, K
    x, _, dx, _, b, _, hb, _, dbdx, _, _, _, U, _, _ = bf.bp_solution('load_franzjosef', bf.load_franzjosef, c_slide=Cs, shape=f, nx=bf.NX_FJ)
    bf.L_EXP, bf.K_G = l, K
    r = bf.steady_routing(np.abs(U[:, 0]), bf.QW_FJ, dbdx, dx, x, hb, b, d_grain=bf.D_FINE, W=bf.W_FJ, c_slide=Cs)
    return x, U, r
x, U, r = run(bf.CS_FJ, bf.SHAPE_FJ, 2.0, bf.K_G_BY_LOADER['load_franzjosef'])                # adopted model
d = 1.2e3 + (xend - x)
pos = lambda dist: (xend + 1.2e3 - dist - x[0]) / 1e3          # km along the flowline from the head (flow left to right, as at Gornergletscher)
def binned(a, dd): return np.array([a[(dd >= edges[i]) & (dd < edges[i + 1])].mean() if ((dd >= edges[i]) & (dd < edges[i + 1])).any() else np.nan for i in range(len(mid))])
fig, ax = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
ax[0].plot(pos(od), ous, 'ro', label='observed surface (2013)')
ax[0].plot(pos(d), np.abs(U[:, -1]), 'k', label='model surface'); ax[0].plot(pos(d), np.abs(U[:, 0]), 'b--', label='model basal')
ax[0].set_ylabel('velocity (m yr$^{-1}$)'); ax[0].legend(fontsize=8)
e_obs = np.exp(ln) * 1e3; lo = np.exp(ln - sg) * 1e3; hi = np.exp(ln + sg) * 1e3
ax[1].errorbar(pos(mid), e_obs, yerr=[e_obs - lo, hi - e_obs], fmt='go', capsize=3, label='mapped erosion (Herman et al. 2015)')
ax[1].semilogy(pos(d), np.maximum(r['e_dot'] * 1e3, 1e-3), 'k', label='modelled erosion, l = 2 (squares: 700 m bins)')
inside = mid > 1.2e3                                              # bins downstream of the terminus are not on the glacier
ax[1].plot(pos(mid[inside]), binned(r['e_dot'] * 1e3, d)[inside], 'ks', ms=5, mfc='none')
ax[1].set_yscale('log'); ax[1].set_ylabel('erosion rate (mm yr$^{-1}$)'); ax[1].legend(fontsize=8)
ax[1].set_xlabel('distance along flowline (km)')
trap = (r['Q_acc'] > 0) | r['supercooled'].astype(bool)
for i in np.where(trap)[0]:
    for a_ in ax: a_.axvline(pos(d[i]), color='m', alpha=.35, lw=1)
for a_ in ax: a_.grid(alpha=.3)
plt.tight_layout(); plt.savefig(bf.FIG_DIR / 'fj_spatial.png', dpi=150, bbox_inches='tight'); plt.savefig(bf.FIG_DIR / 'fj_spatial.pdf', bbox_inches='tight')
print('saved; trapping nodes at km:', np.round(pos(d[trap]), 1))
