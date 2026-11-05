"""Gornergletscher: spatial comparison along the flowline of the provenance-derived erosion (De Doncker et al. 2026, weighted median map),
the observed and modelled velocity, the modelled erosion, and the model's trapping nodes."""
import numpy as np, matplotlib.pyplot as plt, blatter_flow as bf
a = np.loadtxt(bf.DATA_DIR / 'gorner' / 'gorner_flowline.csv', delimiter=',', skiprows=1)   # distance, bed, surface, velocity, uncertainty, provenance erosion
e_prov = a[:, 5]
x, _, dx, _, b, _, hb, _, dbdx, _, _, _, U, _, _ = bf.bp_solution('load_gorner', bf.load_gorner, c_slide=bf.CS_GORNER, shape=bf.SHAPE_GORNER, nx=bf.NX_GORNER)
bf.use_glacier('load_gorner')
r = bf.steady_routing(np.abs(U[:, 0]), bf.QW_GORNER, dbdx, dx, x, hb, b, d_grain=bf.D_FINE, W=bf.W_GORNER, c_slide=bf.CS_GORNER)
e_mod = r['e_dot'] * 1e3; trap = (r['Q_acc'] > 0) | r['supercooled'].astype(bool)
fig, ax = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
_u = a[:, 4]
ax[0].fill_between(a[:, 0] / 1e3, np.maximum(a[:, 3] - _u, 0), a[:, 3] + _u, color='r', alpha=0.2, lw=0, label='uncertainty of the observed velocity')
ax[0].plot(a[:, 0] / 1e3, a[:, 3], 'r', lw=2, label='observed surface (2018)'); ax[0].plot(x / 1e3, np.abs(U[:, -1]), 'k', label='modelled surface'); ax[0].plot(x / 1e3, np.abs(U[:, 0]), 'b--', label='modelled basal'); ax[0].set_ylabel('velocity (m yr$^{-1}$)'); ax[0].legend(fontsize=8)
ax[1].semilogy(x / 1e3, np.maximum(e_mod, 1e-4), 'k', label='modelled erosion rate'); ax[1].semilogy(a[:, 0] / 1e3, np.maximum(e_prov, 1e-4), 'g', lw=2, label='provenance-derived (weighted median map)'); ax[1].set_ylabel('erosion rate (mm yr$^{-1}$)'); ax[1].legend(fontsize=8)
ax[1].set_xlabel('distance along flowline (km)')
for i in np.where(trap)[0]:
    for a_ in ax: a_.axvline(x[i] / 1e3, color='m', alpha=.35, lw=1)
for a_ in ax: a_.grid(alpha=.3)
plt.tight_layout(); plt.savefig(bf.FIG_DIR / 'gorner_spatial.png', dpi=150, bbox_inches='tight'); plt.savefig(bf.FIG_DIR / 'gorner_spatial.pdf', bbox_inches='tight'); print('saved; trapping nodes at km', np.round(x[trap] / 1e3, 1))
