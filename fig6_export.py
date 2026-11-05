"""Fig. 6: modelled erosion and export against the observed suspended-sediment discharge at the three glaciers."""
import numpy as np, matplotlib.pyplot as plt
import blatter_flow as bf

# Franz Josef: observed erosion from the 2015 analysis (suspended load, no bedload factor), integrated over the true glacier width
_D = bf.DATA_DIR / 'franzjosef' / 'herman2015'
_w = np.loadtxt(_D / 'glacierwidth.txt', delimiter=','); _dl = np.loadtxt(_D / 'glacierwidthLENGTH.txt', delimiter=',')
_ln = np.loadtxt(_D / 'ln_edot.txt', delimiter=','); _sg = np.loadtxt(_D / 'sigma_ln_edot.txt', delimiter=',')
_edges = np.arange(0, 10.5e3 + 1, 700.0)
_wb = np.array([_w[(_dl >= _edges[i]) & (_dl < _edges[i + 1])].mean() for i in range(15)])
_ok = ~np.isnan(_ln) & (_wb > 1)
FJ_GEO = bf.RHO_R * np.sum((np.exp(_ln) * _wb * 700.0)[_ok])                       # geometric-mean profile
FJ_ARI = bf.RHO_R * np.sum((np.exp(_ln + _sg**2 / 2) * _wb * 700.0)[_ok])           # arithmetic mean if log-normal
FJ_FULL = 10e-3 * 36.476e6 * bf.RHO_R                                               # 10 mm/yr over 36.476 km2 (full record incl. storms; relativeF.m area)
print(f'Franz Josef observed erosion: geometric profile {FJ_GEO:.2e}, arithmetic profile {FJ_ARI:.2e}, full record {FJ_FULL:.2e} kg/yr')
def w_eff_fj(x, ub, l):
    """Erosion-weighted glacier width (m) of the Franz Josef flowline: sum(e w)/sum(e) with the true width profile."""
    d = 1.2e3 + (x[-1] - x); w = np.interp(d, _dl, _w); e = np.abs(ub)**l
    return np.sum(e * w) / np.sum(e)

G = [("Haut Glacier\nd'Arolla", 'load_arolla', bf.load_arolla, bf.QW_REF, bf.W_AROLLA, bf.CS_AROLLA, bf.SHAPE_AROLLA, bf.NX_AROLLA, 5.0e7, (3.0e7, 8.0e7), 'Swift et al. (2002), 1998'),
     ('Franz Josef\nGlacier', 'load_franzjosef', bf.load_franzjosef, bf.QW_FJ, bf.W_FJ, bf.CS_FJ, bf.SHAPE_FJ, bf.NX_FJ, FJ_ARI, (FJ_GEO, FJ_FULL), 'Herman et al. (2015), 2013-14'),
     ('Gorner-\ngletscher', 'load_gorner', bf.load_gorner, bf.QW_GORNER, bf.W_GORNER, bf.CS_GORNER, bf.SHAPE_GORNER, bf.NX_GORNER, 1.0e8, (0.4e8, 1.7e8), 'De Doncker & Herman (2026), 1973-2020')]
fig, ax = plt.subplots(figsize=(11, 5.5))
w = 0.14; xs = np.arange(3)
for i, (name, key, loader, Qw, W, Cs, f, nx, obs, (lo, hi), src) in enumerate(G):
    x, _, dx, _, b, _, hb, _, dbdx, _, _, _, U, _, Tau = bf.bp_solution(key, loader, c_slide=Cs, shape=f, nx=nx)
    vals = {}
    if key == 'load_franzjosef':
        W = w_eff_fj(x, U[:, 0], bf.L_EXP); print(f'Franz Josef: erosion-weighted width {W:.0f} m (uniform A/L: {bf.W_FJ:.0f} m)')
    for D, lab in ((bf.D_FINE, '63 µm'), (bf.D_GRAIN, '10 mm')):
        r = bf.steady_routing(np.abs(U[:, 0]), Qw, dbdx, dx, x, hb, b, d_grain=D, W=W, c_slide=Cs)
        bud = bf.budget_from_routing(r, dx); vals[lab] = bud
    bud = vals['63 µm']
    ax.bar(i - 2.5 * w, bud['Q_ero'], w, color='0.3', label='modelled erosion (= dark blue + teal + till)' if i == 0 else None)
    ax.bar(i - 1.5 * w, bud['Q_fl'], w, color='#2166ac', label='fluvial export, 63 µm (compare with observation)' if i == 0 else None)
    ax.bar(i - 0.5 * w, bud['Q_fr'] + bud['Q_disp'], w, color='teal', label='basal-ice export, 63 µm' if i == 0 else None)
    ax.bar(i + 0.5 * w, max(bud['Q_acc'], 1.0), w, color='#8c6d31', label='accumulating as till, 63 µm' if i == 0 else None)
    ax.bar(i + 1.5 * w, vals['10 mm']['Q_fl'], w, color='#92c5de', hatch='//', edgecolor='w', label='fluvial export, 10 mm (separate case)' if i == 0 else None)
    ax.errorbar(i + 2.5 * w, obs, yerr=[[obs - lo], [hi - obs]], fmt='o', color='r', ms=9, capsize=5, label='observed suspended export' if i == 0 else None)
    print(f"{name.replace(chr(10),' ')}: erosion {bud['Q_ero']:.2e}, fluvial 63um {bud['Q_fl']:.2e}, 10mm {vals['10 mm']['Q_fl']:.2e}, basal ice {bud['Q_fr']+bud['Q_disp']:.2e}, observed {obs:.1e} ({lo:.1e}-{hi:.1e})")
ax.set_yscale('log'); ax.set_ylim(1e5, 3e9); ax.set_xticks(xs); ax.set_xticklabels([g[0] + '\n' + g[10] for g in G], fontsize=9.5)
ax.set_ylabel('sediment flux (kg yr$^{-1}$)'); ax.grid(alpha=.3, axis='y'); ax.legend(fontsize=8, loc='upper left', ncol=2)
plt.tight_layout(); plt.savefig(bf.FIG_DIR / 'export_comparison.png', dpi=150, bbox_inches='tight'); plt.savefig(bf.FIG_DIR / 'export_comparison.pdf', bbox_inches='tight'); print('saved export_comparison.png/.pdf')
