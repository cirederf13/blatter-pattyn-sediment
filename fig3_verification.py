"""Fig. 3: solver verification (slab, H/dx criterion) and observed vs modelled velocities."""
import numpy as np, matplotlib.pyplot as plt, time
import blatter_flow as bf

def slab(L, H, sin_a=0.2, maxiter=2000):
    x = np.linspace(0, L, bf.NX); b = -x * sin_a; h = H + b
    xs, sig, dx, ds, b, h, hb, dhdx, dbdx, C, d, Z = bf.setup_geometry(x, b, h)
    old = bf.MAXITER; bf.MAXITER = maxiter
    try:
        U, S, T = bf.shoot(xs, dx, ds, hb, dhdx, d, C, verbose=False); ok = True
    except RuntimeError:
        U = None; ok = False
    finally:
        bf.MAXITER = old
    return xs, dx, hb, U, ok

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
# (a) slab: numerical vs analytical Weertman slab, H/dx = 1
ax = axes[0, 0]
xs, dx, hb, U, ok = slab(15000.0, 200.0)
# Three references for the slab: (i) the exact Stokes solution, (ii) the
# closed-form solution of the first-order equations on a uniform slab,
#   u_b = C_s (rho g H tan a)^m,   u_s - u_b = 2A/(n+1) (rho g H tan a)^n H / (1 + 4 tan^2 a)^2,
# which the solver must reproduce, and the difference between (i) and (ii),
# which is the error of the first-order approximation at this slope.
al = np.arctan(0.2); Hp = 200.0 * np.cos(al); tb = bf.RHO_G * Hp * np.sin(al)
tT = bf.RHO_G * 200.0 * 0.2
u_slide_ex = bf.C_SLIDE * tb**bf.M_SLIDE
u_def_ex = 2 * bf.A_RATE / (bf.N_GL + 1) * tb**bf.N_GL * Hp * np.cos(al)
u_slide_fo = bf.C_SLIDE * tT**bf.M_SLIDE
u_def_fo = 2 * bf.A_RATE / (bf.N_GL + 1) * tT**bf.N_GL * 200.0 / (1 + 4 * 0.2**2)**2
ax.plot(xs / 1e3, U[:, 0], 'b-', lw=1.8, label='model basal')
ax.plot(xs / 1e3, U[:, -1], 'k-', lw=1.8, label='model surface')
ax.axhline(u_slide_fo, color='b', ls=':', lw=1.4, label=f'first-order analytic, basal: {u_slide_fo:.0f} m/yr')
ax.axhline(u_slide_fo + u_def_fo, color='k', ls=':', lw=1.4, label=f'first-order analytic, surface: {u_slide_fo+u_def_fo:.0f} m/yr')
ax.axhline(u_slide_ex, color='b', ls='--', lw=1.0, alpha=.6, label=f'exact Stokes, basal: {u_slide_ex:.0f} m/yr')
ax.axhline(u_slide_ex + u_def_ex, color='k', ls='--', lw=1.0, alpha=.6, label=f'exact Stokes, surface: {u_slide_ex+u_def_ex:.0f} m/yr')
i = bf.NX // 2
print(f'slab: model u_b {U[i,0]:.1f} vs first-order {u_slide_fo:.1f}; model u_s {U[i,-1]:.1f} vs first-order {u_slide_fo+u_def_fo:.1f}; exact Stokes {u_slide_ex:.0f} / {u_slide_ex+u_def_ex:.0f}')
ax.set_xlabel('distance (km)'); ax.set_ylabel('velocity (m/yr)')
ax.set_title(f'Uniform slab, 11.5°, H = 200 m, H/Δx = {200/dx:.1f}', fontsize=11); ax.legend(fontsize=8); ax.grid(alpha=.3)
# (b) H/dx criterion
ax = axes[0, 1]
res = []
for L, H in ((15000, 100), (15000, 200), (15000, 300), (15000, 400)):
    xs, dx, hb, U, ok = slab(L, H, maxiter=1500)
    res.append((H / dx, ok))
    print(f'  H/dx = {H/dx:.2f}: {"converged" if ok else "no convergence"}')
# the slab no longer fails (zero-gradient ends); the resolution limit shows on real geometry:
# Haut Glacier d'Arolla re-gridded from 49 to 79 nodes (H/dx = 3.4) does not converge
for L, H in ((3800, 150), (3800, 200)):
    xs, dx, hb, U, ok = slab(L, H, maxiter=1500)
    res.append((H / dx, ok)); print(f'  slab 3.8 km, H/dx = {H/dx:.2f}: {"converged" if ok else "no convergence"}')
xx, bb, hh = bf.load_arolla()
xg, _, dxg, dsg, _, _, hbg, dhdxg, _, Cg, dg, _ = bf.setup_geometry(xx, bb, hh, nx=79)
old = bf.MAXITER; bf.MAXITER = 1500
try:
    bf.shoot(xg, dxg, dsg, hbg, dhdxg, dg, Cg, c_slide=bf.CS_AROLLA, verbose=False); ok_ar = True
except RuntimeError:
    ok_ar = False
bf.MAXITER = old
res.append((hbg.max() / dxg, ok_ar)); print(f'  Arolla, 79 nodes, H/dx = {hbg.max()/dxg:.2f}: {"converged" if ok_ar else "no convergence"}')
for r, ok in res:
    ax.plot(r, 1 if ok else 0, 'o', ms=10, color='g' if ok else 'r', mec='k')
ax.axvline(2.0, color='0.4', ls=':', lw=1.2)
for k, (name, Hdx, c) in enumerate((('Arolla', 164/79.2, '#1b7837'), ('Gorner', 329/152, '#b2182b'), ('Franz Josef', 311/137, '#2166ac'))):
    ax.axvline(Hdx, color=c, lw=1.5, alpha=.7); ax.text(Hdx + 0.06, 0.85 - 0.22 * k, f'{name} ({Hdx:.2f})', color=c, fontsize=9, ha='left')
ax.plot([], [], 'o', color='g', mec='k', label='uniform slabs'); ax.plot([], [], 'o', color='r', mec='k', label="Haut Glacier d'Arolla on a finer grid (79 nodes)"); ax.legend(fontsize=8, loc='lower left')
ax.set_yticks([0, 1]); ax.set_yticklabels(['fails', 'converges']); ax.set_xlabel('$H/\\Delta x$'); ax.set_xlim(0, 4.5); ax.set_ylim(-0.3, 1.3)
ax.set_title('Shooting-method convergence vs aspect ratio', fontsize=11); ax.grid(alpha=.3)
# (c) Franz Josef velocities
ax = axes[1, 0]
x, _, dx, _, b, _, hb, _, _, _, _, _, U, _, _ = bf.bp_solution('load_franzjosef', bf.load_franzjosef, c_slide=bf.CS_FJ, shape=bf.SHAPE_FJ, nx=bf.NX_FJ)
d = (x - x[0])
od, ous = np.loadtxt(bf.DATA_DIR / 'franzjosef' / 'surface_velocity_2013.csv', delimiter=',', skiprows=1).T
pos = lambda dist: (x[-1] + 1.2e3 - dist - x[0]) / 1e3         # distance along the flowline from the head, as in the Franz Josef spatial figure
ax.plot(d / 1e3, U[:, -1], 'k-', lw=1.6, label='model surface'); ax.plot(d / 1e3, U[:, 0], 'b--', lw=1.3, label='model basal')
ax.plot(pos(od), ous, 'ro-', ms=5, lw=1.3, label='observed surface, 2013')
ax.set_xlim(0, 10.9); ax.set_xlabel('distance along flowline (km)'); ax.set_ylabel('velocity (m/yr)'); ax.set_title('Franz Josef Glacier', fontsize=11); ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0.0, 0.92)); ax.grid(alpha=.3)
# (d) Gorner velocities
ax = axes[1, 1]
a = np.loadtxt(bf.DATA_DIR / 'gorner' / 'gorner_flowline.csv', delimiter=',', skiprows=1)
x, _, dx, _, b, _, hb, _, _, _, _, _, U, _, _ = bf.bp_solution('load_gorner', bf.load_gorner, c_slide=bf.CS_GORNER, shape=bf.SHAPE_GORNER, nx=bf.NX_GORNER)
x0, _, _, _, _, _, _, _, _, _, _, _, U0, _, _ = bf.bp_solution('load_gorner', bf.load_gorner, c_slide=5.0, shape=1.0, nx=bf.NX_GORNER)
ax.plot(x0 / 1e3, U0[:, -1], color='0.55', lw=1.3, ls='--', label='model surface, uniform $C_s$')
ax.plot(x / 1e3, U[:, -1], 'k-', lw=1.6, label='model surface, inverted $C_s(x)$, $f(x)$'); ax.plot(x / 1e3, U[:, 0], 'b--', lw=1.3, label='model basal, inverted')
_u = a[:, 4]
ax.fill_between(a[:, 0] / 1e3, np.maximum(a[:, 3] - _u, 0), a[:, 3] + _u, color='r', alpha=0.2, lw=0, label='uncertainty of the observed velocity')
ax.plot(a[:, 0] / 1e3, a[:, 3], 'r-', lw=1.6, label='observed surface, 2018')
axr = ax.twinx(); axr.plot(x / 1e3, bf.CS_GORNER, color='teal', lw=1.0, alpha=.7); axr.set_ylabel('$C_s$ (m yr$^{-1}$ bar$^{-3}$)', color='teal'); axr.tick_params(axis='y', labelcolor='teal'); axr.set_ylim(0, 60)
ax.set_xlabel('distance along flowline (km)'); ax.set_ylabel('velocity (m/yr)'); ax.set_title('Gornergletscher', fontsize=11); ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=.3)
for ax, l in zip(axes.flat, 'abcd'): ax.text(.02, .95, f'({l})', transform=ax.transAxes, fontsize=13, fontweight='bold', va='top')
plt.tight_layout(); plt.savefig(bf.FIG_DIR / 'verification.png', dpi=150, bbox_inches='tight'); plt.savefig(bf.FIG_DIR / 'verification.pdf', bbox_inches='tight'); print('saved verification.png/.pdf')
