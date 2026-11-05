"""
One-at-a-time sensitivity of the sediment-transport parameters (Table 4) over their tested ranges.

For each parameter, glacier and value the script reports
  T63, T10  transfer ratio at 63 um and 10 mm, with K_g at its calibrated value;
  Kg        K_g that reproduces the observed 63 um export, relative to the baseline;
  Da        sediment Damkohler number at the start of the transient experiment, 63 um, 50-yr forcing.
Usage: python sweep_parameters.py     (about 10 minutes on eight cores; writes output/sensitivity.csv)
"""
import csv, inspect, multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import brentq
import blatter_flow as bf

GLACIERS = ['arolla', 'franzjosef', 'gorner']
# (module constant, low, high); the baseline value is in Table 4
PARAMETERS = [('H_TILL_REF', 0.1, 2.0), ('SHIELD_MIN', 0.05, 0.3), ('C_DEF_TILL', 0.1, 5.0), ('H_DEF_MAX', 0.1, 2.0),
              ('N_MANN', 0.02, 0.05), ('THETA_C', 0.03, 0.06), ('H_BUMP', 0.05, 0.2), ('K_SHEET', 0.005, 0.5),
              ('F_W', 0.6, 0.95), ('F_DW', 0.02, 0.5), ('C_T', 7.4e-8, 9.8e-8), ('G_GEO', 0.03, 0.1),
              ('R_PORE', 0.5e-6, 5e-6), ('Z_STAR', 10.0, 200.0), ('D_FRINGE_MAX', 1e-3, 4e-3)]


def set_value(name, value):
    """Set a module constant, including the derived fringe pressure and the default argument f_w."""
    setattr(bf, name, value)
    if name == 'R_PORE':
        bf.P_FRINGE = 2.0 * bf.GAMMA_IW / bf.R_PORE
        bf.T_FRINGE = bf.T_MELT - bf.P_FRINGE * bf.T_MELT / (bf.RHO_ICE * bf.L_FUS)
    if name == 'F_W':
        fn = bf.steady_routing; names = list(inspect.signature(fn).parameters)
        d = list(fn.__defaults__); d[names.index('f_w') - (len(names) - len(d))] = value; fn.__defaults__ = tuple(d)


def glacier_setup(name):
    import transient as tr
    cfg = {'arolla': ('load_arolla', bf.load_arolla, bf.W_AROLLA, bf.CS_AROLLA, bf.SHAPE_AROLLA, tr.BETA_ALPS, bf.QW_REF, bf.NX_AROLLA, 5e7),
           'franzjosef': ('load_franzjosef', bf.load_franzjosef, bf.W_FJ, bf.CS_FJ, bf.SHAPE_FJ, tr.BETA_FJ, bf.QW_FJ, bf.NX_FJ, None),
           'gorner': ('load_gorner', bf.load_gorner, bf.W_GORNER, bf.CS_GORNER, bf.SHAPE_GORNER, {**tr.BETA_ALPS, 'ela0': 3165.0}, bf.QW_GORNER, bf.NX_GORNER, 1e8)}
    return cfg[name]


def fj_export():
    D = bf.DATA_DIR / 'franzjosef' / 'herman2015'
    w = np.loadtxt(D / 'glacierwidth.txt', delimiter=','); dl = np.loadtxt(D / 'glacierwidthLENGTH.txt', delimiter=',')
    ln = np.loadtxt(D / 'ln_edot.txt', delimiter=','); sg = np.loadtxt(D / 'sigma_ln_edot.txt', delimiter=',')
    edges = np.arange(0, 10.5e3 + 1, 700.0)
    wb = np.array([w[(dl >= edges[i]) & (dl < edges[i + 1])].mean() for i in range(15)])
    ok = ~np.isnan(ln) & (wb > 1)
    return bf.RHO_R * np.sum((np.exp(ln + sg**2 / 2) * wb * 700.0)[ok])


def metrics(task):
    glacier, param, value = task
    import transient as tr, transient_sediment as ts
    key, loader, W, Cs, f, mb, Qw, nx, obs = glacier_setup(glacier)
    obs = fj_export() if obs is None else obs
    if param is not None:
        set_value(param, value)
    x, _, dx, _, b, _, hb, _, dbdx, _, _, _, U, _, _ = bf.bp_solution(key, loader, c_slide=Cs, shape=f, nx=nx)
    ub = np.abs(U[:, 0])

    def budget(D, K=None):
        bf.use_glacier(key)
        if K is not None:
            bf.K_G = K
        return bf.budget_from_routing(bf.steady_routing(ub, Qw, dbdx, dx, x, hb, b, d_grain=D, W=W, c_slide=Cs), dx)
    K0 = bf.K_G_BY_LOADER[key]
    T63, T10 = budget(bf.D_FINE)['T'], budget(bf.D_GRAIN)['T']
    try:
        Kg = np.exp(brentq(lambda lk: budget(bf.D_FINE, np.exp(lk))['Q_fl'] - obs, np.log(K0 / 100), np.log(K0 * 100), xtol=0.01)) / K0
    except ValueError:              # the observed export cannot be reached for this value
        Kg = np.nan
    bf.use_glacier(key)
    g = tr.FlowlineIce(key, loader, W, Cs, f, mb, nx=nx)
    view = ts.RefinedView(g, 32)
    r0 = bf.steady_routing(np.abs(view.U[:, 0]), Qw, view.dbdx, view.dx, view.x, view.hb, view.b, d_grain=bf.D_FINE, W=W, c_slide=view.c_slide)
    st = ts.SedimentState(len(view.x)); st.init_from_steady(r0, view.dx, W)
    res = ts.sediment_step(st, view, Qw, bf.D_FINE, dt=1.0)
    Da = res['V_stor'] * bf.RHO_SED / max(res['Q_out'], 1e-30) / 50.0
    return dict(parameter=param or 'baseline', value=value, glacier=glacier, T63=T63, T10=T10, Kg=Kg, Da=Da)


if __name__ == '__main__':
    tasks = [(g, None, None) for g in GLACIERS] + [(g, p, v) for p, lo, hi in PARAMETERS for v in (lo, hi) for g in GLACIERS]
    with ProcessPoolExecutor(max_workers=7, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as ex:    # a fresh process per task
        rows = list(ex.map(metrics, tasks))
    with open(bf.OUT_DIR / 'sensitivity.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    base = {r['glacier']: r for r in rows if r['parameter'] == 'baseline'}
    print('baseline:  ' + '  '.join(f"{g}: T63 {base[g]['T63']:.2f}, T10 {base[g]['T10']:.2f}, Da {base[g]['Da']:.2f}" for g in GLACIERS))
    print(f'{"parameter":14s}{"value":>10s}' + ''.join(f'  | {g[:6]:>6s} T63  T10  Kg/K0   Da ' for g in GLACIERS))
    for p, lo, hi in PARAMETERS:
        for v in (lo, hi):
            line = f'{p:14s}{v:10.3g}'
            for g in GLACIERS:
                r = next(r for r in rows if r['parameter'] == p and r['value'] == v and r['glacier'] == g)
                line += f"  | {r['T63']:11.2f} {r['T10']:5.2f} {r['Kg']:6.2f} {r['Da']:5.2f}"
            print(line)
