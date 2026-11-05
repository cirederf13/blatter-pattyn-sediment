"""
Monte Carlo sensitivity of the sediment and hydrology model.

All 15 parameters of sweep_parameters.py are drawn together by Latin hypercube sampling over their tested ranges (log-uniform for the
scale parameters).  For each sample and glacier, K_g is recalibrated to the observed 63 um export, and the transfer ratio at 63 um and
10 mm, the calibrated K_g relative to the baseline and the sediment Damkohler number at the start of the transient experiment are recorded.

    python monte_carlo.py run [n=200]        steady-state samples, writes output/monte_carlo.csv   (about 30 min on eight cores)
    python monte_carlo.py transient [n=30]   transient runs of Gornergletscher for the first n samples (about 1 h on eight cores)
    python monte_carlo.py control [n=30]     the same runs without forcing (dE = 0), to separate the initial adjustment from the pulse
    python monte_carlo.py summary            prints the percentiles and rank correlations, writes figures/monte_carlo*.png
"""
import csv, multiprocessing, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import brentq
from scipy.stats import qmc, spearmanr
import blatter_flow as bf
from sweep_parameters import PARAMETERS, GLACIERS, set_value, glacier_setup, fj_export

LOG = {'H_TILL_REF', 'C_DEF_TILL', 'H_DEF_MAX', 'K_SHEET', 'F_DW', 'R_PORE', 'Z_STAR'}
CSV = bf.OUT_DIR / 'monte_carlo.csv'
METRICS = ('T63', 'T10', 'Kg', 'Da')


def draw(n, seed=1):
    """Latin hypercube sample: array (n, number of parameters) of parameter values."""
    u = qmc.LatinHypercube(d=len(PARAMETERS), seed=seed).random(n)
    out = np.empty_like(u)
    for j, (name, lo, hi) in enumerate(PARAMETERS):
        out[:, j] = np.exp(np.log(lo) + u[:, j] * (np.log(hi) - np.log(lo))) if name in LOG else lo + u[:, j] * (hi - lo)
    return out


def run_sample(task):
    i, glacier, values = task
    import transient as tr, transient_sediment as ts
    for (name, _, _), v in zip(PARAMETERS, values):
        set_value(name, v)
    key, loader, W, Cs, f, mb, Qw, nx, obs = glacier_setup(glacier)
    obs = fj_export() if obs is None else obs
    x, _, dx, _, b, _, hb, _, dbdx, _, _, _, U, _, _ = bf.bp_solution(key, loader, c_slide=Cs, shape=f, nx=nx)
    ub = np.abs(U[:, 0]); K0 = bf.K_G_BY_LOADER[key]

    def budget(D, K):
        bf.use_glacier(key); bf.K_G = K
        return bf.budget_from_routing(bf.steady_routing(ub, Qw, dbdx, dx, x, hb, b, d_grain=D, W=W, c_slide=Cs), dx)
    out = dict(sample=i, glacier=glacier, Kcal=np.nan, T63=np.nan, T10=np.nan, Kg=np.nan, Da=np.nan)
    try:
        K = np.exp(brentq(lambda lk: budget(bf.D_FINE, np.exp(lk))['Q_fl'] - obs, np.log(K0 / 100), np.log(K0 * 100), xtol=0.01))
    except ValueError:                  # the observed export is not reached for this parameter set
        return out
    bf.K_G_BY_LOADER[key] = K
    bf.use_glacier(key)
    g = tr.FlowlineIce(key, loader, W, Cs, f, mb, nx=nx)
    view = ts.RefinedView(g, 32)
    r0 = bf.steady_routing(np.abs(view.U[:, 0]), Qw, view.dbdx, view.dx, view.x, view.hb, view.b, d_grain=bf.D_FINE, W=W, c_slide=view.c_slide)
    st = ts.SedimentState(len(view.x)); st.init_from_steady(r0, view.dx, W)
    res = ts.sediment_step(st, view, Qw, bf.D_FINE, dt=1.0)
    out.update(Kcal=K, T63=budget(bf.D_FINE, K)['T'], T10=budget(bf.D_GRAIN, K)['T'], Kg=K / K0,
               Da=res['V_stor'] * bf.RHO_SED / max(res['Q_out'], 1e-30) / 50.0)
    return out


def run_transient(task):
    i, values, K, dE, tag = task
    import run_transient as rt
    for (name, _, _), v in zip(PARAMETERS, values):
        set_value(name, v)
    bf.K_G_BY_LOADER['load_gorner'] = K
    rt.main('gorner', dE_total=dE, tag=f'_mc{i}{tag}')


def pool(n_workers=7):
    return ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1)


def read():
    rows = list(csv.DictReader(open(CSV)))
    return [{k: (v if k == 'glacier' else float(v)) for k, v in r.items()} for r in rows]


def summary():
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams['font.size'] = 10
    rows = read(); n = len(PARAMETERS)
    values = {int(r['sample']): [r[name] for name, _, _ in PARAMETERS] for r in rows if r['glacier'] == GLACIERS[0]}
    ref = {(r['glacier'], m): float(r[m]) for r in csv.DictReader(open(bf.OUT_DIR / 'sensitivity.csv')) if r['parameter'] == 'baseline' for m in METRICS}
    names = {'arolla': "Haut Glacier d'Arolla", 'franzjosef': 'Franz Josef Glacier', 'gorner': 'Gornergletscher'}
    titles = {'T63': '$\\mathcal{T}$, 63 µm', 'T10': '$\\mathcal{T}$, 10 mm', 'Kg': '$K_g$ / reference', 'Da': '$\\mathrm{Da_{sed}}$'}
    fig, axes = plt.subplots(3, 4, figsize=(11, 6.5))
    rho = np.full((n, 3 * len(GLACIERS)), np.nan)
    print(f'{"":12s}{"reached":>8s}' + ''.join(f'{m:>20s}' for m in METRICS) + '   (5th, 50th, 95th percentile)')
    for gi, g in enumerate(GLACIERS):
        rr = [r for r in rows if r['glacier'] == g]; ok = [r for r in rr if not np.isnan(r['T63'])]
        pars = np.array([values[int(r['sample'])] for r in ok])
        line = f'{g:12s}{len(ok):4d}/{len(rr):<3d}'
        for mi, m in enumerate(METRICS):
            v = np.array([r[m] for r in ok]); p = np.percentile(v, [5, 50, 95])
            line += f'  {p[0]:5.2f} {p[1]:5.2f} {p[2]:5.2f}   '
            ax = axes[gi, mi]
            if np.ptp(v) < 1e-3:
                ax.text(0.5, 0.5, f'{v.mean():.2f} in all samples', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                if m in ('T63', 'T10'):
                    ax.hist(v, bins=np.linspace(0, 1, 21), color='0.6'); ax.set_xlim(0, 1); r0 = ref[(g, m)]
                else:
                    ax.hist(np.log10(v), bins=20, color='0.6'); r0 = np.log10(ref[(g, m)]) if m == 'Da' else 0.0
                    lo, hi = np.log10(v.min()), np.log10(v.max())
                    if m == 'Kg':
                        span = hi - lo
                        cands = [0.5, 1, 2, 5, 10, 30] if span > 0.6 else [0.7, 0.8, 1, 1.25, 1.5, 2] if span > 0.2 else [0.9, 0.95, 1, 1.05, 1.1, 1.15]
                    else:
                        cands = [0.01, 0.1, 1, 10, 100, 1000]
                        if sum(lo - 0.02 <= np.log10(c) <= hi + 0.02 for c in cands) < 2:
                            cands = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
                    tk = [c for c in cands if lo - 0.02 <= np.log10(c) <= hi + 0.02]
                    if len(tk) > 4:
                        tk = [tk[k] for k in np.unique(np.linspace(0, len(tk) - 1, 4).round().astype(int))]
                    ax.set_xticks(np.log10(tk)); ax.set_xticklabels([f'{c:g}' for c in tk])
                ax.axvline(r0, color='r', lw=1.3)
                if m == 'Da':
                    ax.axvline(0.0, color='k', ls=':', lw=1)
                if m in ('T63', 'T10'):
                    ax.xaxis.set_major_locator(MaxNLocator(4))
                ax.set_yticks([])
            if gi == 0:
                ax.set_title(titles[m], fontsize=10)
            if mi == 0:
                ax.set_ylabel(names[g], fontsize=10)
        print(line)
        for k, m in enumerate(('T63', 'Kg', 'Da')):
            v = np.array([r[m] for r in ok])
            for j in range(n):
                rho[j, 3 * gi + k] = spearmanr(pars[:, j], v)[0]
    plt.tight_layout(); plt.savefig(bf.FIG_DIR / 'monte_carlo.png', dpi=200, bbox_inches='tight'); plt.savefig(bf.FIG_DIR / 'monte_carlo.pdf', bbox_inches='tight'); plt.close()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(rho, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_yticks(range(n)); ax.set_yticklabels([p[0] for p in PARAMETERS])
    ax.set_xticks(range(9)); ax.set_xticklabels([f'{g[:6]}\n{m}' for g in GLACIERS for m in ('T63', 'Kg', 'Da')], fontsize=8)
    plt.colorbar(im, label='Spearman rank correlation'); plt.tight_layout()
    plt.savefig(bf.FIG_DIR / 'monte_carlo_rank.png', dpi=200, bbox_inches='tight'); plt.close()
    print('\nlargest rank correlations (|rho| > 0.3):')
    for gi, g in enumerate(GLACIERS):
        for k, m in enumerate(('T63', 'Kg', 'Da')):
            top = sorted(((abs(rho[j, 3 * gi + k]), PARAMETERS[j][0], rho[j, 3 * gi + k]) for j in range(n)), reverse=True)[:4]
            print(f'  {g:11s} {m:4s} ' + ', '.join(f'{name} {r:+.2f}' for a, name, r in top if a > 0.3))


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'summary'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else (200 if mode == 'run' else 30)
    if mode == 'run':
        samples = draw(n)
        tasks = [(i, g, samples[i]) for i in range(n) for g in GLACIERS]
        with pool() as ex:
            rows = list(ex.map(run_sample, tasks))
        with open(CSV, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['sample', 'glacier', 'Kcal', *METRICS] + [p[0] for p in PARAMETERS]); w.writeheader()
            for r in rows:
                w.writerow({**r, **{p[0]: v for p, v in zip(PARAMETERS, samples[r['sample']])}})
    elif mode in ('transient', 'control'):
        dE, tag = (150.0, '') if mode == 'transient' else (0.0, '_ctrl')
        rows = [r for r in read() if r['glacier'] == 'gorner' and not np.isnan(r['Kcal'])][:n]
        rows = [r for r in rows if not (bf.OUT_DIR / f"transient_gorner_mc{int(r['sample'])}{tag}.npz").exists()]      # skip finished runs
        with pool() as ex:
            list(ex.map(run_transient, [(int(r['sample']), [r[p[0]] for p in PARAMETERS], r['Kcal'], dE, tag) for r in rows]))
    else:
        summary()
