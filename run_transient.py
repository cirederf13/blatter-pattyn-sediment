"""
ELA-forced transient for one glacier (ice, sediment reservoirs and a fresh/stored tracer).

    python run_transient.py arolla|franzjosef|gorner [options]

Writes output/transient_<glacier><tag>.npz with annual time series.  Options:
  --dE         rise of the equilibrium-line altitude (m, default 150; 0 gives the unforced control)
  --ramp       duration of the rise (yr, default 50)
  --end        end of the run (yr, default 100)
  --exponent   erosion exponent l = 1 or 3 (K_g recalibrated, so that the initial erosion is unchanged)
  --ratio      accumulation/ablation gradient ratio of the mass balance
  --fine-um    grain size of the fine class (um, default 63)
  --refine     refinement of the sediment grid over the ice grid (default 32)
  --tag        suffix of the output file
"""
import argparse, time
import numpy as np
import blatter_flow as bf
import transient as tr
import transient_sediment as ts

CFG = dict(
    arolla=('load_arolla', bf.load_arolla, bf.W_AROLLA, bf.CS_AROLLA, bf.SHAPE_AROLLA, tr.BETA_ALPS, bf.QW_REF, bf.NX_AROLLA),
    franzjosef=('load_franzjosef', bf.load_franzjosef, bf.W_FJ, bf.CS_FJ, bf.SHAPE_FJ, tr.BETA_FJ, bf.QW_FJ, bf.NX_FJ),
    gorner=('load_gorner', bf.load_gorner, bf.W_GORNER, bf.CS_GORNER, bf.SHAPE_GORNER, {**tr.BETA_ALPS, 'ela0': 3165.0}, bf.QW_GORNER, bf.NX_GORNER),
)
# K_g (m^(1-l) yr^(l-1)) calibrated to the observed export for l = 1 and 3 (calibrate_kg.py)
K_FOR_L = dict(load_arolla={1.0: 3.69e-4, 3.0: 2.47e-6}, load_franzjosef={1.0: 1.02e-4, 3.0: 2.285e-9},
               load_gorner={1.0: 1.66e-5, 3.0: 7.54e-10})
SERIES = ('Q_ero', 'Q_fl', 'Q_ice', 'Q_out', 'f_stor', 's_fl', 'V_stor', 'M_stored', 'exposed')


def main(name, dE_total=150.0, t_ramp=50, t_end=100, exponent=None, ratio=None, fine_um=63.0, refine=32, tag=''):
    key, loader, W, Cs, f, mb, Qw0, nx = CFG[name]
    grains = dict(fine=fine_um * 1e-6, coarse=bf.D_GRAIN)
    if ratio is not None:
        mb = {**mb, 'beta_acc': mb['beta_abl'] * ratio}
    if exponent is not None:
        bf.L_BY_LOADER[key] = exponent; bf.K_G_BY_LOADER[key] = K_FOR_L[key][exponent]
    bf.use_glacier(key)
    g = tr.FlowlineIce(key, loader, W, Cs, f, mb, nx=nx)
    view = ts.RefinedView(g, refine)
    melt0 = g.ablation_melt(0.0)
    print(f'{name}: beta_abl {g.mb["beta_abl"]:.4f} beta_acc {g.mb["beta_acc"]:.4f} ELA0 {g.ELA0:.0f}', flush=True)

    states = {}
    for gl, D in grains.items():
        r0 = bf.steady_routing(np.abs(view.U[:, 0]), Qw0, view.dbdx, view.dx, view.x, view.hb, view.b,
                               d_grain=D, W=W, c_slide=view.c_slide)
        states[gl] = ts.SedimentState(len(view.x)); states[gl].init_from_steady(r0, view.dx, W)

    out = {k: [] for k in ('t', 'dE', 'V', 'L', 'Qw', 'melt')}
    snap = {k: [] for k in ('hb', 'u_b', 'e_dot', 'N', 'supercooled')}
    for gl in grains:
        out.update({f'{gl}_{k}': [] for k in SERIES})
        snap.update({f'{gl}_{k}': [] for k in ('M_till', 'M_ice', 's_till', 's_ice')})
    t0 = time.time()
    for yr in range(t_end + 1):
        dE = dE_total * min(yr, t_ramp) / t_ramp
        Qw = Qw0 * g.ablation_melt(dE) / melt0            # discharge scales with the melt of the ablation area
        for gl, D in grains.items():
            view = ts.RefinedView(g, refine)
            res = ts.sediment_step(states[gl], view, Qw, D, dt=1.0)
            for k in SERIES:
                out[f'{gl}_{k}'].append(res[k])
        out['t'].append(yr); out['dE'].append(dE); out['V'].append(g.volume()); out['L'].append(g.length())
        out['Qw'].append(Qw); out['melt'].append(g.ablation_melt(dE))
        snap['hb'].append(g.hb.copy()); snap['u_b'].append(np.abs(g.U[:, 0])); snap['e_dot'].append(res['routing']['e_dot'][::refine])
        snap['N'].append(res['routing']['N_fringe'][::refine]); snap['supercooled'].append(res['routing']['supercooled'][::refine])
        for gl in grains:
            st = states[gl]; Mi = st.M_f + st.M_d
            snap[f'{gl}_M_till'].append(view.to_coarse(st.M_till, mass=True)); snap[f'{gl}_M_ice'].append(view.to_coarse(Mi, mass=True))
            snap[f'{gl}_s_till'].append(view.to_coarse(st.s_till, weight=st.M_till))
            snap[f'{gl}_s_ice'].append(view.to_coarse((st.M_f * st.s_f + st.M_d * st.s_d) / np.maximum(Mi, 1e-30), weight=Mi))
        if yr % 10 == 0:
            print(f'{name} t={yr:3d} dE={dE:4.0f} V/V0={out["V"][-1] / out["V"][0]:.3f} L={out["L"][-1]:.0f} Qw={Qw:.1f} | '
                  f'fine: erosion {out["fine_Q_ero"][-1]:.2e} export {out["fine_Q_out"][-1]:.2e} stored fraction {out["fine_f_stor"][-1]:.2f} | {time.time() - t0:.0f} s', flush=True)
        if yr < t_end:
            g.step(dE_total * min(yr + 1, t_ramp) / t_ramp)
    np.savez(bf.OUT_DIR / f'transient_{name}{tag}.npz', **{k: np.array(v) for k, v in out.items()}, **{k: np.array(v) for k, v in snap.items()},
             ELA0=g.ELA0, beta_abl=g.mb['beta_abl'], beta_acc=g.mb['beta_acc'], x=g.x, b=g.b, hb_final=g.hb, hb_init=g.hb_obs)
    print(f'saved transient_{name}{tag}.npz')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('glacier', choices=list(CFG))
    ap.add_argument('--dE', type=float, default=150.0); ap.add_argument('--ramp', type=int, default=50); ap.add_argument('--end', type=int, default=100)
    ap.add_argument('--exponent', type=float); ap.add_argument('--ratio', type=float)
    ap.add_argument('--fine-um', type=float, default=63.0); ap.add_argument('--refine', type=int, default=32); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    main(a.glacier, a.dE, a.ramp, a.end, a.exponent, a.ratio, a.fine_um, a.refine, a.tag)
