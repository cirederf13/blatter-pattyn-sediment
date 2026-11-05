"""
ELA-forced transient experiment: prognostic flowline ice.

The surface mass balance is m(s) = min(beta (s - ELA), c), and the thickness follows from mass
conservation, dH/dt = m - (1/W) dQ/dx, with the ice flux Q = W H <u> from a Blatter-Pattyn solve of
the current geometry (warm-started from the previous basal stress).

The present geometry is the initial state.  Its flux divergence defines a residual m_res(x) that is
added to the mass balance, so that the glacier is exactly steady at t = 0 under the reference
equilibrium-line altitude ELA_0; the forcing ELA(t) = ELA_0 + dE(t) perturbs this steady state.
"""
import numpy as np
import blatter_flow as bf

# Piecewise-linear balance, beta_abl below the ELA and beta_acc above it (yr^-1), with a cap (m/yr ice).
# Alps: elevation-band balances of the World Glacier Monitoring Service give beta_abl 0.004-0.010 and
# beta_acc/beta_abl 0.1-0.3.  Franz Josef Glacier: Anderson et al. (2006), one slope with a cap and ELA_0 = 1800 m.
MELT_H_REF = 20.0     # m; ice thinner than this contributes proportionally less melt to Q_w
BETA_ALPS = dict(beta_abl=0.008, beta_acc=0.0024, cap=2.0, bounds=(0.004, 0.012), fit=False)
BETA_FJ   = dict(beta_abl=0.02, beta_acc=0.02, cap=4.4, bounds=(0.01, 0.03), fit=False, ela0=1800.0)
MAX_BP_FAIL = 3
SMOOTH_M  = 1000.0    # smoothing length of the steady balance (m)


def mass_balance(s, ELA, p):
    z = s - ELA
    return np.minimum(np.where(z < 0, p['beta_abl'], p['beta_acc']) * z, p['cap'])


def ice_flux(U, hb, W):
    """Q(x) (m^3/yr) from sigma-level velocities, trapezoid over sigma in [0, 1]."""
    u_mean = np.trapz(U, dx=1.0 / (U.shape[1] - 1), axis=1)
    Q = W * hb * u_mean
    Q[-1] = Q[-2]                        # the solver pins u = 0 at the end node
    return Q


def flux_divergence(Q, x):
    """Conservative (1/dx) difference of half-node fluxes; zero flux at the head, free outflow at the terminus."""
    dx = x[1] - x[0]
    Qh = 0.5 * (Q[1:] + Q[:-1])
    dQ = np.zeros_like(Q)
    dQ[1:-1] = (Qh[1:] - Qh[:-1]) / dx
    dQ[0] = Qh[0] / dx
    dQ[-1] = (Q[-1] - Qh[-1]) / dx      # outflow: ice reaching the last node leaves the domain
    return dQ


class FlowlineIce:
    """BP-prognostic flowline on the x grid of a bp_solution."""

    def __init__(self, key, loader, W, c_slide, shape, mb, verbose=False, nx=None):
        sol = bf.bp_solution(key, loader, c_slide=c_slide, shape=shape, nx=nx)
        (self.x, self.sig, self.dx, self.ds, self.b, self.h, self.hb, self.dhdx,
         self.dbdx, self.C, self.d, self.Z_o, self.U, self.Sigma, self.Tau) = sol
        self.W, self.c_slide, self.shape, self.mb, self.verbose = W, c_slide, shape, mb, verbose
        self.h_obs, self.hb_obs = self.h.copy(), self.hb.copy()
        self.tau0 = self.Tau[:, 0].copy()
        self.t = 0.0
        self.n_bp_fail = 0
        # steady-state balance of the present geometry.  The raw flux
        # divergence of a diagnostic solve on a DEM is dominated by bed
        # roughness, so it is smoothed over SMOOTH_M before use; the residual
        # m_res = m_steady - m(s; ELA_0) is kept as a fixed spatial correction
        # (the catchment-integrated balance is carried by the parameterisation).
        from scipy.ndimage import gaussian_filter1d
        from scipy.optimize import brentq
        self.ice0 = self.hb > 1.0
        # the DEM-scale roughness of the diagnostic flux is removed from the
        # flux itself (same smoothing used in every later step), so that the
        # residual balance keeps the geometry EXACTLY steady at t = 0
        dQ = self.flux_div(self.U, self.hb)
        self.m_steady = np.where(self.ice0, dQ, 0.0)
        lo, hi = self.h[self.ice0].min() - 500, self.h[self.ice0].max() + 500
        def ela_for(p):
            return brentq(lambda E: np.mean((self.m_steady - mass_balance(self.h, E, p))[self.ice0]), lo, hi)
        def rms(p):
            E = ela_for(p)
            return np.sqrt(np.mean((self.m_steady - mass_balance(self.h, E, p))[self.ice0] ** 2))
        mb = dict(mb)
        self.rms_start = rms(mb)
        if mb.get('fit', False):
            # slopes fitted to the steady balance (ELA_0 follows from zero mean residual)
            from scipy.optimize import minimize
            bl = mb['bounds']
            f = lambda v: rms({**mb, 'beta_abl': v[0], 'beta_acc': v[1]})
            sol_ = minimize(f, [mb['beta_abl'], mb['beta_acc']], bounds=[bl, bl], method='L-BFGS-B',
                            options=dict(eps=1e-6))
            mb['beta_abl'], mb['beta_acc'] = float(sol_.x[0]), float(sol_.x[1])
        self.mb = mb
        # ELA_0 pinned when observed (mb['ela0']); the residual then carries the mismatch
        self.ELA0 = mb['ela0'] if mb.get('ela0') else ela_for(mb)
        self.rms_fit = float(np.sqrt(np.mean((self.m_steady - mass_balance(self.h, self.ELA0, mb))[self.ice0] ** 2)))
        self.m_res = np.where(self.ice0, self.m_steady - mass_balance(self.h, self.ELA0, mb), 0.0)
        if verbose:
            print(f'balance: beta_abl {mb["beta_abl"]:.4f} beta_acc {mb["beta_acc"]:.4f} ELA0 {self.ELA0:.0f} '
                  f'rms residual {self.rms_start:.2f} -> {self.rms_fit:.2f} m/yr')

    def flux_div(self, U, hb):
        from scipy.ndimage import gaussian_filter1d
        Q = gaussian_filter1d(ice_flux(U, hb, self.W), SMOOTH_M / self.dx)
        return flux_divergence(Q, self.x) / self.W

    def smb(self, dE):
        m = mass_balance(self.h, self.ELA0 + dE, self.mb) + self.m_res
        # no accumulation on ice-free bedrock outside the initial extent
        return np.where(self.ice0 | (self.hb > 0.0), m, np.minimum(m, 0.0))

    def solve_bp(self):
        (self.x, self.sig, self.dx, self.ds, self.b, self.h, self.hb, self.dhdx,
         self.dbdx, self.C, self.d, self.Z_o) = bf.setup_geometry(self.x, self.b, self.h, nx=len(self.x))
        ice = self.hb > 1.0
        dhdx = np.where(ice, self.dhdx, 0.0)
        C = self.C.copy(); C[~ice, :] = 0.0
        d = np.where(ice, self.d, 0.0)
        tau0 = np.where(ice, self.tau0, 0.0)
        try:
            U, Sigma, Tau = bf.shoot(self.x, self.dx, self.ds, self.hb, dhdx, d, C,
                                     c_slide=self.c_slide, shape=self.shape,
                                     tau0=tau0, verbose=self.verbose)
            U[~ice, :] = 0.0
            self.U, self.Sigma, self.Tau = U, Sigma, Tau
            self.tau0 = Tau[:, 0].copy()
        except RuntimeError as err:
            try:    # cold restart from the driving stress
                U, Sigma, Tau = bf.shoot(self.x, self.dx, self.ds, self.hb, dhdx, d, C,
                                         c_slide=self.c_slide, shape=self.shape,
                                         tau0=None, verbose=self.verbose)
                U[~ice, :] = 0.0
                self.U, self.Sigma, self.Tau = U, Sigma, Tau
                self.tau0 = Tau[:, 0].copy()
            except RuntimeError:
                self.n_bp_fail += 1
                if self.verbose:
                    print(f'  BP solve failed at t = {self.t:.0f}: {err}')
                self.U[~ice, :] = 0.0
                if self.n_bp_fail > MAX_BP_FAIL:
                    raise RuntimeError(f'{self.n_bp_fail} consecutive BP failures; '
                                       f'geometry too thick for the grid (H/dx > 2)?')

    def step(self, dE, dt=1.0):
        """Advance H by dt (yr) under ELA_0 + dE with the flux of the current solve, then re-solve."""
        dQ = self.flux_div(self.U, self.hb)
        self.hb = np.maximum(self.hb + dt * (self.smb(dE) - dQ), 0.0)
        self.h = self.b + self.hb
        self.t += dt
        self.solve_bp()

    def drift_check(self, years=10):
        """Unforced run; returns (dV/V, max |dH|) after `years`."""
        V0, hb0 = self.volume(), self.hb.copy()
        for _ in range(years):
            self.step(0.0)
        return self.volume() / V0 - 1.0, float(np.max(np.abs(self.hb - hb0)))

    def volume(self):
        return self.W * np.sum(self.hb) * self.dx

    def length(self):
        ice = np.where(self.hb > 1.0)[0]
        return self.x[ice[-1]] - self.x[ice[0]] if len(ice) else 0.0

    def ablation_melt(self, dE=0.0):
        """Ablation-area melt (m^3 ice/yr), used to scale Q_w(t)."""
        m = self.smb(dE)
        wt = np.clip(self.hb / MELT_H_REF, 0.0, 1.0)      # continuous in thickness: a thinning terminus node must not drop out in one step
        return self.W * np.sum(np.maximum(-m, 0.0) * wt) * self.dx
