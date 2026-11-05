"""
Transient sediment reservoirs on an evolving flowline (Part B of the
ELA-forced experiment).  Each year the rates of blatter_flow.steady_routing
are evaluated for the current ice geometry, velocity and hydrology, but the
reservoirs are advanced in time instead of being set to equilibrium:

  till       dM_t/dt = supply_to_bed - channel - deformation(H_till) - uptake + release
  basal ice  M_f advected with u_b (upwind), fed by freeze-on and fringe uptake
             toward the equilibrium fringe, released at the melt-limited rate
  dispersed  M_d advected with u_b, fed by regelation, melted out where no fringe

All masses in kg per node.  A two-component tracer (fresh / stored) rides on
every reservoir and flux: sediment eroded during the run is "fresh"; the
initial reservoir content is "stored".  f_stor(t) = stored share of the
export is the quantity compared with De Doncker et al. (2026).
Till left on deglaciated nodes is counted separately ("exposed").
"""
import numpy as np
import blatter_flow as bf


H_TERM = 5.0      # m; ice thinner than this traps sediment less (weight hb/H_TERM), none at the margin


class RefinedView:
    """
    The ice fields of a FlowlineIce interpolated onto a grid `k` times finer, so
    that the sediment reservoirs are resolved well below the ice-flow spacing.
    On the ice grid a moving boundary (e.g. where the equilibrium fringe goes
    to zero) releases a whole node's pool within a few years and then nothing
    for a decade; on the refined grid the same boundary sweeps smoothly.
    """
    def __init__(self, g, k):
        self.k = k
        n = len(g.x)
        self.x = np.linspace(g.x[0], g.x[-1], (n - 1) * k + 1)
        self.dx = self.x[1] - self.x[0]
        ip = lambda a: np.interp(self.x, g.x, a)
        self.W = g.W
        self.hb, self.b, self.dbdx = ip(g.hb), ip(g.b), ip(g.dbdx)
        # sub-node terminus: extend the upstream surface slope to the zero-thickness crossing instead of the next ice-free node
        ice_nodes = np.nonzero(g.hb > 0.0)[0]
        j = ice_nodes[-1] if len(ice_nodes) else 0
        if 1 <= j < n - 1:
            slope = (g.hb[j - 1] - g.hb[j]) / (g.x[j] - g.x[j - 1])
            if slope > 0:
                x_t = min(g.x[j] + g.hb[j] / slope, g.x[j + 1])
                tail = self.x > g.x[j]
                self.hb[tail] = np.maximum(g.hb[j] - slope * (self.x[tail] - g.x[j]), 0.0)
        self.U = ip(g.U[:, 0])[:, None]
        cs = np.asarray(g.c_slide)
        self.c_slide = ip(cs) if cs.ndim else g.c_slide

    def to_coarse(self, a, mass=False, weight=None):
        """Fine-node array -> ice-grid nodes: sum over each coarse cell (mass) or sample / weighted mean."""
        k, nf = self.k, len(a)
        idx = np.minimum(np.rint(np.arange(nf) / k).astype(int), (nf - 1) // k)
        nc = (nf - 1) // k + 1
        if mass:
            out = np.zeros(nc); np.add.at(out, idx, a); return out
        if weight is None:
            return a[::k]
        num = np.zeros(nc); den = np.zeros(nc)
        np.add.at(num, idx, a * weight); np.add.at(den, idx, weight)
        return np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)


class SedimentState:
    def __init__(self, n):
        # reservoir masses (kg) and their stored-tracer share
        self.M_till = np.zeros(n); self.M_f = np.zeros(n); self.M_d = np.zeros(n)
        self.s_till = np.zeros(n); self.s_f = np.zeros(n); self.s_d = np.zeros(n)
        self.exposed = 0.0; self.exposed_stored = 0.0

    @staticmethod
    def _mix(M, s, dM, s_in):
        """Add mass dM with stored share s_in to a pool (M, s); return new (M, s)."""
        Mn = M + dM
        if Mn <= 0.0:
            return 0.0, 0.0
        return Mn, (M * s + dM * s_in) / Mn

    def init_from_steady(self, r, dx, W):
        """Initial reservoirs = steady state of the present geometry; all 'stored'."""
        c_fr = bf.RHO_SED * (1.0 - bf.PHI_TILL) * W
        c_d = bf.RHO_ICE * bf.C_D * W
        self.M_till = bf.RHO_B * W * dx * r['H_till'].copy()
        self.M_f = c_fr * r['H_f'] * dx
        self.M_d = c_d * r['H_d'] * dx
        self.s_till[:] = 1.0; self.s_f[:] = 1.0; self.s_d[:] = 1.0
        # trapping nodes carry the nominal H_TILL_TRAP: cap the initial till at 2 m
        self.M_till = np.minimum(self.M_till, bf.RHO_B * W * dx * 2.0)
        self.M_till[-1] = self.M_f[-1] = self.M_d[-1] = 0.0

    def total(self):
        return self.M_till.sum() + self.M_f.sum() + self.M_d.sum()

    def total_stored(self):
        return (self.M_till * self.s_till).sum() + (self.M_f * self.s_f).sum() + (self.M_d * self.s_d).sum()


def sediment_step(st, g, Qw, d_grain, dt=1.0, f_w=bf.F_W):
    """
    Advance the sediment state by dt on the current glacier `g`
    (transient.FlowlineIce).  Returns a dict of annual fluxes (kg/yr) and
    their stored shares: Q_fl, Q_ice (fringe + dispersed past terminus),
    Q_ero, exposed, plus tracer shares s_fl, s_ice.
    """
    x, dx, W = g.x, g.dx, g.W
    n = len(x)
    u_b = np.abs(g.U[:, 0]); hb = g.hb; b = g.b
    ice = hb > 1.0
    # ---- rates for the current geometry (capacities, hydrology, fringe equilibrium, freeze-on, basal melt)
    r = bf.steady_routing(u_b, Qw, g.dbdx, dx, x, hb, b, d_grain=d_grain,
                          W=W, c_slide=g.c_slide, f_w=f_w)
    Q_cap, m_fr, m_net = r['Q_cap'], r['m_freeze'], r['m_basal'] - r['m_freeze']
    H_f_eq, theta, has_fr = r['H_f_eq'], r['theta'], r['H_f_eq'] > 0
    chan_bed = r['chan_bedload']
    tau_b = np.maximum((u_b / (g.c_slide + 1e-30))**(1.0 / bf.M_SLIDE), 0.01)
    B_def = bf.RHO_B * bf.C_DEF_TILL * tau_b * W
    c_fr = bf.RHO_SED * (1.0 - bf.PHI_TILL) * W
    c_d = bf.RHO_ICE * bf.C_D * W

    # ---- deglaciated nodes: till becomes exposed, basal ice melts out onto the bed (exposed too)
    gone = ~ice & (st.M_till + st.M_f + st.M_d > 0)
    for i in np.where(gone)[0]:
        m = st.M_till[i] + st.M_f[i] + st.M_d[i]
        ms = st.M_till[i] * st.s_till[i] + st.M_f[i] * st.s_f[i] + st.M_d[i] * st.s_d[i]
        st.exposed += m; st.exposed_stored += ms
        st.M_till[i] = st.M_f[i] = st.M_d[i] = 0.0

    # ---- erosion (shielded by current till)
    H_till = st.M_till / (bf.RHO_B * W * dx)
    shield = bf.SHIELD_MIN + (1.0 - bf.SHIELD_MIN) * np.exp(-H_till / bf.H_TILL_REF)
    e_dot = np.where(ice, bf.K_G * u_b**bf.L_EXP * shield, 0.0); e_dot[-1] = 0.0
    fresh = bf.RHO_R * e_dot * W * dx * dt                      # kg this year, stored share 0

    # ---- advect basal-ice pools (upwind, u_b dt / dx <= 1 assumed; sub-step otherwise)
    it = int(np.where(ice)[0][-1]) if ice.any() else n - 1     # terminus: last ice node
    def advect(M, s):
        cfl = u_b * dt / dx
        nsub = int(np.ceil(max(cfl.max(), 1e-9))); f = cfl / nsub
        out_total = 0.0; out_stored = 0.0
        for _ in range(nsub):
            flux = f * M; fs = flux * s
            flux[it + 1:] = 0.0; fs[it + 1:] = 0.0        # nothing moves beyond the terminus
            Mn = M - flux; Mn[1:] += flux[:-1]
            Ms = M * s - fs; Ms[1:] += fs[:-1]
            # basal ice reaching the terminus melts out into the stream, not onto the foreland
            if it < n - 1:
                Mn[it + 1] -= flux[it]; Ms[it + 1] -= fs[it]
                out_total += flux[it]; out_stored += fs[it]
            out_total += flux[-1]; out_stored += fs[-1]
            M = Mn; s = np.where(M > 0, Ms / np.maximum(M, 1e-30), 0.0)
        return M, s, out_total, out_stored
    st.M_f, st.s_f, out_f, out_fs = advect(st.M_f, st.s_f)
    st.M_d, st.s_d, out_d, out_ds = advect(st.M_d, st.s_d)

    # ---- downstream pass: bed pool, exchanges, channel
    wt = np.where(ice, np.clip(hb / H_TERM, 0.0, 1.0), 0.0)         # trapping weight: 1 in ice thicker than H_TERM, fading to 0 at the margin
    transparent = wt <= 0.0
    Q_fl = np.zeros(n); s_fl = np.zeros(n)
    Q_in = 0.0; s_in = 0.0; Q_def_in = 0.0; s_def_in = 0.0
    exp_tail = 0.0; exp_tail_s = 0.0                                  # exported at the terminal node
    for i in range(n):
        if i == n - 1:
            # outside the budget domain: everything arriving here is exported
            exp_tail = Q_in + Q_def_in; exp_tail_s = Q_in * s_in + Q_def_in * s_def_in
            Q_fl[i] = Q_in; s_fl[i] = s_in
            break
        bed = fresh[i]; bed_s = 0.0                                   # kg, stored kg
        if Q_def_in > 0: bed += Q_def_in; bed_s += Q_def_in * s_def_in
        susp = 0.0; susp_s = 0.0
        bed += chan_bed[i] * Q_in; bed_s += chan_bed[i] * Q_in * s_in
        susp, susp_s = (1.0 - chan_bed[i]) * Q_in, (1.0 - chan_bed[i]) * Q_in * s_in
        # till joins the bed pool (not on transparent nodes: the till of a thin margin node stays there and is left on the foreland)
        till_here = st.M_till[i]; s_till_here = st.s_till[i]
        keep = (1.0 - wt[i]) * till_here                              # share of the node's till left in place
        bed += wt[i] * till_here; bed_s += wt[i] * till_here * s_till_here
        # dispersed melt-out where no fringe
        if not has_fr[i] and st.M_d[i] > 0:
            melt = min(c_d * max(m_net[i], 0.0) * dx * dt, st.M_d[i])
            bed += melt; bed_s += melt * st.s_d[i]; st.M_d[i] -= melt
        # fringe release (melt-limited) where above equilibrium
        Mf_eq = c_fr * H_f_eq[i] * dx
        if st.M_f[i] > Mf_eq:
            rel = min(st.M_f[i] - Mf_eq, c_fr * max(m_net[i], 0.0) * dx * dt)
            bed += rel; bed_s += rel * st.s_f[i]; st.M_f[i] -= rel
        sb = bed_s / bed if bed > 0 else 0.0
        # freeze-on accretion from the bed pool
        if m_fr[i] > 0 and bed > 0:
            acc = min(c_fr * m_fr[i] * dx * dt, bed)
            st.M_f[i], st.s_f[i] = st._mix(st.M_f[i], st.s_f[i], acc, sb); bed -= acc; bed_s -= acc * sb
        # fringe uptake toward equilibrium (pore-scale infiltration: sand and finer only)
        if st.M_f[i] < Mf_eq and bed > 0 and d_grain <= bf.D_FRINGE_MAX:
            upt = min(Mf_eq - st.M_f[i], bed)
            st.M_f[i], st.s_f[i] = st._mix(st.M_f[i], st.s_f[i], upt, sb); bed -= upt; bed_s -= upt * sb
        # regelation into the dispersed layer
        if has_fr[i] and st.M_f[i] > 0 and u_b[i] > 0:
            H_d_i = st.M_d[i] / (c_d * dx)
            reg = min(c_d * bf.regelation_rate(theta[i], H_d_i) * dx * dt, st.M_f[i])
            st.M_d[i], st.s_d[i] = st._mix(st.M_d[i], st.s_d[i], reg, st.s_f[i]); st.M_f[i] -= reg
        # channel takes what capacity allows from bed + suspended
        avail = bed + susp; avail_s = bed_s + susp_s
        take = avail if transparent[i] else min(avail, Q_cap[i] * dt / wt[i])   # no trapping in ice too thin for a channel network, nor beyond the terminus
        s_take = avail_s / avail if avail > 0 else 0.0
        Q_fl[i] = take; s_fl[i] = s_take
        left = avail - take; left_s = left * s_take
        # of what is left, suspended load that could not pass settles into till too
        # till deformation moves part of the remaining bed load downstream
        H_left = left / (bf.RHO_B * W * dx)
        Q_def = min(B_def[i] * min(H_left, bf.H_DEF_MAX) * dt, left) if ice[i] else 0.0
        if transparent[i]:
            st.M_till[i] = till_here; st.s_till[i] = s_till_here           # untouched; nothing deposits on or is lifted from it
        else:
            M_new = left - Q_def + keep
            st.s_till[i] = ((left - Q_def) * s_take + keep * s_till_here) / M_new if M_new > 0 else s_take
            st.M_till[i] = M_new
        Q_in, s_in = take, s_take
        Q_def_in, s_def_in = Q_def, s_take

    # terminus convention as in budget_from_routing: the budget domain is nodes 0..n-2 and the rest is exported
    Q_fl_out = exp_tail / dt; s_out = exp_tail_s / max(exp_tail, 1e-30)
    # basal ice arriving at the terminal node by advection is exported too
    tail = st.M_f[-1] + st.M_d[-1]; tail_s = st.M_f[-1] * st.s_f[-1] + st.M_d[-1] * st.s_d[-1]
    st.M_f[-1] = st.M_d[-1] = 0.0
    out_f += tail; out_fs += tail_s
    Q_ice_out = (out_f + out_d) / dt
    s_ice = (out_fs + out_ds) / max(out_f + out_d, 1e-30)
    return dict(Q_ero=fresh.sum() / dt, Q_fl=Q_fl_out, s_fl=s_out, Q_ice=Q_ice_out, s_ice=s_ice,
                Q_out=Q_fl_out + Q_ice_out,
                f_stor=(Q_fl_out * s_out + Q_ice_out * s_ice) / max(Q_fl_out + Q_ice_out, 1e-30),
                V_stor=st.total() / bf.RHO_SED, M_stored=st.total_stored(),
                exposed=st.exposed, routing=r, Q_fl_nodes=Q_fl.copy(), Q_cap_nodes=Q_cap.copy(), M_till_nodes=st.M_till.copy(), ice_nodes=ice.copy())
