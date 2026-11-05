"""Analytical verification tests for blatter_flow.py: simple uniform inputs with known answers.

    python tests/test_blatter_flow.py [--slow]    (--slow adds the BP-solver convergence test)
    python -m pytest tests/test_blatter_flow.py -v"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import blatter_flow as bf

try:
    import pytest
    _slow = pytest.mark.slow
except ImportError:
    _slow = lambda f: f     # no-op decorator if pytest not installed

SPY = 365.25 * 24 * 3600   # seconds per year

# Shared synthetic geometry: uniform slope, same length as Arolla
NX    = bf.NX
L     = 3800.0
x     = np.linspace(0.0, L, NX)
dx    = x[1] - x[0]
ALPHA = 0.08                        # rad ≈ 4.6°
DBDX  = np.full(NX, np.sin(ALPHA))  # uniform bed slope


def test_rothlisberger_channel():
    """At the terminus the capacity equals the Engelund-Hansen formula applied analytically at Q_w = Q_w_total."""
    Q_w_total = 10.0
    d = 0.01
    _, Q_s_cap, _ = bf.fluvial_transport(
        Q_w_total, np.ones(NX), DBDX, dx, x, d_grain=d)

    S       = max(np.sin(ALPHA), 1e-4)
    r       = (2.0**(5.0/3.0) * bf.N_MANN * Q_w_total / (np.pi * np.sqrt(S)))**0.375
    theta   = bf.RHO_W * (r / 2.0) * S / ((bf.RHO_SED - bf.RHO_W) * d)
    Phi     = 0.05 * max(theta - bf.THETA_C, 0.0)**2.5
    q_s     = Phi * np.sqrt((bf.RHO_SED / bf.RHO_W - 1.0) * 9.81 * d**3)
    expected = bf.RHO_SED * q_s * 2.0 * r * SPY

    np.testing.assert_allclose(Q_s_cap[-1], expected, rtol=1e-10,
                                err_msg="Terminal transport capacity mismatch")


def test_fluvial_supply_limited():
    """When capacity >> supply the cumulative channel flux equals the cumulative erosional supply at every node except node 0, where Q_w = 0."""
    u_b   = np.full(NX, 5.0)
    e_dot = bf.K_G * np.abs(u_b)**bf.L_EXP

    Q_s_fl, Q_s_cap, _ = bf.fluvial_transport(
        500.0, e_dot, DBDX, dx, x, d_grain=0.001)

    Q_supply = np.cumsum((1.0 - bf.ALPHA_D) * bf.RHO_R * e_dot * bf.W_GL) * dx

    assert np.all(Q_s_cap[1:] > Q_s_fl[1:]), "Capacity must exceed flux at nodes 1..NX-1"
    np.testing.assert_allclose(Q_s_fl[1:], Q_supply[1:], rtol=1e-10)


def test_fluvial_transport_limited():
    """
    When supply >> capacity (high velocity, coarse grains, low Q_w), the
    channel flux must be capped at capacity:  Q_s_fl = min(Q_s_cap, Q_s_sup).
    """
    u_b   = np.full(NX, 500.0)
    e_dot = bf.K_G * np.abs(u_b)**bf.L_EXP

    Q_s_fl, Q_s_cap, Q_s_sup = bf.fluvial_transport(
        0.5, e_dot, DBDX, dx, x, d_grain=0.20)

    assert np.all(Q_s_sup[5:] > Q_s_cap[5:]), "Supply must exceed capacity"
    np.testing.assert_allclose(
        Q_s_fl, np.minimum(Q_s_cap, Q_s_sup), rtol=1e-12)


# Till shielding, transport-limited: closed-form H_till


# Till shielding, supply-limited: no till, no shielding


# Dispersed layer: linear steady state when entrainment exceeds melt-out


# Dispersed layer: no layer forms when melt-out exceeds entrainment


# Coupled solve: supply-limited everywhere


# Coupled solve: transport-limited nodes at capacity


# Coupled solve: sediment conservation along the flowline


# Coupled solve: a genuine trap reports accumulation, not a discarded flux


# Full budget closes with Da_sed = 1 when nothing accumulates


def test_hydrology_darcy_consistency_and_bounds():
    """On a uniform slab the sheet satisfies Darcy's law node by node, 0 <= p_w <= p_i, S_hyd >= 0 and phi decreasing downstream.
    A very large discharge pressurises the sheet and a tiny one leaves it unpressurised (S_hyd = bed slope)."""
    b   = -x * np.sin(ALPHA)
    hb  = np.full(NX, 150.0)
    u_b = np.full(NX, 30.0)

    N, p_w, hs, S, phi = bf.hydrology_1d(x, hb, b, u_b, 20.0, dx)
    q = np.maximum(20.0 * x / L, 1e-12) / bf.W_GL
    qi = 0.5 * (q[:-1] + q[1:])
    darcy = bf.K_SHEET * hs[:-1]**3 * (phi[:-1] - phi[1:]) / dx
    pressurised = p_w[:-1] > 1.0
    assert pressurised.any(), "expected a pressurised sheet at Q_w = 20 m³/s"
    np.testing.assert_allclose(darcy[pressurised], qi[pressurised], rtol=1e-6)
    assert np.all(p_w >= -1e-6) and np.all(p_w <= bf.RHO_ICE * 9.81 * hb + 1e-6)
    assert np.all(S >= 0.0) and np.all(np.diff(phi) <= 1e-9)

    N2, p_w2, _, S2, _ = bf.hydrology_1d(x, hb, b, u_b, 1e-4, dx)
    np.testing.assert_allclose(p_w2, 0.0, atol=1e-6)
    np.testing.assert_allclose(S2[:-1], np.sin(ALPHA), rtol=1e-9)


def test_fringe_regimes():
    """No fringe forms below the entry pressure or for N <= 0; above it the fringe thickens with effective pressure and thins with basal melt."""
    m = 0.05
    assert bf.fringe_steady_thickness(0.0, m) == (0.0, 1.0)
    assert bf.fringe_steady_thickness(0.5 * bf.P_FRINGE, m)[0] == 0.0
    H1, th1 = bf.fringe_steady_thickness(2e5, m)
    H2, th2 = bf.fringe_steady_thickness(1e6, m)
    H3, _   = bf.fringe_steady_thickness(1e6, 5.0 * m)
    assert 0.0 < H1 < H2, "fringe must thicken with N"
    assert 0.0 < H3 < H2, "fringe must thin with basal melt"
    G  = -(m / bf.SPY) * bf.RHO_ICE * bf.L_FUS / bf.K_ICE
    np.testing.assert_allclose(H2, (th2 - 1.0) * (bf.T_MELT - bf.T_FRINGE) / (-G), rtol=1e-9)
    # steady state: V(theta) = -m
    num, den, V_n = bf._fringe_terms(th2, 1e6, G)
    np.testing.assert_allclose(V_n * num / den, -m / bf.SPY, rtol=1e-6)


def test_supercooling_energy_balance():
    """A normal reach has no freeze-on; an adverse reach where p_w drops faster than phi freezes on at (P_req - P_diss)/(L rho_i W)."""
    n   = NX
    b   = np.where(x < L / 2, -0.1 * x, -0.1 * L / 2 + 0.3 * (x - L / 2))   # adverse 2nd half
    phi = 1e7 - 50.0 * x                                                  # Pa, falls along flow; p_w stays > 0
    p_w = phi - bf.RHO_W * 9.81 * b
    m_fr, sc = bf.supercooling_freeze_on(x, 10.0, p_w, phi, dx)

    # centred gradients: exclude the two nodes straddling the slope break
    first  = x < L / 2 - 1.5 * dx
    second = x > L / 2 + 1.5 * dx
    assert not sc[first].any() and np.all(m_fr[first] == 0.0)
    assert sc[second].all(), "adverse reach must be supercooled"
    Q_w = 10.0 * x / L
    P_diss = Q_w * 50.0
    P_req  = bf.RHO_W * bf.C_W * bf.C_T * Q_w * (50.0 + bf.RHO_W * 9.81 * 0.3)
    expect = (P_req - P_diss) / bf.L_FUS / (bf.RHO_ICE * bf.W_GL) * bf.SPY
    np.testing.assert_allclose(m_fr[second][:-1], expect[second][:-1], rtol=1e-9)


def test_sheet_capacity_is_grain_size_selective():
    """In a supercooled-reach sheet (q = 0.014 m^2/s, h = 0.08 m) silt is carried, sand and gravel are not, and the capacity falls with grain size."""
    # q chosen so that the sheet velocity q/(F_SHEET h) = 0.18 m/s, tau = 0.4 Pa
    q, h = np.full(3, 1.43e-3), np.full(3, 0.08)
    caps = [bf.sheet_transport_capacity(q, h, d)[0] for d in (63e-6, 1e-3, 1e-2)]
    assert caps[0] > 0.0, "silt must be carried"
    assert caps[1] == 0.0 and caps[2] == 0.0, "sand and gravel must be deposited"
    tau = bf.F_DW / 8.0 * bf.RHO_W * (1.43e-3 / bf.F_SHEET / 0.08)**2
    assert 0.1 < tau < 1.0


def test_basal_ice_takes_only_bed_sediment():
    """Fringe uptake plus freeze-on at a node cannot exceed the bed pool; where u_b = 0 there is no local erosion, so the upstream
    channel load passes and the basal-ice uptake is zero."""
    b   = -x * np.sin(ALPHA)
    hb  = np.full(NX, 150.0)
    u_b = np.where(x < L / 2, 40.0, 0.0)
    r = bf.steady_routing(u_b, 5.0, DBDX, dx, x, hb, b, d_grain=63e-6)
    lower = x > L / 2 + dx
    assert not r['chan_bedload'][lower].any(), "63 um must be suspended in the channel"
    np.testing.assert_allclose((r['uptake_fr'] + r['accretion_fr'])[lower], 0.0, atol=1e-9)
    # with the ice at rest the fringe can only release (melt-limited), so the
    # basal-ice flux is non-increasing and the channel flux non-decreasing
    assert np.all(np.diff(r['Q_fr'][lower][:-1]) <= 1e-9)
    assert np.all(np.diff(r['Q_fl'][lower][:-1]) >= -1e-9)
    # and the channel gains exactly what the fringe and dispersed layer release
    gain = np.diff(r['Q_fl'][lower][:-1])
    rel  = (r['release_fr'][lower][1:-1]
            - np.diff(r['Q_d'][lower][:-1]))        # dispersed melt-out = -dQ_d
    np.testing.assert_allclose(gain, rel, rtol=1e-9, atol=1e-6)


def test_sheet_capacity_rouse():
    """In the sheet silt has Rouse number < 2.5 and passes freely, gravel is deposited, and the settling velocity is monotone in grain size."""
    q, h = np.full(2, 1.43e-3), np.full(2, 0.08)
    cap_silt   = bf.sheet_transport_capacity(q, h, 63e-6)[0]
    cap_gravel = bf.sheet_transport_capacity(q, h, 1e-2)[0]
    assert cap_silt >= 1e29 and cap_gravel == 0.0
    ws = bf.settling_velocity(np.array([63e-6, 1e-3, 1e-2]))
    assert np.all(np.diff(ws) > 0) and 1e-3 < ws[0] < 5e-3


def test_prescribed_fw_thins_fringe_and_conserves():
    """Raising f_w thins the equilibrium fringe everywhere, the budget stays closed and the sheet hydrology is unaffected."""
    b   = -x * np.sin(ALPHA)
    hb  = 60.0 + 90.0 * np.sin(np.pi * x / L)
    u_b = 5.0 + 45.0 * np.sin(np.pi * x / L)**2
    r_lo = bf.steady_routing(u_b, 2.0, DBDX, dx, x, hb, b, d_grain=0.05, f_w=0.6)
    r_hi = bf.steady_routing(u_b, 2.0, DBDX, dx, x, hb, b, d_grain=0.05, f_w=0.95)
    np.testing.assert_allclose(r_lo['N'], r_hi['N'])
    np.testing.assert_allclose(r_lo['S_hyd'], r_hi['S_hyd'])
    np.testing.assert_allclose(r_lo['N_fringe'], 0.4 * bf.RHO_ICE * 9.81 * hb)
    assert np.all(r_hi['H_f_eq'] <= r_lo['H_f_eq'] + 1e-12)
    assert (r_hi['H_f_eq'] < r_lo['H_f_eq']).any()
    for r in (r_lo, r_hi):
        bud = bf.budget_from_routing(r, dx)
        res = bud['Q_fl'] + bud['Q_def'] + bud['Q_fr'] + bud['Q_disp'] + bud['Q_acc'] - bud['Q_ero']
        assert abs(res) / bud['Q_ero'] < 1e-9


def test_steady_routing_conserves_sediment():
    """At every node of the budget domain the flux leaving equals the flux entering plus the local shielded erosion, and the integrated budget closes."""
    b   = -x * np.sin(ALPHA)
    hb  = 60.0 + 90.0 * np.sin(np.pi * x / L)
    u_b = 5.0 + 45.0 * np.sin(np.pi * x / L)**2

    r = bf.steady_routing(u_b, 2.0, DBDX, dx, x, hb, b, d_grain=0.05)
    fresh = bf.RHO_R * r['e_dot'] * bf.W_GL * dx
    out   = r['Q_fl'] + r['Q_def'] + r['Q_fr'] + r['Q_d'] + r['Q_acc']
    inn   = np.concatenate([[0.0], (r['Q_fl'] + r['Q_def'] + r['Q_fr'] + r['Q_d'])[:-1]])
    np.testing.assert_allclose(out[:-1], (inn + fresh)[:-1], rtol=1e-9)
    assert np.all(r['Q_fr'] >= 0) and np.all(r['Q_d'] >= 0) and np.all(r['H_till'] >= 0)
    assert (r['H_f_eq'] > 0).any(), "expected a fringe somewhere on this slab"

    bud = bf.budget_from_routing(r, dx)
    res = bud['Q_fl'] + bud['Q_def'] + bud['Q_fr'] + bud['Q_disp'] + bud['Q_acc'] - bud['Q_ero']
    assert abs(res) / bud['Q_ero'] < 1e-9
    if bud['Q_acc'] == 0.0:
        np.testing.assert_allclose(bud['T'], 1.0, rtol=1e-9)
        np.testing.assert_allclose(bud['Da'], 1.0, rtol=1e-9)


def test_newton_sigma_raises_on_nonconvergence():
    """
    With zero iterations the residual of a non-trivial problem cannot be below
    tolerance, so newton_sigma_vec must raise rather than return the guess.
    """
    s   = np.zeros(NX)
    c   = np.full(NX, 0.05)
    g   = np.full(NX, 1e-2)          # du/dx (1/yr)
    tau = np.full(NX, 0.5)           # bar
    try:
        bf.newton_sigma_vec(s, c, g, tau, n_iter=0, strict=True)
    except RuntimeError:
        return
    raise AssertionError("newton_sigma_vec returned without converging")


def test_shoot_raises_on_nonconvergence():
    """
    With MAXITER = 1 the shooter cannot satisfy the stress-free surface BC on
    a slab and must raise RuntimeError instead of returning the partial state.
    """
    sin_a  = 0.2
    b_slab = -x * sin_a + 20.0 * np.sin(2 * np.pi * x / 800.0)   # a uniform slab is solved exactly at iteration 0
    h_slab = 200.0 - x * sin_a
    _, _, dx_s, ds_s, _, _, hb_s, dhdx_s, dbdx_s, C_s, d_s, _ = \
        bf.setup_geometry(x, b_slab, h_slab)

    old = bf.MAXITER
    bf.MAXITER = 1
    try:
        try:
            bf.shoot(x, dx_s, ds_s, hb_s, dhdx_s, d_s, C_s)
        except RuntimeError:
            return
        raise AssertionError("shoot returned without converging")
    finally:
        bf.MAXITER = old


@_slow
def test_shoot_convergence_and_weertman_bc():
    """On a uniform slab (11.5 degrees, H = 200 m) the shooter satisfies the stress-free surface condition within TKRIT, the Weertman
    law holds at the base and the surface is faster than the base.  The shooter converges only for H/dx <~ 2, so the slab is placed
    on a 15 km grid; see test_slab_converges_at_high_aspect_ratio."""
    sin_a  = 0.2        # ≈ 11.5°; driving stress 3.5 bar, u_b ≈ 215 m/yr
    H_s    = 200.0
    x_s    = np.linspace(0.0, 15000.0, NX)
    b_slab = -x_s * sin_a
    h_slab = H_s + b_slab

    _, _, dx_s, ds_s, _, _, hb_s, dhdx_s, dbdx_s, C_s, d_s, _ = \
        bf.setup_geometry(x_s, b_slab, h_slab)
    assert H_s / dx_s < 2.0

    U, _, Tau = bf.shoot(x_s, dx_s, ds_s, hb_s, dhdx_s, d_s, C_s)

    # (a) stress-free surface BC
    tmax = np.max(np.abs(Tau[:, -1]))
    assert tmax < bf.TKRIT, \
        f"Shooter did not converge: τ_surface max = {tmax:.3e} bar (TKRIT = {bf.TKRIT})"

    # (b) Weertman law at the base (set exactly by the last iteration)
    tau_b = Tau[:, 0]
    u_b_weertman = bf.C_SLIDE * np.abs(tau_b)**(bf.M_SLIDE - 1) * tau_b
    np.testing.assert_allclose(U[:, 0], u_b_weertman, rtol=1e-10,
                                err_msg="Basal velocity must satisfy Weertman law")

    # (c) surface faster than base (internal deformation is positive)
    assert np.mean(U[:, -1]) > np.mean(U[:, 0]), \
        "Mean surface velocity must exceed mean basal velocity"

    # (d) interior basal velocity close to the Weertman slab value
    tau_d = bf.RHO_G * H_s * sin_a
    u_slab = bf.C_SLIDE * tau_d**bf.M_SLIDE
    assert 0.8 * u_slab < np.median(U[5:-5, 0]) < 1.2 * u_slab


@_slow
def test_slab_converges_at_high_aspect_ratio():
    """A uniform 200 m slab on the 3.8 km grid (H/dx = 4.1) is solved exactly with zero-gradient end conditions."""
    sin_a  = 0.2
    b_slab = -x * sin_a
    h_slab = 200.0 + b_slab
    _, _, dx_s, ds_s, _, _, hb_s, dhdx_s, dbdx_s, C_s, d_s, _ = \
        bf.setup_geometry(x, b_slab, h_slab)
    assert 200.0 / dx_s > 3.0
    U, Sigma, Tau = bf.shoot(x, dx_s, ds_s, hb_s, dhdx_s, d_s, C_s, verbose=False)
    tau_d = bf.RHO_G * 200.0 * sin_a
    u_slab = bf.C_SLIDE * tau_d**bf.M_SLIDE
    assert abs(U[0, 0] / u_slab - 1) < 0.05 and abs(U[-1, 0] / u_slab - 1) < 0.05


# Runner
_FAST = [
    test_rothlisberger_channel,
    test_fluvial_supply_limited,
    test_fluvial_transport_limited,
    test_hydrology_darcy_consistency_and_bounds,
    test_fringe_regimes,
    test_supercooling_energy_balance,
    test_sheet_capacity_is_grain_size_selective,
    test_basal_ice_takes_only_bed_sediment,
    test_sheet_capacity_rouse,
    test_prescribed_fw_thins_fringe_and_conserves,
    test_steady_routing_conserves_sediment,
    test_newton_sigma_raises_on_nonconvergence,
    test_shoot_raises_on_nonconvergence,
]
_SLOW = [test_shoot_convergence_and_weertman_bc,
         test_slab_converges_at_high_aspect_ratio]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Analytical verification tests for blatter_flow.py')
    parser.add_argument('--slow', action='store_true',
                        help='also run the BP-solver convergence test (~30 s)')
    args = parser.parse_args()

    tests  = _FAST + (_SLOW if args.slow else [])
    passed, failed = 0, []

    for t in tests:
        try:
            t()
            print(f'  PASS  {t.__name__}')
            passed += 1
        except AssertionError as e:
            print(f'  FAIL  {t.__name__}: {e}')
            failed.append(t.__name__)
        except Exception as e:
            print(f'  ERROR {t.__name__}: {type(e).__name__}: {e}')
            failed.append(t.__name__)

    tag = '' if args.slow else '  (BP-solver test skipped; use --slow to include)'
    print(f'\n{passed}/{len(tests)} tests passed.{tag}')
    if failed:
        print('Failed:', ', '.join(failed))
        sys.exit(1)
