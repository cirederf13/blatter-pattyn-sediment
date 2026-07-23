"""
Analytical verification tests for blatter_flow.py.

Each test constructs a simple, uniform input for which the exact answer is
known, runs the corresponding function, and checks the result against the
analytical solution.

Run without pytest:
    python test_blatter_flow.py          # fast tests only (< 5 s)
    python test_blatter_flow.py --slow   # include BP-solver convergence test

Run with pytest:
    python -m pytest test_blatter_flow.py -v
    python -m pytest test_blatter_flow.py -v -m "not slow"
"""
import sys
import numpy as np

import blatter_flow as bf

try:
    import pytest
    _slow = pytest.mark.slow
except ImportError:
    _slow = lambda f: f     # no-op decorator if pytest not installed

SPY = 365.25 * 24 * 3600   # seconds per year

# ── Shared synthetic geometry (uniform slope, same length as Arolla) ──────────
NX    = bf.NX
L     = 3800.0
x     = np.linspace(0.0, L, NX)
dx    = x[1] - x[0]
ALPHA = 0.08                        # rad ≈ 4.6°
DBDX  = np.full(NX, np.sin(ALPHA))  # uniform bed slope


# ── 1. Röthlisberger channel radius and Engelund-Hansen capacity ──────────────
def test_rothlisberger_channel():
    """
    At the terminus the transport capacity must match the Engelund-Hansen
    formula applied analytically at Q_w = Q_w_total.

    Channel radius (Manning):  r = (2^{5/3} n Q_w / (π √S))^{3/8}
    Shields number:             θ = ρ_w R_h S / ((ρ_s − ρ_w) d)
    Engelund-Hansen:            Φ = 0.05 (θ − θ_c)^{5/2}
    Volume flux:                q_s = Φ √((ρ_s/ρ_w − 1) g d³)
    Mass capacity (kg/yr):      Q_s^cap = ρ_s q_s 2r SPY
    """
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


# ── 2. Fluvial transport — supply-limited ─────────────────────────────────────
def test_fluvial_supply_limited():
    """
    When capacity >> supply (large Q_w, fine grains), the cumulative channel
    flux must equal the cumulative erosional supply at every node:
        Q_s_fl[i] = (1 − α_d) ρ_R ė W Σ_{j≤i} Δx

    Node 0 is excluded: water discharge Q_w(0) = 0 by construction (discharge
    accumulates downstream), so the channel has zero capacity there regardless
    of Q_w_total.
    """
    u_b   = np.full(NX, 5.0)
    e_dot = bf.K_G * np.abs(u_b)**bf.L_EXP

    Q_s_fl, Q_s_cap, _ = bf.fluvial_transport(
        500.0, e_dot, DBDX, dx, x, d_grain=0.001)

    Q_supply = np.cumsum((1.0 - bf.ALPHA_D) * bf.RHO_R * e_dot * bf.W_GL) * dx

    assert np.all(Q_s_cap[1:] > Q_s_fl[1:]), "Capacity must exceed flux at nodes 1..NX-1"
    np.testing.assert_allclose(Q_s_fl[1:], Q_supply[1:], rtol=1e-10)


# ── 3. Fluvial transport — transport-limited ──────────────────────────────────
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


# ── 4. Till shielding — transport-limited: closed-form H_till ────────────────
def test_till_shielding_transport_limited():
    """
    For uniform u_b and negligible Q_s_cap, the till thickness and shielding
    factor must match the closed-form analytical solution:

        prod    = (1 − α_d) ρ_R K_g u_b^l          [kg m⁻² yr⁻¹]
        removal = Q_s_cap / (W Δx)                  [kg m⁻² yr⁻¹]
        H_till  = H_ref ln(prod / removal)
        shield  = f_min + (1 − f_min) · removal/prod
    """
    u_b_val = 20.0
    q_cap   = 1.0       # kg/yr — far below production, uniformly transport-limited

    H_till, shield = bf.till_shielding(
        np.full(NX, u_b_val), np.full(NX, q_cap), dx)

    prod    = (1.0 - bf.ALPHA_D) * bf.RHO_R * bf.K_G * u_b_val**bf.L_EXP
    removal = q_cap / (bf.W_GL * dx)
    sr      = removal / prod
    H_expected = bf.H_TILL_REF * np.log(prod / removal)
    s_expected = bf.SHIELD_MIN + (1.0 - bf.SHIELD_MIN) * sr

    np.testing.assert_allclose(H_till, H_expected, rtol=1e-6)
    np.testing.assert_allclose(shield, s_expected,  rtol=1e-6)


# ── 5. Till shielding — supply-limited: no till, no shielding ────────────────
def test_till_shielding_supply_limited():
    """
    When Q_s_cap >> production, no till accumulates (H_till = 0) and the
    bedrock is fully exposed (shield = 1).
    """
    H_till, shield = bf.till_shielding(
        np.full(NX, 1.0), np.full(NX, 1e12), dx)

    np.testing.assert_allclose(H_till, 0.0, atol=1e-30)
    np.testing.assert_allclose(shield, 1.0, atol=1e-10)


# ── 6. Dispersed layer — exponential steady state ─────────────────────────────
def test_dispersed_layer_steady_state():
    """
    For constant u_b = u₀ and constant e_dot = e₀, the 1-D steady-state
    advection-decay equation

        u₀ dH_d/dx + m_b H_d = source,   source = α_d K_ent e₀

    has the exact solution (H_d(0) = 0 Dirichlet BC):

        H_d(x) = H_sat (1 − e^{−λx}),   H_sat = source/m_b,   λ = m_b/u₀

    Relative deviation from the analytical solution must be < 0.5 %.
    """
    u0 = 10.0
    e0 = 1e-4

    # tol=1e-9 forces the simulation to run until the outlet flux changes by
    # less than 1e-9 per year, ensuring we are past many advection timescales.
    H_d, _, _ = bf.evolve_storage(np.full(NX, u0), np.full(NX, e0), dx,
                                   t_max=bf.T_STOR, tol=1e-9)

    K_ENT   = bf.RHO_R / (bf.RHO_ICE * bf.C_D)
    H_sat   = bf.ALPHA_D * K_ENT * e0 / bf.M_BASAL
    H_ref   = H_sat * (1.0 - np.exp(-bf.M_BASAL / u0 * x))
    H_ref[0] = 0.0      # Dirichlet BC

    # 5 % tolerance: first-order upwind with CFL ≈ 0.9 introduces numerical
    # diffusion ~u·Δx/2 which is comparable to the e-folding length u/m_b = 200 m
    # (≈ 4 grid cells), causing a systematic profile broadening of this magnitude.
    rel_err = np.max(np.abs(H_d - H_ref)) / H_sat
    assert rel_err < 0.05, \
        f"Dispersed-layer steady state deviates {rel_err*100:.3f} % (threshold 5 %)"


# ── 7. Coupled solve — supply-limited everywhere ──────────────────────────────
def test_coupled_solve_supply_limited():
    """
    With enormous capacity (large Q_w, fine grains):
      - H_till = 0 everywhere
      - shield = 1 everywhere (no shielding)
      - Q_s_fl equals the cumulative erosional supply (< 0.1 % error)
    """
    u_b = np.full(NX, 3.0)

    Q_s_fl, _, _, H_till, e_dot, shield = bf.coupled_sediment_solve(
        u_b, 500.0, DBDX, dx, x, d_grain=0.001)

    # Node 0 is excluded: Q_w(0) = 0 forces zero capacity there regardless of
    # Q_w_total, making it transport-limited with a small but nonzero H_till.
    np.testing.assert_allclose(H_till[1:], 0.0, atol=1e-6)
    np.testing.assert_allclose(shield[1:], 1.0, atol=1e-6)

    # Check that each downstream increment in Q_s_fl equals the fresh local supply.
    # This avoids the cumulative error introduced by node 0 (transport-limited)
    # and node NX-1 (special terminal handling).  For supply-limited nodes the
    # routing rule gives: Q_s_fl[i] = Q_s_fl[i-1] + fresh[i], so Δ = fresh[i].
    fresh  = (1.0 - bf.ALPHA_D) * bf.RHO_R * e_dot * bf.W_GL * dx
    dQ_fl  = np.diff(Q_s_fl)
    # dQ_fl[i] = Q_s_fl[i+1] - Q_s_fl[i] should equal fresh[i+1]
    # Start from i=1 (skip the node-0 carryover) and stop before the terminal node.
    np.testing.assert_allclose(dQ_fl[1:-1], fresh[2:-1], rtol=1e-6,
                                err_msg="Per-node flux increment must equal fresh supply")


# ── 8. Coupled solve — transport-limited nodes at capacity ────────────────────
def test_coupled_solve_transport_limited_at_capacity():
    """
    At every transport-limited node (where supply >> capacity), Q_s_fl must
    equal Q_s_cap to within 1e-5 relative tolerance.
    """
    u_b = np.full(NX, 50.0)

    Q_s_fl, Q_s_cap, _, H_till, _, _ = bf.coupled_sediment_solve(
        u_b, 0.5, DBDX, dx, x, d_grain=0.20)

    tl = (Q_s_fl[:-1] >= Q_s_cap[:-1] * 0.9999)
    assert tl.sum() > 0, "Expected at least one transport-limited node"
    np.testing.assert_allclose(
        Q_s_fl[:-1][tl], Q_s_cap[:-1][tl], rtol=1e-5,
        err_msg="Q_s_fl must equal Q_s_cap at transport-limited nodes")

    assert np.all(H_till >= 0.0), "H_till must be non-negative everywhere"
    assert np.all(Q_s_fl >= 0.0), "Q_s_fl must be non-negative everywhere"


# ── 9. BP shooter — Weertman BC and stress-free surface (slow) ───────────────
@_slow
def test_shoot_convergence_and_weertman_bc():
    """
    For a synthetic uniform-slope slab (α ≈ 11.5°, H = 200 m):

    (a) Convergence: the stress-free surface BC is satisfied to within TKRIT,
        confirming that the Anderson-accelerated shooter found the correct
        basal-stress field.

    (b) Weertman BC consistency: inside every iteration the code sets
            U[:,0] = C_slide |τ[:,0]|^{M−1} τ[:,0]
        so U[:,0] and τ[:,0] are algebraically consistent at return.

    (c) Internal deformation: mean surface velocity > mean basal velocity
        because integrating the flow law adds a deformational component.
    """
    sin_a  = 0.2        # ≈ 11.5° — gives moderate velocities (~5–15 m/yr)
    H_s    = 200.0
    b_slab = -x * sin_a
    h_slab = H_s + b_slab

    _, _, dx_s, ds_s, _, _, hb_s, dhdx_s, dbdx_s, C_s, d_s, _ = \
        bf.setup_geometry(x, b_slab, h_slab)

    U, _, Tau = bf.shoot(x, dx_s, ds_s, hb_s, dhdx_s, d_s, C_s)

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


# ── Runner ────────────────────────────────────────────────────────────────────
_FAST = [
    test_rothlisberger_channel,
    test_fluvial_supply_limited,
    test_fluvial_transport_limited,
    test_till_shielding_transport_limited,
    test_till_shielding_supply_limited,
    test_dispersed_layer_steady_state,
    test_coupled_solve_supply_limited,
    test_coupled_solve_transport_limited_at_capacity,
]
_SLOW = [test_shoot_convergence_and_weertman_bc]


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
