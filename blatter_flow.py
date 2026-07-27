"""
Blatter-Pattyn first-order glacier flow model with coupled subglacial sediment transport.

Solves the first-order momentum equations (Blatter 1995; Pattyn 2003) along a 1-D
flowline via shooting with Anderson acceleration.  Sediment transport combines:
  - Engelund-Hansen bedload in a Röthlisberger channel
  - viscoplastic till remobilisation (Boulton & Hindmarsh 1987)
  - dispersed-layer advection (Pierce et al. 2026)

Units: bar, yr, m throughout.
Data files expected in data/; output figures written to the script directory.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

plt.rcParams['font.size'] = 9

_HERE    = Path(__file__).parent
DATA_DIR = _HERE / 'data'
OUT_DIR  = _HERE

AROLLA_FILE     = DATA_DIR / 'arolla.dat'
FRANZJOSEF_FILE = DATA_DIR / 'franzjosef' / 'Bedrock_Anderson2014.csv'
GORNER_FILE     = DATA_DIR / 'gorner' / 'gorner.dat'
QW_FJ           = 50.0   # annual-mean water discharge Franz Josef (m³/s; high rainfall)
QW_GORNER       = 10.0   # annual-mean water discharge Gornergletscher (m³/s)

RHO_G   = 900.0 * 9.81 * 1e-5   # bar/m
A_RATE  = 0.08                    # bar^-3 yr^-1
N_GL    = 3.0
T0      = 0.1                     # bar^2, regularisation
BETA    = 0.004                   # shooting relaxation (fallback / Anderson damping)
TKRIT   = 1e-3                    # convergence criterion
MAXITER = int(1e5)
AA_M    = 10                      # Anderson acceleration history depth
AA_REG  = 1e-4                    # Tikhonov regularisation for Anderson lstsq

C_SLIDE = 5.0                     # Weertman coefficient  (m yr^-1 bar^-m)
M_SLIDE = 3                       # Weertman exponent (3 = Glen flow law)
QW_REF  = 1.0                     # reference annual mean water discharge (m³/s)

K_G        = 1e-4                  # erosion coefficient (m^(1-l) yr^(l-1))
L_EXP      = 1.0                   # erosion exponent
H_TILL_REF = 0.5                   # e-folding till thickness for bedrock shielding (m)
SHIELD_MIN = 0.1                   # minimum shielding factor (residual bedrock wear under thick till)
# Q_deform = C_DEF_TILL * τ_b * H_till * W * ρ_B  (Boulton & Hindmarsh 1987)
# Field range 0.1–5 m yr⁻¹ bar⁻¹; outputs (shielding, T) are insensitive because
# shield → SHIELD_MIN for H_till > 3 m, well before H saturates.
C_DEF_TILL = 1.0                   # till deformation coefficient [m yr⁻¹ bar⁻¹]

RHO_R   = 2700.0                  # bedrock density (kg m^-3)
RHO_B   = 1600.0                  # bulk sediment density (kg m^-3)
RHO_ICE = 900.0                   # ice density (kg m^-3)
W_GL    = 500.0                   # glacier width (m), constant

# Dispersed sediment layer (storage model, Pierce et al. 2026)
C_D      = 0.03    # dispersed layer sediment mass fraction
ALPHA_D  = 0.1     # fraction of eroded material entering dispersed layer
M_BASAL  = 0.05    # basal melt rate of dispersed layer (m ice/yr)
T_STOR   = 3000.0  # max integration time for storage spin-up (yr)
TOL_STOR = 1e-5    # convergence: relative change in Q_s_outlet per yr

# Subglacial fluvial transport (Engelund-Hansen / Delaney et al. 2019 JGR-ES)
D_GRAIN  = 0.01     # coarse reference grain diameter (m): fine gravel / bedload
D_FINE   = 63e-6    # fine reference grain diameter (m): silt/sand boundary (Wentworth),
                    # representative of the suspended fraction retained by 0.22 um
                    # filtration in the Franz Josef / Gornergletscher field campaigns
RHO_W    = 1000.0  # water density (kg m^-3)
RHO_SED  = 2650.0  # sediment grain density (kg m^-3), quartz
N_MANN   = 0.03    # Manning roughness (m^{-1/3} s), subglacial R-channel
THETA_C  = 0.047   # critical Shields parameter for motion initiation

NX = 79
NZ = 20


def load_arolla():
    d  = np.loadtxt(AROLLA_FILE)
    return d[:, 0], d[:, 1], d[:, 2]


def load_franzjosef():
    """
    Franz Josef Glacier flowline geometry (Anderson et al. 2014).

    CSV: cols 0-1 = bedrock (x, z a.s.l.); col 2 = blank; cols 3-4 = ice surface (x, z).
    Both profiles were digitised at different x positions.  Ice surface is interpolated
    onto the bedrock x grid and returned for setup_geometry to resample to NX points.
    """
    data  = np.genfromtxt(FRANZJOSEF_FILE, delimiter=',', skip_header=1,
                          filling_values=np.nan)
    x_bed = data[:, 0]
    z_bed = data[:, 1]
    mask  = ~np.isnan(data[:, 3])
    x_ice = data[mask, 3]
    z_ice = data[mask, 4]

    x_min = max(x_bed[0], x_ice[0])
    x_max = min(x_bed[-1], x_ice[-1])
    keep  = (x_bed >= x_min) & (x_bed <= x_max)
    xx    = x_bed[keep]
    bb    = z_bed[keep]
    hh    = np.maximum(interp1d(x_ice, z_ice, kind='linear')(xx), bb)
    return xx, bb, hh



def load_gorner():
    d = np.loadtxt(GORNER_FILE)
    return d[:, 0], d[:, 1], d[:, 2]


def setup_geometry(xx, bb, hh):
    x   = np.linspace(xx[0], xx[-1], NX)
    sig = np.linspace(0.0, 1.0, NZ)
    dx  = x[1] - x[0]
    ds  = sig[1] - sig[0]

    b   = interp1d(xx, bb, kind='cubic')(x)
    h   = interp1d(xx, hh, kind='cubic')(x)
    hb  = np.maximum(h - b, 0.0)   # cubic spline can undershoot → negative H
    h   = b + hb                    # keep surface consistent

    dhdx = np.gradient(h, x)
    dbdx = np.gradient(b, x)

    # BP solver needs zero slope at the end nodes (stress-free boundary).
    # Fluvial transport must use the physical gradient, so we zero only the
    # local copies used for the σ-coordinate geometry factors C and d.
    dhdx_bc        = dhdx.copy(); dhdx_bc[[0, -1]] = 0.0
    dbdx_bc        = dbdx.copy(); dbdx_bc[[0, -1]] = 0.0

    C = np.outer(dbdx_bc, (1.0 - sig)) + np.outer(dhdx_bc, sig)  # (NX, NZ)
    d = dhdx_bc - dbdx_bc                                          # (NX,)

    Z_o = b[:, None] + np.outer(hb, sig)                    # (NX, NZ) real z

    return x, sig, dx, ds, b, h, hb, dhdx, dbdx, C, d, Z_o


def newton_sigma_vec(s, c, g, tau, n_iter=30):
    s = s.copy()
    for _ in range(n_iter):
        t_xz = tau + 2.0*c*s
        eff2 = t_xz**2 + s**2
        base = eff2 + T0
        F    = base**((N_GL - 1.0) / 2.0)
        G    = -2.0*c*tau - (1.0 + 4.0*c**2)*s
        y    = g + A_RATE * F * G
        dF   = ((N_GL-1.0)/2.0) * base**((N_GL-3.0)/2.0) * (4.0*t_xz*c + 2.0*s)
        dy   = A_RATE * (dF*G + F*(-(1.0 + 4.0*c**2)))
        step = np.where(np.abs(dy) > 1e-30, y / dy, 0.0)
        s   -= np.clip(step, -0.5, 0.5)
    return np.clip(s, -5.0, 5.0)


def solve_sigma_row(Sigma_j, C_j, U_j, x, Tau_j):
    dudx       = np.gradient(U_j, x)
    S          = newton_sigma_vec(Sigma_j, C_j, dudx, Tau_j)
    S[[0, -1]] = 0.0
    return S


def integrate(U, Sigma, Tau, hb, dhdx, d, C, ds, x, use_sigma=True):
    for j in range(NZ - 1):
        S_j  = Sigma[:, j] if use_sigma else np.zeros(NX)
        t_xz = Tau[:, j] + 2.0*C[:, j]*S_j
        ff   = A_RATE * (S_j**2 + t_xz**2 + T0)
        dudz = ff * (2.0*Tau[:, j] + 4.0*S_j*C[:, j]) * hb
        dSdx = np.gradient(S_j, x); dSdx[[0,-1]] = 0.0
        dtdz = RHO_G*dhdx*hb - dSdx*hb - 2.0*d*S_j

        up = U[:, j] + ds*dudz
        tp = Tau[:, j] + ds*dtdz

        sk = solve_sigma_row(S_j, C[:, j+1], up, x, tp) if use_sigma else np.zeros(NX)

        t_xz = 2.0*C[:, j+1]*sk + tp
        ff   = A_RATE * (sk**2 + t_xz**2 + T0)
        dudz = ff * (2.0*tp + 4.0*sk*C[:, j+1]) * hb
        dSkdx = np.gradient(sk, x); dSkdx[[0,-1]] = 0.0
        dtdz = RHO_G*dhdx*hb - dSkdx*hb - 2.0*d*sk

        U[:, j+1]   = 0.5*(up + U[:, j] + ds*dudz)
        Tau[:, j+1] = 0.5*(tp + Tau[:, j] + ds*dtdz)

        if use_sigma:
            Sigma[:, j+1] = solve_sigma_row(sk, C[:, j+1],
                                             U[:, j+1], x, Tau[:, j+1])
    return U, Sigma, Tau


def anderson_update(x_hist, f_hist, x_k, f_k, m=AA_M, beta=BETA):
    """
    Anderson(m) acceleration for fixed-point x = g(x).
    x_k : current iterate, f_k : residual r(x_k) that we want → 0.
    Stores history in-place; returns the next iterate.
    The Picard step is x - beta*r(x); Anderson finds the optimal
    linear combination of the last m Picard steps.
    """
    x_hist.append(x_k.copy())
    f_hist.append(f_k.copy())
    if len(x_hist) > m + 1:
        x_hist.pop(0)
        f_hist.pop(0)

    mk = len(x_hist) - 1
    if mk == 0:
        return x_k - beta * f_k

    dF = np.diff(np.column_stack(f_hist), axis=1)   # (NX, mk)
    dX = np.diff(np.column_stack(x_hist), axis=1)   # (NX, mk)

    DtD = dF.T @ dF
    reg = AA_REG * (np.trace(DtD) / mk) * np.eye(mk)
    gamma = np.linalg.solve(DtD + reg, dF.T @ f_hist[-1])
    return x_hist[-1] - beta * f_hist[-1] - (dX - beta * dF) @ gamma


def shoot(x, dx, ds, hb, dhdx, d, C):
    U     = np.zeros((NX, NZ))
    Sigma = np.zeros((NX, NZ))
    Tau   = np.zeros((NX, NZ))
    Tau[:, 0] = -RHO_G * hb * dhdx

    x_hist, f_hist = [], []
    for nkk in range(MAXITER):
        U[:, 0] = C_SLIDE * np.abs(Tau[:, 0])**(M_SLIDE - 1) * Tau[:, 0]
        Sigma[:, 0] = solve_sigma_row(Sigma[:, 0], C[:, 0], U[:, 0], x, Tau[:, 0])
        U, Sigma, Tau = integrate(U, Sigma, Tau, hb, dhdx, d, C, ds, x)

        residual = Tau[:, -1].copy()
        tmax = np.max(np.abs(residual))

        Tau[:, 0] = anderson_update(x_hist, f_hist, Tau[:, 0], residual)

        if nkk % 50 == 0:
            print(f'  iter {nkk:6d},  tmax = {tmax:.6f} bar')
        if tmax < TKRIT:
            print(f'  Converged after {nkk} iterations,  tmax = {tmax:.6f} bar')
            break

    return U, Sigma, Tau


def fluvial_transport(Q_w_total, e_dot, dbdx, dx, x, d_grain=D_GRAIN, dhdx=None):
    """
    Subglacial fluvial bedload transport (Delaney, Werder & Farinotti 2019, JGR-ES).

    Engelund-Hansen formula applied to a Röthlisberger semicircular R-channel
    whose geometry is determined by Manning's equation.

    Water discharge accumulates linearly: Q_w(x) = Q_w_total * x / L.

    Hydraulic slope:
      default (dhdx=None): S = |∂b/∂x|  (bed slope only)
      hydraulic potential  (dhdx provided): Shreve (1972) hydraulic head gradient
          h_w = (1 - ρ_i/ρ_w) b + (ρ_i/ρ_w) h
          S   = -dh_w/dx = -(1-ρ_i/ρ_w)·∂b/∂x - (ρ_i/ρ_w)·∂h/∂x
      floored at 1e-4 in both cases (channels stall but do not reverse).
      In overdeepenings S_hyd → 0, collapsing the transport capacity.

    Channel radius from Manning:
        Q_w = (1/n) * (π r²/2) * (r/2)^(2/3) * S^(1/2)
        r   = (2^(5/3) * n * Q_w / (π √S))^(3/8)

    Shields parameter:  θ = ρ_w R_h S / ((ρ_s - ρ_w) d)
    Engelund-Hansen:    Φ = 0.05 (θ - θ_c)^(5/2),  q_s = Φ √((s-1) g d³)
    Capacity (kg/yr):   Q_s^cap = ρ_s * q_s * 2r * SPY

    Parameters
    ----------
    d_grain : grain diameter (m); controls Shields number and settling velocity
    dhdx    : (NX,) ice-surface slope ∂h/∂x; if given, use Shreve hydraulic slope

    Returns
    -------
    Q_s_fl  : (NX,) actual cumulative sediment flux (kg/yr) — min(cap, supply)
    Q_s_cap : (NX,) transport capacity (kg/yr)
    Q_s_sup : (NX,) erosional supply available to fluvial system (kg/yr)
    """
    SPY = 365.25 * 24 * 3600                      # seconds per year

    Q_w = np.maximum(Q_w_total * x / x[-1], 1e-10)
    if dhdx is not None:
        f_rho = RHO_ICE / RHO_W                   # ≈ 0.9
        S_hyd = -(1.0 - f_rho) * dbdx - f_rho * dhdx   # Shreve hydraulic gradient
        S = np.maximum(S_hyd, 1e-4)
    else:
        S = np.maximum(np.abs(dbdx), 1e-4)        # bed-slope approximation

    r   = (2.0**(5.0/3.0) * N_MANN * Q_w / (np.pi * np.sqrt(S)))**0.375
    R_h = r / 2.0

    theta   = RHO_W * R_h * S / ((RHO_SED - RHO_W) * d_grain)
    Phi     = 0.05 * np.maximum(theta - THETA_C, 0.0)**2.5
    q_s     = Phi * np.sqrt((RHO_SED / RHO_W - 1.0) * 9.81 * d_grain**3)  # m²/s
    Q_s_cap = RHO_SED * q_s * 2.0 * r * SPY      # kg/yr

    # erosion supply to fluvial system: (1-α_d) fraction of total erosion
    Q_s_sup = np.cumsum((1.0 - ALPHA_D) * RHO_R * e_dot * W_GL) * dx  # kg/yr

    Q_s_fl = np.minimum(Q_s_cap, Q_s_sup)
    return Q_s_fl, Q_s_cap, Q_s_sup


def till_shielding(u_b, Q_s_cap, dx):
    """
    Steady-state subglacial till thickness and bedrock shielding factor.

    When the fluvial system cannot evacuate all eroded material, till accumulates
    on the bed and shields the bedrock from further abrasion.  Shielding is
    modelled as an exponential decay with e-folding scale H_TILL_REF:

        e_dot_eff = K_G * |u_b|^L * exp(-H_till / H_TILL_REF)

    At steady state, shielded production equals transport removal per unit area:

        RHO_R * K_G * |u_b|^L * exp(-H_till/H_ref) = Q_s_cap / (W_GL * dx)

    Solving analytically:
        H_till = max(0,  H_TILL_REF * ln(prod / removal))
        shield = exp(-H_till / H_TILL_REF) = min(removal / prod, 1)

    where  prod    = RHO_R * K_G * |u_b|^L   [kg m^-2 yr^-1]
           removal = Q_s_cap / (W_GL * dx)   [kg m^-2 yr^-1]

    Q_s_cap is hydraulically determined (independent of e_dot), so no iteration
    is needed.

    Parameters
    ----------
    u_b     : (NX,) basal velocity (m/yr)
    Q_s_cap : (NX,) local transport capacity (kg/yr) from fluvial_transport
    dx      : grid spacing (m)

    Returns
    -------
    H_till : (NX,) steady-state till thickness (m)
    shield : (NX,) bedrock shielding factor in [0, 1]
    """
    # Only the (1-alpha_d) fraction not entering the dispersed layer arrives at
    # the subglacial channel and can accumulate as till.
    prod       = (1.0 - ALPHA_D) * RHO_R * K_G * np.abs(u_b)**L_EXP  # kg m^-2 yr^-1
    removal    = Q_s_cap / (W_GL * dx)                                 # kg m^-2 yr^-1
    shield_raw = np.minimum(removal / (prod + 1e-30), 1.0)
    shield     = SHIELD_MIN + (1.0 - SHIELD_MIN) * shield_raw
    # When removal → 0 (overdeepening), shield_raw → 0 and H_till → ∞.
    # The +1e-30 floor caps initialization at ~34.5 m; coupled_sediment_solve
    # subsequently replaces this with a physically finite value via till
    # deformation (Newton-Raphson), so the cap is only a warm-start artefact.
    H_till     = -H_TILL_REF * np.log(shield_raw + 1e-30)
    H_till     = np.where(shield_raw < 1.0, H_till, 0.0)
    return H_till, shield


def coupled_sediment_solve(u_b, Qw, dbdx, dx, x,
                            d_grain=D_GRAIN, dhdx=None,
                            max_iter=30, tol=1e-3):
    """
    Coupled till-shielding / fluvial-transport solve with till remobilisation.

    Replaces the three-step pattern
        fluvial_transport(ones) → till_shielding → fluvial_transport(e_dot)
    with an iterative sequential routing scheme:

      * Transport-limited nodes (running flux ≥ capacity): channel runs at
        capacity; equilibrium H_till is set so that the shielded local
        production fills exactly the remaining capacity after the upstream
        running flux — correctly accounting for the upstream context instead
        of treating each node in isolation as till_shielding() does.

      * Supply-limited nodes (running flux < capacity): any stored till is
        remobilised by the channel until capacity is reached or the till is
        exhausted; H_till decreases accordingly.

    H_till and shielding are updated after each routing pass and the solve
    repeats until H_till converges.

    Returns
    -------
    Q_s_fl  : (NX,) actual cumulative sediment flux (kg/yr)
    Q_s_cap : (NX,) transport capacity (kg/yr)
    Q_deform: (NX,) till-deformation flux leaving each node (kg/yr)
    H_till  : (NX,) converged till thickness (m)
    e_dot   : (NX,) shielded erosion rate (m/yr)
    shield  : (NX,) bedrock shielding factor
    """
    NX = len(u_b)

    # Hydraulic capacity — independent of erosion rate
    _, Q_s_cap, _ = fluvial_transport(Qw, np.ones(NX), dbdx, dx, x,
                                       d_grain=d_grain, dhdx=dhdx)

    # Unshielded production per unit area [kg m⁻² yr⁻¹]
    prod = (1.0 - ALPHA_D) * RHO_R * K_G * np.abs(u_b)**L_EXP

    # Basal shear stress from Weertman inverse: u_b = C_SLIDE * tau_b^M_SLIDE
    tau_b = np.maximum((np.abs(u_b) / (C_SLIDE + 1e-30))**(1.0 / M_SLIDE), 0.01)
    # Per-node till-deformation stiffness B_def [kg yr⁻¹ m⁻¹]: Q_deform = B_def * H_till
    B_def = RHO_B * C_DEF_TILL * tau_b * W_GL

    # Initialise from the local-equilibrium (node-isolated) estimate
    H_till, shield = till_shielding(u_b, Q_s_cap, dx)

    for _ in range(max_iter):
        e_dot = K_G * np.abs(u_b)**L_EXP * shield

        Q_s_fl     = np.zeros(NX)
        H_till_new = np.zeros(NX)
        Q_deform   = np.zeros(NX)
        Q_run      = 0.0

        for i in range(NX):
            fresh_i   = (1.0 - ALPHA_D) * RHO_R * e_dot[i] * W_GL * dx  # kg/yr
            available = Q_run + fresh_i

            # The terminal node has u_b → 0 (BC), making B_def → 0 and the NR
            # result unphysical.  Treat it as supply-limited (same convention as
            # using Q_s_fl[-2] for the outlet flux).
            if i == NX - 1:
                Q_s_fl[i]     = min(available, Q_s_cap[i])
                H_till_new[i] = 0.0
                Q_run = Q_s_fl[i]
                continue

            if available >= Q_s_cap[i]:
                # Transport-limited: till builds up until till deformation drains
                # the excess supply.  Steady-state balances:
                #   Q_run + supply_shielded(H) = Q_s_cap[i] + B_def[i] * H
                # Rearranged as f(H) = B_def[i]*H - residual(H) = 0 and solved
                # by Newton-Raphson (f is monotone increasing; guaranteed root).
                Q_s_fl[i]  = Q_s_cap[i]
                sup_coeff  = prod[i] * W_GL * dx   # unshielded supply (kg/yr)
                H = max(H_till[i], 0.0)            # warm-start from previous iter
                for _ in range(12):
                    sh_H = SHIELD_MIN + (1.0 - SHIELD_MIN) * np.exp(-H / H_TILL_REF)
                    f    = B_def[i] * H - (Q_run + sup_coeff * sh_H - Q_s_cap[i])
                    fp   = B_def[i] + sup_coeff * (1.0 - SHIELD_MIN) / H_TILL_REF * np.exp(-H / H_TILL_REF)
                    H    = max(0.0, H - f / fp)
                H_till_new[i] = H
                Q_deform[i]   = B_def[i] * H
            else:
                # Supply-limited: remobilise stored till to reach capacity.
                excess         = Q_s_cap[i] - available
                till_mass      = RHO_B * H_till[i] * W_GL * dx   # kg (instantaneous)
                remob          = min(excess, till_mass)
                Q_s_fl[i]      = available + remob
                H_till_new[i]  = max(0.0,
                                     H_till[i] - remob / (RHO_B * W_GL * dx))

            # Channel flux propagates downstream; Q_deform is a local till-layer
            # drain (limits H_till) but does not re-enter the channel immediately.
            Q_run = Q_s_fl[i]

        # Update shielding from new till thickness
        shield_raw_new = np.exp(-H_till_new / H_TILL_REF)
        shield_raw_new[H_till_new <= 0.0] = 1.0
        shield_new = SHIELD_MIN + (1.0 - SHIELD_MIN) * shield_raw_new

        delta = np.max(np.abs(H_till_new - H_till))
        H_till = H_till_new
        shield = shield_new
        if delta < tol:
            break

    e_dot = K_G * np.abs(u_b)**L_EXP * shield
    return Q_s_fl, Q_s_cap, Q_deform, H_till, e_dot, shield


def evolve_storage(u_b, e_dot, dx,
                   m_basal=M_BASAL, alpha_d=ALPHA_D,
                   t_max=T_STOR, tol=TOL_STOR):
    """
    1-D upwind advection of dispersed-layer thickness H_d (m ice-equivalent).

    Governing equation (Pierce et al. 2026, Eq. 5, 1-D):
        dH_d/dt + u_b * dH_d/dx = source(x) - sink(x)

    Source: entrainment of eroded material into dispersed layer
        source = alpha_d * K_ENT * e_dot,   K_ENT = rho_R / (rho_ice * C_d)
    Sink: basal melt releases sediment downward out of dispersed layer
        sink   = m_basal * H_d / H_d_ref   (first-order decay, H_d_ref=1 m)

    Upwind finite difference, CFL-limited time step.
    Inflow BC at x[0]: H_d = 0  (no dispersed layer at bergschrund).
    Outflow at x[-1]: free (upwind naturally gives zero gradient).

    Returns
    -------
    H_d   : (NX,) steady-state dispersed-layer thickness (m)
    Qs_outlet_series : list of (t, Q_s_outlet) for convergence diagnostic
    dS_dt : (NX,) ∂S/∂t at steady state (m/yr, <0 means release)
              = -m_basal * H_d  (negative = sediment exits storage)
    """
    K_ENT  = RHO_R / (RHO_ICE * C_D)          # dimensionless entrainment factor
    source = alpha_d * K_ENT * e_dot            # m/yr

    u_max  = np.max(np.abs(u_b)) + 1e-10
    dt_cfl = 0.9 * dx / u_max                   # CFL condition

    n   = len(u_b)
    H_d = np.zeros(n)
    t   = 0.0
    Qs_prev   = 0.0
    Qs_series = []

    while t < t_max:
        dt = min(dt_cfl, t_max - t)

        # upwind advection (u_b > 0 everywhere along flowline)
        flux        = u_b * H_d
        dflux_dx    = np.zeros(NX)
        dflux_dx[1:]  = (flux[1:] - flux[:-1]) / dx   # upwind
        dflux_dx[0]   = flux[0] / dx                   # inflow = 0, so flux[0]/dx

        sink        = m_basal * H_d                    # m/yr per unit H_d (1/yr × H_d)
        dHdt        = source - dflux_dx - sink
        H_d         = np.maximum(H_d + dt * dHdt, 0.0)
        H_d[0]      = 0.0                              # Dirichlet BC at bergschrund

        t += dt

        # sediment exiting the glacier (kg/yr):
        #   (1) integrated melt-out along flowline → enters subglacial drainage
        #   (2) advective export at terminus
        Qs_out = RHO_ICE * C_D * W_GL * (
            m_basal * np.sum(H_d) * dx   # (1)
            + u_b[-1] * H_d[-1]          # (2)
        )

        if t > 0:
            Qs_series.append((t, Qs_out))
            rel_change = abs(Qs_out - Qs_prev) / (abs(Qs_prev) + 1e-30) / dt
            if rel_change < tol and t > 100.0:
                print(f'  Storage spin-up converged at t={t:.1f} yr  '
                      f'(rel_change={rel_change:.2e})', flush=True)
                break
            Qs_prev = Qs_out

    dS_dt = -m_basal * H_d    # negative: melt releases sediment (m/yr)
    return H_d, Qs_series, dS_dt


def check_mass_conservation(e_dot, Q_s_fl, Qs_disp, H_d, dx,
                            label='', tol=0.01):
    """
    Print a sediment mass budget and check conservation at steady state.

    At steady state the total erosional flux must equal the sum of:
      (1) fluvial outlet        Q_s_fl[-2]  (penultimate node; avoids terminal slope artefact)
      (2) till storage          (1-alpha_d)*Q_ero - Q_s_fl[-2]   [>= 0]
      (3) dispersed outlet      Qs_disp
      (4) dispersed storage     RHO_ICE*C_D*W*integral(H_d)dx    [should be ~0 change]

    The dispersed layer is at steady state when Qs_disp ≈ alpha_d * Q_ero, so
    items (1)+(2)+(3) should sum to Q_ero within the dispersed-layer convergence
    tolerance.

    Parameters
    ----------
    tol : relative tolerance for the residual (default 1 %)

    Returns
    -------
    True if mass is conserved within tol, False otherwise.
    """
    Q_ero            = RHO_R * W_GL * np.sum(e_dot) * dx          # total eroded (kg/yr)

    Q_fluvial_supply = (1.0 - ALPHA_D) * Q_ero                    # available to channel
    Q_fl_out         = Q_s_fl[-2]                                  # exits at front (penultimate node)
    Q_till           = Q_fluvial_supply - Q_fl_out                 # stored as till

    Q_disp_input     = ALPHA_D * Q_ero                             # enters dispersed layer
    Q_disp_out       = Qs_disp                                     # exits dispersed layer
    Q_disp_stored    = RHO_ICE * C_D * W_GL * np.sum(H_d) * dx    # currently in layer (kg)
    disp_imbalance   = Q_disp_input - Q_disp_out                   # >0 if still filling

    # Full budget: eroded = fluvial_out + till_stored + dispersed_out + dispersed_imbalance
    Q_accounted      = Q_fl_out + Q_till + Q_disp_out + disp_imbalance
    residual         = Q_accounted - Q_ero
    rel_err          = abs(residual) / (Q_ero + 1e-30)

    tag = f'  [{label}] ' if label else '  '
    print(f'\n{tag}--- Mass conservation budget ---')
    print(f'{tag}  Total eroded            : {Q_ero:.3e} kg/yr')
    print(f'{tag}  Fluvial supply (1-αd)   : {Q_fluvial_supply:.3e} kg/yr')
    print(f'{tag}    → fluvial outlet      : {Q_fl_out:.3e} kg/yr  '
          f'({Q_fl_out/Q_ero*100:.1f} % of ero)')
    print(f'{tag}    → till storage        : {Q_till:.3e} kg/yr  '
          f'({Q_till/Q_ero*100:.1f} % of ero)'
          + ('  *** NEGATIVE — check shielding ***' if Q_till < -1e-6 * Q_ero else ''))
    print(f'{tag}  Dispersed input (αd)    : {Q_disp_input:.3e} kg/yr')
    print(f'{tag}    → dispersed outlet    : {Q_disp_out:.3e} kg/yr  '
          f'({Q_disp_out/Q_ero*100:.1f} % of ero)')
    print(f'{tag}    → dispersed imbalance : {disp_imbalance:.3e} kg/yr  '
          f'(steady-state residual)')
    print(f'{tag}  Dispersed layer storage : {Q_disp_stored:.3e} kg  (total mass in layer)')
    print(f'{tag}  Residual (should be 0)  : {residual:.3e} kg/yr  '
          f'(rel. error = {rel_err*100:.4f} %)')
    ok = rel_err < tol and Q_till >= -1e-6 * Q_ero
    print(f'{tag}  Conservation check      : {"PASS" if ok else "FAIL"}'
          f'  (tol = {tol*100:.1f} %)')
    return ok


def run_glacier_diagnostics(d_grain=D_GRAIN, tag=None):
    """
    Solve the Blatter-Pattyn flow and coupled sediment model for the three
    real diagnostic glaciers (Arolla, Franz Josef, Gornergletscher) and plot
    them side by side in one figure: rows are panel type (velocity profile,
    2D velocity field, erosion rate, dispersed-layer thickness, sediment
    fluxes), columns are glaciers.

    d_grain : representative grain diameter (m) used for the erosion/till/
        fluvial panels (rows c, d, e); the velocity panels (a, b) do not
        depend on grain size.
    tag     : filename suffix (default: derived from d_grain).
    """
    if tag is None:
        tag = f'{d_grain*1000:.2f}mm'.replace('.', 'p') if d_grain >= 1e-4 \
            else f'{d_grain*1e6:.0f}um'

    glaciers = [
        dict(name="Haut Glacier d'Arolla", loader=load_arolla, Qw=QW_REF,
             label='Arolla reference', vmax=20.0),
        dict(name='Franz Josef Glacier', loader=load_franzjosef, Qw=QW_FJ,
             label='Franz Josef reference', vmax=None),
        dict(name='Gornergletscher', loader=load_gorner, Qw=QW_GORNER,
             label='Gornergletscher reference', vmax=None),
    ]

    fig, axes = plt.subplots(5, 3, figsize=(15, 17))

    for col, g in enumerate(glaciers):
        print(f"\nSolving Blatter-Pattyn flow ({g['name']})...")
        xx, bb, hh = g['loader']()
        x, sig, dx, ds, b, h, hb, dhdx, dbdx, C, d, Z_o = setup_geometry(xx, bb, hh)
        U, Sigma, Tau = shoot(x, dx, ds, hb, dhdx, d, C)
        u_s, u_b = U[:, -1], U[:, 0]
        L = x[-1]

        Q_s_fl, Q_s_cap, _, H_till, e_dot, shield = \
            coupled_sediment_solve(u_b, g['Qw'], dbdx, dx, x, d_grain=d_grain, dhdx=dhdx)

        print('Spinning up dispersed-layer storage model...')
        H_d, _, _ = evolve_storage(u_b, e_dot, dx)

        Q_s_ero      = RHO_R * W_GL * np.cumsum(e_dot) * dx
        Qs_total_ero = Q_s_ero[-1]
        Qs_disp      = RHO_ICE * C_D * W_GL * (
            M_BASAL * np.sum(H_d) * dx + u_b[-1] * H_d[-1]
        )
        Qs_fl_out    = Q_s_fl[-2]
        Qs_total_out = Qs_fl_out + Qs_disp
        transfer     = Qs_total_out / (Qs_total_ero + 1e-30)

        alpha_mean = float(np.mean(np.abs(dhdx[dhdx != 0])))
        H_mean     = float(np.mean(hb[hb > 1.0]))
        print(f'  Mean surface slope:     {np.degrees(alpha_mean):.1f}°')
        print(f'  Mean ice thickness:     {H_mean:.0f} m')
        print(f'  Max surface speed:      {u_s.max():.0f} m/yr')
        print(f'  Total erosional flux:   {Qs_total_ero:.3e} kg/yr')
        print(f'  Fluvial outlet:         {Qs_fl_out:.3e} kg/yr')
        print(f'  Dispersed outlet:       {Qs_disp:.3e} kg/yr')
        print(f'  Total transfer ratio:   {transfer:.3f}')
        print(f'  Max H_d:                {np.max(H_d):.4f} m')
        print(f'  Max H_till:             {np.max(H_till):.4f} m  '
              f'(shielded frac: {1.0 - np.mean(shield):.3f})')
        check_mass_conservation(e_dot, Q_s_fl, Qs_disp, H_d, dx, label=g['label'])

        # (a) velocity profile
        ax = axes[0, col]
        ax.plot(x, u_s, 'k-', lw=1.5, label='surface')
        ax.plot(x, u_b, 'b--', lw=1.2, label='basal')
        ax.set_title(f"{g['name']}\n" + r'$Q_w$ = ' + f"{g['Qw']:.0f}" + r' m$^3$ s$^{-1}$',
                     fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel('Velocity (m/yr)')

        # (b) 2D velocity field
        ax = axes[1, col]
        v_max = g['vmax'] if g['vmax'] is not None \
            else max(np.ceil(u_s.max() / 100.0) * 100.0, 100.0)
        vc  = np.linspace(0, v_max, 11)
        X_o = np.tile(x[:, None], (1, NZ))
        cf  = ax.contourf(X_o, Z_o, U, vc, cmap='Blues')
        cax = ax.inset_axes([0.76, 0.55, 0.025, 0.38])
        cb  = plt.colorbar(cf, cax=cax, label='u (m/yr)')
        cb.ax.tick_params(labelsize=7)
        cb.set_label('u (m/yr)', fontsize=7)
        ax.plot(x, b, 'k-', lw=1.2)
        ax.plot(x, h, 'k-', lw=1.2)
        ax.set_xlim(0, L)
        if col == 0:
            ax.set_ylabel('Elevation (m a.s.l.)')
            ax.set_ylim(2400, 3200)

        # (c) erosion rate + till thickness
        ax = axes[2, col]
        e_dot_raw = K_G * np.abs(u_b)**L_EXP
        ax.plot(x, e_dot_raw, 'r--', lw=1.0, alpha=0.5, label='unshielded')
        ax.plot(x, e_dot,     'r-',  lw=1.5,             label='shielded')
        ax.legend(fontsize=7, loc='upper right')
        ax2b = ax.twinx()
        ax2b.plot(x, H_till, 'saddlebrown', lw=1.0, ls=':')
        ax2b.tick_params(axis='y', labelcolor='saddlebrown')
        ax.set_xlim(0, L)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel(r'$\dot{e}$ (m/yr)')
        if col == 2:
            ax2b.set_ylabel(r'$H_\mathrm{till}$ (m)', color='saddlebrown')

        # (d) dispersed layer thickness
        ax = axes[3, col]
        ax.plot(x, H_d, 'g-', lw=1.5)
        ax.set_title(r'$\mathcal{T}$ = ' + f'{transfer:.2f}', fontsize=9)
        ax.set_xlim(0, L)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel(r'$H_d$ (m ice-eq.)')

        # (e) sediment fluxes
        ax = axes[4, col]
        ax.plot(x, Q_s_ero * 1e-3, 'm-',  lw=1.5, label='erosion')
        ax.plot(x, Q_s_fl  * 1e-3, 'b-',  lw=1.5, label='fluvial')
        ax.plot(x, Q_s_cap * 1e-3, 'b--', lw=1.0, alpha=0.5, label='capacity')
        ax.axhline(Qs_disp * 1e-3, color='g', ls='--', lw=1.2, label='dispersed')
        ax.axhline(Qs_total_out * 1e-3, color='k', ls=':', lw=1.5, label='total')
        ax.legend(fontsize=6, loc='upper right')
        ax.set_xlim(0, L)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Distance along flowline (m)', fontsize=8)
        if col == 0:
            ax.set_ylabel(r'$Q_s$ (t/yr)')

    panel_labels = 'abcdefghijklmno'
    for row in range(5):
        for col in range(3):
            lbl = panel_labels[row * 3 + col]
            axes[row, col].text(0.02, 0.97, f'({lbl})', transform=axes[row, col].transAxes,
                                fontsize=10, fontweight='bold', va='top', ha='left',
                                zorder=10,
                                bbox=dict(facecolor='white', edgecolor='none',
                                          alpha=0.85, pad=1.5))

    d_label = f'{d_grain*1e6:.0f} um' if d_grain < 1e-4 else f'{d_grain*1000:.1f} mm'
    fig.suptitle('Blatter-Pattyn first-order model with coupled sediment transport '
                 f'($K_g={K_G}$, $l={L_EXP}$, $H_{{\\mathrm{{ref}}}}={H_TILL_REF}$ m, '
                 f'$\\alpha_d={ALPHA_D}$, $m_b={M_BASAL}$ m/yr, $D={d_label}$) --- '
                 'fluvial (Engelund-Hansen) + dispersed (Pierce et al.) pathways',
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'glacier_diagnostics_{tag}.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUT_DIR / f'glacier_diagnostics_{tag}.pdf', bbox_inches='tight')
    plt.show(block=False)


def compare_glaciers():
    """
    Sediment transfer ratio T = Qs_outlet / Qs_erosion as a function of grain
    size D for three glaciers (one panel per glacier), comparing the Shreve
    hydraulic potential slope against the bed-slope-only approximation.

    The total transfer ratio T (solid line) is split into:
      - fluvial component T_fl (dashed line, darker fill below)
      - dispersed / englacial component T - T_fl (lighter fill above)

    The dispersed contribution is approximately constant at ALPHA_D (~10 %)
    regardless of grain size; the fluvial contribution decreases with D because
    coarser grains require higher transport capacity (steeper slopes or larger Qw).
    The grey dash-dot line repeats the total T using dhdx=None (bed slope
    magnitude only), to show how much the hydraulic-slope choice matters
    across the full grain-size range, not just at the reference grain size.
    """
    D_vals = np.logspace(np.log10(10e-6), np.log10(0.40), 30)
    D_vals = np.sort(np.unique(np.concatenate([D_vals, [D_FINE, D_GRAIN]])))
    D_mm   = D_vals * 1000

    glaciers = [
        ('Arolla',          load_arolla,     QW_REF,    '#1b7837'),
        ('Franz Josef',            load_franzjosef,        QW_FJ,     '#2166ac'),
        ('Gornergletscher', load_gorner,     QW_GORNER, '#b2182b'),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)

    for ax, (name, loader, Qw, col) in zip(axes, glaciers):
        print(f'\nSolving BP for {name}...')
        xx, bb, hh = loader()
        x, _, dx, ds, _, _, hb, dhdx, dbdx, C, d, _ = setup_geometry(xx, bb, hh)
        U, _, _ = shoot(x, dx, ds, hb, dhdx, d, C)
        u_b = U[:, 0]

        T_arr, Tfl_arr, T_bed_arr = [], [], []
        for D in D_vals:
            Q_s_fl, _, _, _, e_dot, _ = coupled_sediment_solve(
                u_b, Qw, dbdx, dx, x, d_grain=D, dhdx=dhdx)
            H_d, _, _ = evolve_storage(u_b, e_dot, dx)
            Q_ero   = RHO_R * W_GL * np.sum(e_dot) * dx
            Qs_disp = RHO_ICE * C_D * W_GL * (
                M_BASAL * np.sum(H_d) * dx + u_b[-1] * H_d[-1])
            Qs_fl = Q_s_fl[-2]
            T_arr.append((Qs_fl + Qs_disp) / (Q_ero + 1e-30))
            Tfl_arr.append(Qs_fl / (Q_ero + 1e-30))

            Q_s_fl_bed, _, _, _, e_dot_bed, _ = coupled_sediment_solve(
                u_b, Qw, dbdx, dx, x, d_grain=D, dhdx=None)
            H_d_bed, _, _ = evolve_storage(u_b, e_dot_bed, dx)
            Q_ero_bed   = RHO_R * W_GL * np.sum(e_dot_bed) * dx
            Qs_disp_bed = RHO_ICE * C_D * W_GL * (
                M_BASAL * np.sum(H_d_bed) * dx + u_b[-1] * H_d_bed[-1])
            T_bed_arr.append((Q_s_fl_bed[-2] + Qs_disp_bed) / (Q_ero_bed + 1e-30))

        T     = np.array(T_arr)
        Tfl   = np.array(Tfl_arr)
        T_bed = np.array(T_bed_arr)

        ax.fill_between(D_mm, 0,   Tfl, color=col, alpha=0.30, label='fluvial')
        ax.fill_between(D_mm, Tfl, T,   color=col, alpha=0.10, label='dispersed (ice)')
        ax.plot(D_mm, T,   color=col, lw=2.0, label='total  $T$  (Shreve)')
        ax.plot(D_mm, Tfl, color=col, lw=1.2, ls='--', label='fluvial only  $T_{fl}$')
        ax.plot(D_mm, T_bed, color='0.35', lw=1.6, ls='-.',
               label='total  $T$  (bed slope only)')
        ax.axhline(ALPHA_D, color='k', lw=0.8, ls=':', alpha=0.5,
                   label=f'dispersed minimum  ({ALPHA_D:.0%})')
        ax.axvline(D_FINE * 1000, color='darkorange', lw=1.0, ls=':', alpha=0.8,
                   label=f'fine ref.  ({D_FINE*1e6:.0f} $\\mu$m)')
        ax.axvline(D_GRAIN * 1000, color='saddlebrown', lw=1.0, ls=':', alpha=0.8,
                   label=f'coarse ref.  ({D_GRAIN*1000:.0f} mm)')

        od_count = int(np.sum(dbdx > 0))
        ax.set_xscale('log')
        ax.set_ylabel('Transfer ratio  $T$')
        ax.set_title(f'{name}  —  $Q_w$ = {Qw:.0f} m³/s  '
                     f'({od_count} overdeepening nodes)', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Grain size  $D$  (mm)')

    for ax, lbl in zip(axes, 'abc'):
        ax.text(0.01, 0.97, f'({lbl})', transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top')

    fig.suptitle(
        r'Sediment transfer ratio vs grain size — '
        r'$T = Q_\mathrm{outlet} / Q_\mathrm{erosion}$' + '\n'
        'fluvial flux evaluated at penultimate node; '
        'dispersed (englacial) component shaded light; '
        'grey line shows the bed-slope-only approximation',
        fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'glacier_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUT_DIR / 'glacier_comparison.pdf', bbox_inches='tight')
    plt.show(block=False)
    print('Comparison figure saved.')


def qw_sensitivity():
    """
    Sensitivity of the sediment Damkoehler number Da_sed and the transfer
    ratio T to water discharge Q_w, at the default grain size, for the three
    real diagnostic glaciers.

    Basal velocity is independent of Q_w in this model (the Weertman sliding
    law depends on stress alone), so each glacier's Blatter-Pattyn solve is
    reused across the whole Q_w sweep; only fluvial_transport / evolve_storage
    are re-run per Q_w value (both use the Shreve hydraulic potential slope).
    """
    Qw_vals = np.logspace(-2, 2.5, 26)

    glaciers = [
        ('Arolla',          load_arolla,     QW_REF,    '#1b7837'),
        ('Franz Josef',     load_franzjosef, QW_FJ,     '#2166ac'),
        ('Gornergletscher', load_gorner,     QW_GORNER, '#b2182b'),
    ]

    fig, (ax_da, ax_T) = plt.subplots(2, 1, figsize=(7, 9), sharex=True)

    for name, loader, Qw_real, col in glaciers:
        print(f'\nSolving BP for {name}...')
        xx, bb, hh = loader()
        x, _, dx, ds, _, _, hb, dhdx, dbdx, C, d, _ = setup_geometry(xx, bb, hh)
        U, _, _ = shoot(x, dx, ds, hb, dhdx, d, C)
        u_b = U[:, 0]

        Da_arr, T_arr = [], []
        for Qw in Qw_vals:
            Q_s_fl, _, _, _, e_dot, _ = coupled_sediment_solve(
                u_b, Qw, dbdx, dx, x, dhdx=dhdx)
            H_d, _, _ = evolve_storage(u_b, e_dot, dx)

            Qs_ero  = RHO_R * W_GL * np.sum(e_dot) * dx
            Qs_disp = RHO_ICE * C_D * W_GL * (
                M_BASAL * np.sum(H_d) * dx + u_b[-1] * H_d[-1])
            T_arr.append((Q_s_fl[-2] + Qs_disp) / (Qs_ero + 1e-30))

            e_dot_vol      = W_GL * np.sum(e_dot) * dx
            Q_release_disp = C_D * W_GL * (
                M_BASAL * np.sum(H_d) * dx + u_b[-1] * H_d[-1])
            Q_release_till = Q_s_fl[-2] / RHO_SED
            Da_arr.append(e_dot_vol / (Q_release_disp + Q_release_till + 1e-30))

        Da_arr = np.array(Da_arr)
        T_arr  = np.array(T_arr)

        # real-world operating point, interpolated in log-Qw
        Da_real = np.interp(np.log10(Qw_real), np.log10(Qw_vals), Da_arr)
        T_real  = np.interp(np.log10(Qw_real), np.log10(Qw_vals), T_arr)

        ax_da.plot(Qw_vals, Da_arr, color=col, lw=2.0, label=name)
        ax_da.plot(Qw_real, Da_real, 'o', color=col, ms=9, mec='k', mew=1.0, zorder=5)
        ax_T.plot(Qw_vals, T_arr, color=col, lw=2.0, label=name)
        ax_T.plot(Qw_real, T_real, 'o', color=col, ms=9, mec='k', mew=1.0, zorder=5)

    ax_da.axhline(1.0, color='k', lw=1.0, ls=':', alpha=0.7,
                  label=r'$\mathrm{Da_{sed}}=1$ (production = release)')
    ax_da.set_ylabel('Sediment Damköhler number  $\\mathrm{Da_{sed}}$')
    ax_da.legend(fontsize=9)
    ax_da.grid(True, alpha=0.3)
    ax_da.text(0.01, 0.97, '(a)', transform=ax_da.transAxes,
               fontsize=11, fontweight='bold', va='top')

    ax_T.set_xscale('log')
    ax_T.set_ylim(-0.02, 1.05)
    ax_T.set_xlabel(r'Water discharge  $Q_w$  (m$^3$ s$^{-1}$)')
    ax_T.set_ylabel(r'Transfer ratio  $\mathcal{T}$')
    ax_T.grid(True, alpha=0.3)
    ax_T.text(0.01, 0.97, '(b)', transform=ax_T.transAxes,
              fontsize=11, fontweight='bold', va='top')

    fig.suptitle('Sensitivity to water discharge (grain size fixed at $D=10$ mm)\n'
                 'dots mark each glacier at its own real $Q_w$', fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'qw_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUT_DIR / 'qw_sensitivity.pdf', bbox_inches='tight')
    plt.show(block=False)
    print('Qw sensitivity figure saved.')


if __name__ == '__main__':
    run_glacier_diagnostics(d_grain=D_FINE, tag='fine')
    run_glacier_diagnostics(d_grain=D_GRAIN, tag='coarse')
    compare_glaciers()
    qw_sensitivity()
