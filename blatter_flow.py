"""
Blatter-Pattyn flowline model with an erosion rule, subglacial hydrology and sediment routing
(Herman and De Doncker, Erosion by a temperate glacier).

Ice flow: first-order momentum equations (Blatter 1995; Pattyn 2003) on a 1-D flowline, solved by
shooting with Anderson acceleration.  Sediment: erosion with till shielding, a distributed drainage
system with supercooling, a frozen fringe and dispersed basal-ice layer, till deformation and
fluvial transport, routed downglacier node by node (steady_routing).

Units are bar, yr and m.  Input data are read from data/, outputs go to figures/ and output/.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq

plt.rcParams['font.size'] = 18

_HERE    = Path(__file__).parent
DATA_DIR = _HERE / 'data'
OUT_DIR  = _HERE / 'output'
FIG_DIR  = _HERE / 'figures'
CACHE_DIR = _HERE / '_cache'          # cached Blatter-Pattyn solutions
INV_DIR  = DATA_DIR / 'inverted'      # sliding coefficient and shape factor inverted from observed velocities
for _d in (OUT_DIR, FIG_DIR, CACHE_DIR, INV_DIR):
    _d.mkdir(exist_ok=True)

AROLLA_FILE     = DATA_DIR / 'arolla.dat'
FRANZJOSEF_FILE = DATA_DIR / 'franzjosef' / 'Bedrock_Anderson2014.csv'
GORNER_FILE     = DATA_DIR / 'gorner' / 'gorner.dat'
QW_FJ     = 50.0    # annual-mean water discharge (m^3/s), Franz Josef Glacier (Herman et al. 2015)
QW_GORNER = 7.0     # Gornergletscher (De Doncker et al. 2026)
QW_REF    = 3.0     # Haut Glacier d'Arolla (Swift et al. 2005)

# Width (m), number of nodes, shape factor f and Weertman coefficient C_s (m yr^-1 bar^-m) per glacier;
# the number of nodes follows from H/dx <~ 2.3 for convergence of the shooting method (Section 4.1).
W_AROLLA, W_FJ, W_GORNER    = 1660.0, 3310.0, 3390.0
NX_AROLLA, NX_FJ, NX_GORNER = 49, 79, 79
SHAPE_AROLLA, SHAPE_FJ, SHAPE_GORNER = 1.0, 0.6, 1.0
CS_AROLLA, CS_FJ, CS_GORNER = 5.0, 10.8, 5.0
# Franz Josef Glacier and Gornergletscher: C_s(x) and f(x) inverted from observed velocities (invert_*_velocity.py).
if (INV_DIR / 'franzjosef_Cs_f.npy').exists():
    CS_FJ, SHAPE_FJ = np.load(INV_DIR / 'franzjosef_Cs_f.npy')
if (INV_DIR / 'gorner_Cs_f.npy').exists():
    CS_GORNER, SHAPE_GORNER = np.load(INV_DIR / 'gorner_Cs_f.npy')

RHO_G   = 900.0 * 9.81 * 1e-5   # bar/m
A_RATE  = 0.08                    # bar^-3 yr^-1
N_GL    = 3.0
T0      = 1e-3                    # bar^2, viscosity regularisation
BETA    = 0.004                   # shooting relaxation (fallback / Anderson damping)
TKRIT   = 1e-3                    # convergence criterion
MAXITER = int(1e5)
NEWTON_TOL = 1e-8                 # max |residual| (yr^-1) accepted from newton_sigma_vec
AA_M    = 10                      # Anderson acceleration history depth
AA_REG  = 1e-4                    # Tikhonov regularisation for Anderson lstsq

C_SLIDE = 5.0                     # Weertman coefficient (m yr^-1 bar^-m)
M_SLIDE = 3                       # Weertman exponent

# Erosion rule e = K_g |u_b|^l; K_g (m^(1-l) yr^(l-1)) is calibrated per glacier at l = 2 (calibrate_kg.py).
K_G        = 2.17e-7
L_EXP      = 2.0
K_G_DEFAULT = K_G
K_G_BY_LOADER = {'load_arolla': 3.14e-5, 'load_franzjosef': 5.93e-7, 'load_gorner': 1.13e-7}
L_BY_LOADER = {}                  # exponent per glacier where it differs from the default
L_DEFAULT = L_EXP
K_G_SCALE = 1.0                   # multiplier on K_G (sensitivity tests)
def use_glacier(key):
    """Set the module erosion coefficient K_G and exponent L_EXP for the glacier named by its loader key."""
    global K_G, L_EXP
    K_G = K_G_SCALE * K_G_BY_LOADER.get(key, K_G_DEFAULT)
    L_EXP = L_BY_LOADER.get(key, L_DEFAULT)
H_TILL_REF = 0.5                   # e-folding till thickness for bedrock shielding (m)
SHIELD_MIN = 0.1                   # minimum shielding factor (residual bedrock wear under thick till)
# Till deformation flux Q = C_DEF_TILL tau_b min(H_till, H_DEF_MAX) W rho_B (Boulton and Hindmarsh 1987), saturating
# in H_till (Truffer et al. 2000; Hansen and Zoet 2022). Field range of C_DEF_TILL is 0.1-5 m yr^-1 bar^-1.
C_DEF_TILL = 1.0                   # till deformation coefficient [m yr⁻¹ bar⁻¹]
H_DEF_MAX  = 0.5                   # max deforming till thickness (m)
H_TILL_TRAP = 10.0                 # nominal H_till reported at a trapping node (m)

RHO_R   = 2700.0                  # bedrock density (kg m^-3)
RHO_B   = 1600.0                  # bulk sediment density (kg m^-3)
RHO_ICE = 900.0                   # ice density (kg m^-3)
W_GL    = 500.0                   # glacier width (m), constant

# Dispersed sediment layer (storage model, Pierce et al. 2026)
C_D      = 0.03    # dispersed layer sediment mass fraction

# Subglacial fluvial transport (Engelund-Hansen / Delaney et al. 2019 JGR-ES)
D_GRAIN  = 0.01     # coarse reference grain diameter (m): fine gravel / bedload
D_FINE   = 63e-6    # fine reference grain diameter (m), silt/sand boundary
RHO_W    = 1000.0  # water density (kg m^-3)
RHO_SED  = 2650.0  # sediment grain density (kg m^-3), quartz
N_MANN   = 0.03    # Manning roughness (m^{-1/3} s), subglacial R-channel
THETA_C  = 0.047   # critical Shields parameter for motion initiation

# Effective-pressure hydrology, frozen fringe and regelation (SI units inside the functions that use them);
# parameters after Pierce et al. (2026), Meyer et al. (2019, 2023), Werder et al. (2013) and Hill et al. (2023).
SPY      = 365.25 * 24 * 3600   # seconds per year
G_ACC    = 9.81                 # m s^-2
G_GEO    = 0.06                 # geothermal heat flux (W m^-2)
L_FUS    = 3.34e5               # latent heat of fusion (J kg^-1)
K_ICE    = 2.1                  # ice thermal conductivity (W m^-1 K^-1)
T_MELT   = 273.15               # K
GAMMA_IW = 0.034                # ice-water surface energy (J m^-2)
R_PORE   = 1e-6                 # pore-throat radius (m)
ETA_W    = 1.8e-3               # water viscosity (Pa s)
PHI_TILL = 0.35                 # till porosity
K0_TILL  = 4.1e-17              # till permeability (m^2)
R_GRAIN  = 1.5e-4               # till grain radius (m), medium sand
D_FILM   = 1e-8                 # premelted film thickness (m)
ALPHA_F  = 3.1                  # fringe permeability exponent
BETA_F   = 0.53                 # fringe saturation exponent
Z_STAR   = 100.0                # characteristic depth of the temperate layer (m)
D_FRINGE_FULL = 0.5e-3           # full-efficiency fringe uptake up to this grain size (m)
D_FRINGE_MAX = 2e-3             # largest grain the fringe can take up (m)
P_FRINGE = 2.0 * GAMMA_IW / R_PORE                       # fringe entry pressure (Pa) = 68 kPa
T_FRINGE = T_MELT - P_FRINGE * T_MELT / (RHO_ICE * L_FUS)  # fringe base temperature (K)
# Distributed drainage (linked cavities, laminar sheet); cavity geometry after Werder et al. (2013).
K_SHEET  = 0.05                 # sheet conductivity, q = -K_SHEET h^3 dphi/dx (m^2/s with phi in Pa)
H_BUMP   = 0.1                  # bed bump height / max cavity size (m), Werder et al. 2013
L_BUMP   = 10.0                 # cavity spacing (m), Pierce et al. 2026
A_ICE_SI = 2.4e-24              # ice rate factor (Pa^-3 s^-1), closure of cavities
# Glaciohydraulic supercooling (Röthlisberger & Lang 1987; Alley et al. 1998;
# Creyts & Clarke 2010)
C_W      = 4218.0               # specific heat of water (J kg^-1 K^-1)
C_T      = 9.8e-8               # pressure dependence of the melting point, air-saturated (K Pa^-1)
F_SHEET  = 0.1                  # fraction of the width carrying sediment in the sheet where channels are frozen shut
F_DW     = 0.1                  # Darcy-Weisbach friction factor of the sheet flow
# Water-pressure level for the fringe, p_w = F_W p_i (boreholes give 0.8-0.9); None uses the sheet model's N.
F_W      = 0.8

NX = 79
NZ = 20


def load_arolla():
    d  = np.loadtxt(AROLLA_FILE)
    return d[:, 0], d[:, 1], d[:, 2]


FJ_END_BED_CORRECTION = True
FJ_END_FIT_M = 500.0


def load_franzjosef():
    """Franz Josef Glacier flowline (Anderson et al. 2010): bedrock and ice surface, the surface interpolated onto the bedrock grid."""
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
    if FJ_END_BED_CORRECTION:
        # the last ~220 m of the digitised bed rise; extrapolate the preceding down-glacier trend instead
        ib = int(np.argmin(bb))
        if ib < len(xx) - 1:
            sel = xx >= xx[ib] - FJ_END_FIT_M
            sel[ib + 1:] = False
            slope = np.polyfit(xx[sel], bb[sel], 1)[0]
            bb[ib + 1:] = bb[ib] + slope * (xx[ib + 1:] - xx[ib])
            hh = np.maximum(hh, bb)
    return xx, bb, hh


def load_gorner():
    """Gornergletscher flowline: distance (m), bed and surface elevation (m a.s.l.)."""
    d = np.loadtxt(GORNER_FILE)
    return d[:, 0], d[:, 1], d[:, 2]


def setup_geometry(xx, bb, hh, nx=None):
    nx  = NX if nx is None else nx
    x   = np.linspace(xx[0], xx[-1], nx)
    sig = np.linspace(0.0, 1.0, NZ)
    dx  = x[1] - x[0]
    ds  = sig[1] - sig[0]

    b   = interp1d(xx, bb, kind='cubic')(x)
    h   = interp1d(xx, hh, kind='cubic')(x)
    hb  = np.maximum(h - b, 0.0)   # cubic spline can undershoot → negative H
    h   = b + hb                    # keep surface consistent

    dhdx = np.gradient(h, x)
    dbdx = np.gradient(b, x)

    # the geometry factors C and d use the physical slopes at every node, including the ends
    C = np.outer(dbdx, (1.0 - sig)) + np.outer(dhdx, sig)
    d = dhdx - dbdx

    Z_o = b[:, None] + np.outer(hb, sig)

    return x, sig, dx, ds, b, h, hb, dhdx, dbdx, C, d, Z_o


def _sigma_residual(s, c, g, tau):
    """Residual y(s) and derivative dy/ds of the σ_xx compatibility condition."""
    t_xz = tau + 2.0*c*s
    eff2 = t_xz**2 + s**2
    base = eff2 + T0
    F    = base**((N_GL - 1.0) / 2.0)
    G    = -2.0*c*tau - (1.0 + 4.0*c**2)*s
    y    = g + A_RATE * F * G
    dF   = ((N_GL-1.0)/2.0) * base**((N_GL-3.0)/2.0) * (4.0*t_xz*c + 2.0*s)
    dy   = A_RATE * (dF*G + F*(-(1.0 + 4.0*c**2)))
    return y, dy


def newton_sigma_vec(s, c, g, tau, n_iter=30, tol=NEWTON_TOL, strict=False):
    """Newton solve of the sigma_xx compatibility condition at one vertical level, with clipped steps.
    With strict=True a RuntimeError is raised if any interior node fails to converge."""
    s = s.copy()
    for _ in range(n_iter):
        y, dy = _sigma_residual(s, c, g, tau)
        step = np.where(np.abs(dy) > 1e-30, y / dy, 0.0)
        s   -= np.clip(step, -0.5, 0.5)
    s = np.clip(s, -5.0, 5.0)
    if strict and len(s) > 2:
        # Interior nodes only: the end nodes are overwritten with σ = 0 by the caller.
        y, _  = _sigma_residual(s, c, g, tau)
        worst = float(np.max(np.abs(y[1:-1])))
        nclip = int(np.sum(np.abs(s[1:-1]) >= 5.0))
        if not np.isfinite(worst) or worst > tol or nclip > 0:
            raise RuntimeError(
                f'newton_sigma_vec: max |residual| = {worst:.3e} yr^-1 '
                f'(tol = {tol:.1e}), {nclip} node(s) on the ±5 bar clip')
    return s


def solve_sigma_row(Sigma_j, C_j, U_j, x, Tau_j, strict=False):
    dudx       = np.gradient(U_j, x)
    S          = newton_sigma_vec(Sigma_j, C_j, dudx, Tau_j, strict=strict)
    S[0] = S[1]; S[-1] = S[-2]          # zero-gradient ends: sigma_xx continuous to the boundary
    return S


def integrate(U, Sigma, Tau, hb, dhdx, d, C, ds, x, use_sigma=True, strict=False, rho_g=RHO_G):
    NX = len(x)
    for j in range(NZ - 1):
        S_j  = Sigma[:, j] if use_sigma else np.zeros(NX)
        t_xz = Tau[:, j] + 2.0*C[:, j]*S_j
        ff   = A_RATE * (S_j**2 + t_xz**2 + T0)
        dudz = ff * (2.0*Tau[:, j] + 4.0*S_j*C[:, j]) * hb
        dSdx = np.gradient(S_j, x); dSdx[[0,-1]] = 0.0
        dtdz = rho_g*dhdx*hb - 2.0*dSdx*hb - 2.0*d*S_j   # 2 from d(2 tau_xx)/dx

        up = U[:, j] + ds*dudz
        tp = Tau[:, j] + ds*dtdz

        sk = solve_sigma_row(S_j, C[:, j+1], up, x, tp, strict=strict) \
             if use_sigma else np.zeros(NX)

        t_xz = 2.0*C[:, j+1]*sk + tp
        ff   = A_RATE * (sk**2 + t_xz**2 + T0)
        dudz = ff * (2.0*tp + 4.0*sk*C[:, j+1]) * hb
        dSkdx = np.gradient(sk, x); dSkdx[[0,-1]] = 0.0
        dtdz = rho_g*dhdx*hb - 2.0*dSkdx*hb - 2.0*d*sk

        U[:, j+1]   = 0.5*(up + U[:, j] + ds*dudz)
        Tau[:, j+1] = 0.5*(tp + Tau[:, j] + ds*dtdz)

        if use_sigma:
            Sigma[:, j+1] = solve_sigma_row(sk, C[:, j+1],
                                             U[:, j+1], x, Tau[:, j+1], strict=strict)
    return U, Sigma, Tau


def anderson_update(x_hist, f_hist, x_k, f_k, m=AA_M, beta=BETA):
    """Anderson(m) acceleration of the fixed-point iteration x = g(x); x_k is the iterate and f_k its residual.
    The history is stored in place and the next iterate is returned."""
    x_hist.append(x_k.copy())
    f_hist.append(f_k.copy())
    if len(x_hist) > m + 1:
        x_hist.pop(0)
        f_hist.pop(0)

    mk = len(x_hist) - 1
    if mk == 0:
        return x_k - beta * f_k

    dF = np.diff(np.column_stack(f_hist), axis=1)
    dX = np.diff(np.column_stack(x_hist), axis=1)

    DtD = dF.T @ dF
    reg = AA_REG * (np.trace(DtD) / mk) * np.eye(mk)
    gamma = np.linalg.solve(DtD + reg, dF.T @ f_hist[-1])
    return x_hist[-1] - beta * f_hist[-1] - (dX - beta * dF) @ gamma


def shoot(x, dx, ds, hb, dhdx, d, C, c_slide=C_SLIDE, shape=1.0, tau0=None, verbose=True):
    NX = len(x)
    """
    shape : valley shape factor f multiplying the driving stress (Nye 1965),
            representing lateral drag absent from a flowline model.
    """
    rho_g = RHO_G * shape
    U     = np.zeros((NX, NZ))
    Sigma = np.zeros((NX, NZ))
    Tau   = np.zeros((NX, NZ))
    Tau[:, 0] = -rho_g * hb * dhdx if tau0 is None else np.asarray(tau0, float).copy()

    x_hist, f_hist = [], []
    for nkk in range(MAXITER):
        U[:, 0] = c_slide * np.abs(Tau[:, 0])**(M_SLIDE - 1) * Tau[:, 0]
        Sigma[:, 0] = solve_sigma_row(Sigma[:, 0], C[:, 0], U[:, 0], x, Tau[:, 0])
        U, Sigma, Tau = integrate(U, Sigma, Tau, hb, dhdx, d, C, ds, x, rho_g=rho_g)

        residual = Tau[:, -1].copy()
        tmax = np.max(np.abs(residual))

        Tau[:, 0] = anderson_update(x_hist, f_hist, Tau[:, 0], residual)

        if verbose and nkk % 50 == 0:
            print(f'  iter {nkk:6d},  tmax = {tmax:.6f} bar')
        if not np.isfinite(tmax):
            raise RuntimeError(f'shoot diverged at iteration {nkk} (tmax = {tmax})')
        if tmax < TKRIT:
            if verbose:
                print(f'  Converged after {nkk} iterations,  tmax = {tmax:.6f} bar')
            break
    else:
        raise RuntimeError(
            f'shoot did not converge in {MAXITER} iterations: '
            f'max |tau_xx(surface)| = {tmax:.3e} bar (TKRIT = {TKRIT})')

    # Verification pass: re-integrate from the last accepted basal stress with strict Newton checks;
    # this is the state that is returned.
    Tau[:, 0]   = x_hist[-1]
    U[:, 0]     = c_slide * np.abs(Tau[:, 0])**(M_SLIDE - 1) * Tau[:, 0]
    Sigma[:, 0] = solve_sigma_row(Sigma[:, 0], C[:, 0], U[:, 0], x, Tau[:, 0], strict=True)
    U, Sigma, Tau = integrate(U, Sigma, Tau, hb, dhdx, d, C, ds, x, strict=True, rho_g=rho_g)
    tmax_final = np.max(np.abs(Tau[:, -1]))
    if tmax_final >= TKRIT:
        raise RuntimeError(
            f'shoot: verification pass gives max |tau_xx(surface)| = {tmax_final:.3e} bar '
            f'(TKRIT = {TKRIT})')
    if verbose:
        print(f'  Verified: surface residual {tmax_final:.2e} bar, '
              f'all sigma-level Newton residuals < {NEWTON_TOL:.0e} yr^-1, '
              f'no node on the stress clip.')

    return U, Sigma, Tau


def invert_sliding(loader, u_obs_x, u_obs, nx=None, shape=1.0, c0=C_SLIDE, n_iter=10,
                   smooth_m=500.0, tol=1.15, bounds=(0.3, 200.0), verbose=False,
                   f_min=None, hold_outside=False):
    """Invert the Weertman coefficient C_s(x) so that the modelled surface velocity matches the observed profile,
    by damped fixed-point iteration C_s <- C_s (u_obs/u_model)^0.7 smoothed over smooth_m.
    If f_min is given, a lateral shape factor f(x) in [f_min, 1] is inverted together with C_s.
    If hold_outside, the misfit is evaluated only between the first and last observation.
    Returns (C_s, f) if f_min is given, else C_s, then the log-misfit factor and the solution."""
    from scipy.ndimage import gaussian_filter1d
    xx, bb, hh = loader()
    x, sig, dx, ds, b, h, hb, dhdx, dbdx, C, d, Z_o = setup_geometry(xx, bb, hh, nx=nx)
    uo = np.interp(x, u_obs_x, u_obs)
    ice = hb > 20.0
    span = (x >= np.min(u_obs_x)) & (x <= np.max(u_obs_x))
    if hold_outside:
        ice = ice & span
    Cs = np.full(len(x), float(c0)); tau0 = None; mis = np.inf; sol = None
    fx = np.full(len(x), float(shape)) if f_min is not None else shape
    for it in range(n_iter):
        U, Sigma, Tau = shoot(x, dx, ds, hb, dhdx, d, C, c_slide=Cs, shape=fx, tau0=tau0, verbose=False)
        tau0 = Tau[:, 0].copy(); sol = (x, sig, dx, ds, b, h, hb, dhdx, dbdx, C, d, Z_o, U, Sigma, Tau)
        r = np.where(ice, uo / np.maximum(U[:, -1], 1.0), 1.0)
        mis = float(np.exp(np.sqrt(np.mean(np.log(r[ice])**2))))
        if verbose:
            print(f'  invert_sliding iter {it}: misfit factor {mis:.3f}, C_s {Cs[ice].min():.1f}-{Cs[ice].max():.1f}')
        if mis < tol:
            break
        g = 0.7 * np.log(np.clip(r, 0.3, 3.0))
        if f_min is None:
            Cs = np.clip(gaussian_filter1d(Cs * np.exp(g), smooth_m / dx), *bounds)
        else:
            f_new = np.clip(fx * np.exp(g / 3.0), f_min, 1.0)
            g_rem = g - 3.0 * np.log(f_new / fx)
            fx = np.clip(gaussian_filter1d(f_new, smooth_m / dx), f_min, 1.0)
            Cs = np.clip(gaussian_filter1d(Cs * np.exp(g_rem), smooth_m / dx), *bounds)
        if hold_outside:
            i0, i1 = np.where(span)[0][[0, -1]]
            Cs[:i0] = Cs[i0]; Cs[i1 + 1:] = Cs[i1]
            if f_min is not None:
                fx[:i0] = fx[i0]; fx[i1 + 1:] = fx[i1]
    if f_min is not None:
        return (Cs, fx), mis, sol
    return Cs, mis, sol

_BP_FIELDS = ('x', 'sig', 'dx', 'ds', 'b', 'h', 'hb', 'dhdx', 'dbdx', 'C', 'd',
              'Z_o', 'U', 'Sigma', 'Tau')


def bp_solution(key, loader, c_slide=C_SLIDE, shape=1.0, nx=None):
    """Cached geometry and Blatter-Pattyn solution for a named glacier, stored in CACHE_DIR with the sliding
    coefficient in the file name (delete _cache/ after changing other ice-flow parameters).
    Selects the glacier's K_G (use_glacier)."""
    use_glacier(key)
    if np.ndim(c_slide) > 0:
        import hashlib
        cs_tag = '_Csinv' + hashlib.md5(np.asarray(c_slide, float).tobytes()).hexdigest()[:8]
    else:
        cs_tag = '' if c_slide == C_SLIDE else f'_Cs{c_slide:g}'
    if np.ndim(shape) > 0:
        import hashlib
        f_tag = '_finv' + hashlib.md5(np.asarray(shape, float).tobytes()).hexdigest()[:8]
    else:
        f_tag = '' if shape == 1.0 else f'_f{shape:g}'
    tag = cs_tag + f_tag + ('' if nx in (None, NX) else f'_nx{nx}')
    cache_file = CACHE_DIR / f'bp_{key}{tag}.npz'
    if cache_file.exists():
        data = np.load(cache_file)
        return tuple(data[f] for f in _BP_FIELDS)

    xx, bb, hh = loader()
    x, sig, dx, ds, b, h, hb, dhdx, dbdx, C, d, Z_o = setup_geometry(xx, bb, hh, nx=nx)
    U, Sigma, Tau = shoot(x, dx, ds, hb, dhdx, d, C, c_slide=c_slide, shape=shape)
    result = (x, sig, dx, ds, b, h, hb, dhdx, dbdx, C, d, Z_o, U, Sigma, Tau)
    np.savez(cache_file, **dict(zip(_BP_FIELDS, result)))
    return result


def fluvial_transport(Q_w_total, e_dot, dbdx, dx, x, d_grain=D_GRAIN, dhdx=None,
                      S_hyd=None):
    """Subglacial transport capacity (Delaney et al. 2019): Engelund-Hansen rate in a semicircular R-channel whose
    radius follows from Manning's equation, with Q_w(x) = Q_w_total x / L.  The hydraulic slope is S_hyd if given,
    else the Shreve gradient if dhdx is given, else the bed slope.
    Returns the flux min(cap, supply), the capacity and the supply (kg/yr)."""
    SPY = 365.25 * 24 * 3600

    Q_w = np.maximum(Q_w_total * x / x[-1], 1e-10)
    if S_hyd is not None:
        S = np.maximum(S_hyd, 1e-6)               # emergent gradient from hydrology_1d
    elif dhdx is not None:
        f_rho = RHO_ICE / RHO_W                   # ≈ 0.9
        S_shreve = -(1.0 - f_rho) * dbdx - f_rho * dhdx   # Shreve hydraulic gradient
        S = np.maximum(S_shreve, 1e-4)
    else:
        S = np.maximum(np.abs(dbdx), 1e-4)        # bed-slope approximation

    r   = (2.0**(5.0/3.0) * N_MANN * Q_w / (np.pi * np.sqrt(S)))**0.375
    R_h = r / 2.0

    theta   = RHO_W * R_h * S / ((RHO_SED - RHO_W) * d_grain)
    Phi     = 0.05 * np.maximum(theta - THETA_C, 0.0)**2.5
    q_s     = Phi * np.sqrt((RHO_SED / RHO_W - 1.0) * 9.81 * d_grain**3)  # m²/s
    Q_s_cap = RHO_SED * q_s * 2.0 * r * SPY      # kg/yr

    # erosion supply to the fluvial system
    Q_s_sup = np.cumsum(RHO_R * e_dot * W_GL) * dx  # kg/yr

    Q_s_fl = np.minimum(Q_s_cap, Q_s_sup)
    return Q_s_fl, Q_s_cap, Q_s_sup


# Effective-pressure hydrology, basal melt, frozen fringe, regelation

def basal_melt_rate(u_b, tau_b_bar):
    """
    Basal melt rate (m ice yr^-1) from geothermal plus frictional heating:
        m = (G_geo + tau_b u_b) / (rho_i L)
    u_b in m/yr, tau_b in bar.  Strictly positive (G_geo > 0).
    """
    heat = G_GEO + (tau_b_bar * 1e5) * (np.abs(u_b) / SPY)      # W m^-2
    return heat / (RHO_ICE * L_FUS) * SPY                          # m/yr


def _cavity_sheet_thickness(N_pa, u_b_mps):
    """
    Steady linked-cavity sheet thickness h(N): opening by sliding over bumps
    balances creep closure (Werder et al. 2013; Pierce et al. 2026):
        u_b (h_r - h)/l_r = (2/n^n) A h |N|^(n-1) N
    For N <= 0 closure vanishes and h = h_r.
    """
    n = N_GL
    if N_pa <= 0.0:
        return H_BUMP
    open_rate  = u_b_mps / L_BUMP
    close_coef = (2.0 / n**n) * A_ICE_SI * N_pa**n
    return open_rate * H_BUMP / (open_rate + close_coef + 1e-300)


def hydrology_1d(x, hb, b, u_b, Q_w_total, dx, W=W_GL):
    """Steady distributed drainage along the flowline (laminar linked-cavity sheet).  The flux q(x) = Q_w(x)/W passes
    through a sheet with q = -K_SHEET h^3 dphi/dx, integrated upstream from the terminus (p_w = 0) with h from
    _cavity_sheet_thickness(N); the implicit relation for N at each node is solved by bracketing.
    Returns N (Pa), p_w (Pa), h_sheet (m), S_hyd (-), phi (Pa)."""
    n   = len(x)
    L   = x[-1]
    q   = np.maximum(Q_w_total * x / L, 1e-12) / W           # m^2/s
    u   = np.abs(u_b) / SPY                                   # m/s
    p_i = RHO_ICE * G_ACC * hb                                # Pa
    phi = np.zeros(n)
    N   = np.zeros(n)
    hs  = np.zeros(n)

    # terminus: atmospheric water pressure
    phi[-1] = RHO_W * G_ACC * b[-1]
    N[-1]   = p_i[-1]
    hs[-1]  = _cavity_sheet_thickness(N[-1], u[-1])

    for i in range(n - 2, -1, -1):
        base = p_i[i] + RHO_W * G_ACC * b[i]                  # phi at N = 0
        qi   = 0.5 * (q[i] + q[i + 1])

        def F(Ni):
            h = _cavity_sheet_thickness(Ni, u[i])
            phi_i = phi[i + 1] + qi * dx / (K_SHEET * h**3)
            return Ni - (base - phi_i)

        # F increases monotonically with N: expand the bracket until the signs differ.
        lo = -1e3
        while F(lo) > 0.0 and lo > -1e12:
            lo *= 10.0
        hi = max(p_i[i], 1.0) + 1.0
        while F(hi) < 0.0 and hi < 1e12:
            hi *= 2.0
        N[i]   = brentq(F, lo, hi, xtol=1e-3, maxiter=200)
        # p_w cannot fall below atmospheric: where the sheet passes q with less head than the bed slope gives,
        # the flow is unpressurised (N = p_i) and the gradient is the bed slope.
        if N[i] > p_i[i]:
            N[i] = p_i[i]
        hs[i]  = _cavity_sheet_thickness(N[i], u[i])
        phi[i] = base - N[i]

    p_w  = phi - RHO_W * G_ACC * b
    S    = np.zeros(n)
    S[:-1] = (phi[:-1] - phi[1:]) / dx / (RHO_W * G_ACC)
    S[-1]  = S[-2]
    return N, p_w, hs, np.maximum(S, 0.0), phi


def supercooling_freeze_on(x, Q_w_total, p_w, phi, dx, W=W_GL):
    """Glaciohydraulic supercooling (Rothlisberger and Lang 1987; Alley et al. 1998): where the heat needed to follow the
    melting point exceeds viscous dissipation, ice freezes on at a width-averaged rate m_fr (m ice yr^-1).
    Returns (m_fr >= 0, supercooled mask)."""
    Q_w  = np.maximum(Q_w_total * x / x[-1], 1e-12)            # m^3/s
    dphi = np.gradient(phi, x)
    dpw  = np.gradient(p_w, x)
    P_diss = Q_w * np.maximum(-dphi, 0.0)                         # W/m
    P_req  = RHO_W * C_W * C_T * Q_w * np.maximum(-dpw, 0.0)      # W/m
    # Pressurised reaches only; a drop onto the atmospheric floor is a clamp, not a pressure gradient.
    pressurised = p_w > 1.0
    pressurised[:-1] &= pressurised[1:]
    P_req  = np.where(pressurised, P_req, 0.0)
    F      = np.maximum(P_req - P_diss, 0.0) / L_FUS              # kg/s per m
    m_fr   = F / (RHO_ICE * W) * SPY                              # m ice / yr
    return m_fr, m_fr > 0.0


def settling_velocity(d_grain):
    """Ferguson & Church (2004) settling velocity (m/s) for natural grains."""
    R, nu = RHO_SED / RHO_W - 1.0, 1.0e-6
    return R * G_ACC * d_grain**2 / (18.0 * nu + np.sqrt(0.75 * R * G_ACC * d_grain**3))


def rouse_number(u_star, d_grain):
    """Rouse number P = w_s / (kappa u*); P < ROUSE_SUSP: carried in suspension."""
    return settling_velocity(d_grain) / (0.4 * np.maximum(u_star, 1e-9))


ROUSE_SUSP = 2.5                # Rouse number above which grains move only as bedload
BED_ONLY_ENTRAINMENT = True     # fringe and freeze-on draw only on bed sediment (Pierce et al. 2026); False also feeds channel bedload to the bed pool
ROUSE_WASH = 0.8                # Rouse number below which grains are fully suspended (wash load)


def bedload_fraction(P):
    """Fraction of the load that moves on the bed: 1 for P >= ROUSE_SUSP, 0 for P <= ROUSE_WASH,
    linear in between (the usual Rouse classification is graded: bedload above 2.5,
    partly suspended 1.2-2.5, suspended 0.8-1.2, wash load below 0.8)."""
    return np.clip((P - ROUSE_WASH) / (ROUSE_SUSP - ROUSE_WASH), 0.0, 1.0)


def channel_shear_velocity(Q_w_total, x, S_hyd):
    """Shear velocity u* = sqrt(g R_h S) (m/s) in the R-channel of fluvial_transport."""
    Q_w = np.maximum(Q_w_total * x / x[-1], 1e-10)
    S   = np.maximum(S_hyd, 1e-6)
    r   = (2.0**(5.0/3.0) * N_MANN * Q_w / (np.pi * np.sqrt(S)))**0.375
    return np.sqrt(G_ACC * (r / 2.0) * S)


def sheet_transport_capacity(q, h_sheet, d_grain, W=W_GL):
    """Capacity (kg/yr) of the distributed sheet where channels are frozen shut.  Grains with Rouse number below
    ROUSE_SUSP are suspended and pass; coarser grains move as bedload at the Engelund-Hansen rate over a fraction
    F_SHEET of the width."""
    # the sediment-carrying flow occupies a fraction F_SHEET of the width
    u      = q / F_SHEET / np.maximum(h_sheet, 1e-6)
    tau    = F_DW / 8.0 * RHO_W * u**2
    u_star = np.sqrt(tau / RHO_W)
    theta  = tau / ((RHO_SED - RHO_W) * G_ACC * d_grain)
    Phi    = 0.05 * np.maximum(theta - THETA_C, 0.0)**2.5
    q_s    = Phi * np.sqrt((RHO_SED / RHO_W - 1.0) * G_ACC * d_grain**3)
    bedload = RHO_SED * q_s * F_SHEET * W * SPY
    # only the bedload fraction is capacity-limited; the suspended part passes freely
    bf_ = bedload_fraction(rouse_number(u_star, d_grain))
    return np.where(bf_ <= 1e-9, 1e30, np.minimum(bedload / np.maximum(bf_, 1e-9), 1e30))


def _fringe_terms(theta, N_pa, G, phi=PHI_TILL):
    """Heave-rate numerator, denominator and V_n (Pierce et al. 2026 code)."""
    a, bt = ALPHA_F, BETA_F
    dT   = T_MELT - T_FRINGE
    V_n  = -RHO_W**2 * L_FUS * G * K0_TILL / (RHO_ICE * T_MELT * ETA_W)
    Pi   = -RHO_W**2 * K0_TILL * G * R_GRAIN**2 / (RHO_ICE**2 * dT * D_FILM**3)
    num  = theta + phi * (1.0 - theta + (theta**(1.0 - bt) - 1.0) / (1.0 - bt)) - N_pa / P_FRINGE
    den  = ((1.0 - phi)**2 / (a + 1.0) * (theta**(a + 1.0) - 1.0)
            + 2.0 * (1.0 - phi) * phi / (a - bt + 1.0) * (theta**(a - bt + 1.0) - 1.0)
            + phi**2 / (a - 2.0 * bt + 1.0) * (theta**(a - 2.0 * bt + 1.0) - 1.0)
            + Pi)
    return num, den, V_n


def fringe_steady_thickness(N_pa, m_myr):
    """Steady frozen-fringe thickness H_f (m) at one node (Meyer et al. 2019, 2023; Pierce et al. 2026), from V(theta) = -m
    with theta the undercooling.  Returns (H_f, theta); H_f = 0 and theta = 1 where no fringe forms."""
    m = m_myr / SPY
    if m <= 0.0 or N_pa <= 0.0:
        return 0.0, 1.0
    G  = -m * RHO_ICE * L_FUS / K_ICE
    dT = T_MELT - T_FRINGE

    def V(theta):
        num, den, V_n = _fringe_terms(theta, N_pa, G)
        return V_n * num / den

    if V(1.0) >= -m:
        return 0.0, 1.0
    # upper bracket: theta where Num = 0 (V = 0 > -m)
    th_hi = 1.0
    while _fringe_terms(th_hi, N_pa, G)[0] < 0.0:
        th_hi *= 2.0
        if th_hi > 1e6:
            break
    theta = brentq(lambda t: V(t) + m, 1.0, th_hi, xtol=1e-10, maxiter=200)
    H_f = (theta - 1.0) * dT / (-G)
    return H_f, theta


def regelation_rate(theta, H_d):
    """
    Dispersed-layer growth rate (m ice yr^-1) by regelation above a frozen
    fringe (Pierce et al. 2026, Eq. 5 and code):
        v = -3 k_i G_top / (rho_i L),   G_top = (T_top - T_m)/max(H_d, z*),
        T_top = T_m - theta (T_m - T_f).
    """
    dT    = T_MELT - T_FRINGE
    G_top = -theta * dT / max(H_d, Z_STAR)
    return -3.0 * K_ICE * G_top / (RHO_ICE * L_FUS) * SPY


def steady_routing(u_b, Qw, dbdx, dx, x, hb, b, d_grain=D_GRAIN,
                   max_iter=30, tol=1e-3, f_w=F_W, W=W_GL, c_slide=C_SLIDE, tau_b=None):
    """Steady sediment routing along the flowline, one downstream marching pass per shielding iteration.  At each node the bed
    pool (local erosion, till arriving by deformation, melt-out and channel bedload) exchanges with the frozen fringe and
    the dispersed layer and passes to the channel (at capacity), to till deformation (capped) or to accumulation.
    f_w prescribes the fringe effective pressure N = (1 - f_w) p_i; None uses the sheet model's N.
    Returns a dict of (NX,) arrays (Q_fl, Q_cap, Q_def, Q_acc, H_till, e_dot, shield, Q_fr, Q_d, H_f, H_d, N, p_w, S_hyd, ...)."""
    n = len(u_b)
    N, p_w, h_sheet, S_hyd, phi = hydrology_1d(x, hb, b, u_b, Qw, dx, W=W)
    p_i = RHO_ICE * G_ACC * hb
    N_fringe = (1.0 - f_w) * p_i if f_w is not None else N

    _, Q_cap, _ = fluvial_transport(Qw, np.ones(n), dbdx, dx, x,
                                    d_grain=d_grain, S_hyd=S_hyd)
    # channel-borne grains on the bed (bedload) or in the water (suspension)?
    chan_bedload = bedload_fraction(rouse_number(channel_shear_velocity(Qw, x, S_hyd), d_grain))
    if BED_ONLY_ENTRAINMENT:
        chan_bedload = np.zeros(n)

    # Glaciohydraulic supercooling: freeze-on and channel closure
    m_fr, supercooled = supercooling_freeze_on(x, Qw, p_w, phi, dx, W=W)
    if supercooled.any():
        q_w = np.maximum(Qw * x / x[-1], 1e-12) / W
        Q_cap = np.where(supercooled,
                         sheet_transport_capacity(q_w, h_sheet, d_grain, W=W), Q_cap)
    # accreted ice thickness (upper bound: width-averaged, no conduction)
    H_accreted = np.cumsum(m_fr * dx / np.maximum(np.abs(u_b), 1e-3))

    if tau_b is None:   # Weertman inverse; pass the BP basal stress when u_b is prescribed from data
        tau_b = np.maximum((np.abs(u_b) / (c_slide + 1e-30))**(1.0 / M_SLIDE), 0.01)
    B_def = RHO_B * C_DEF_TILL * tau_b * W
    m_bas = basal_melt_rate(u_b, tau_b)
    m_net = m_bas - m_fr                        # net basal melt (< 0: freeze-on)

    H_f_eq = np.zeros(n); theta = np.ones(n)
    for i in range(n):
        H_f_eq[i], theta[i] = fringe_steady_thickness(N_fringe[i], max(m_net[i], 1e-6))
    has_fringe = H_f_eq > 0.0

    prod = RHO_R * K_G * np.abs(u_b)**L_EXP                     # kg m^-2 yr^-1
    c_fr = RHO_SED * (1.0 - PHI_TILL) * W                       # kg per (m^2/yr) of fringe flux
    c_d  = RHO_ICE * C_D * W

    def _shield_of(H):
        return SHIELD_MIN + (1.0 - SHIELD_MIN) * np.exp(-H / H_TILL_REF)

    # grain-size efficiency of fringe uptake: 1 for fine grains, 0 above D_FRINGE_MAX
    eta_fringe = float(np.clip(np.log(D_FRINGE_MAX / d_grain) / np.log(D_FRINGE_MAX / D_FRINGE_FULL), 0.0, 1.0))

    H_till = np.zeros(n)
    shield = np.ones(n)

    for _ in range(max_iter):
        e_dot = K_G * np.abs(u_b)**L_EXP * shield
        Q_fl = np.zeros(n); Q_def = np.zeros(n); Q_acc = np.zeros(n)
        Q_fr = np.zeros(n); Q_d = np.zeros(n); H_till_new = np.zeros(n)
        upt = np.zeros(n); rel = np.zeros(n); v_reg = np.zeros(n); acc = np.zeros(n)
        Q_fl_in = 0.0; Q_def_in = 0.0; Qf = 0.0; Qd = 0.0

        for i in range(n):
            fresh = RHO_R * e_dot[i] * W * dx
            # Bed pool: local erosion, till arriving by deformation, and channel bedload.
            bed  = fresh + Q_def_in + chan_bedload[i] * Q_fl_in
            susp = (1.0 - chan_bedload[i]) * Q_fl_in

            # --- dispersed layer melt-out lands on the bed (where no fringe)
            if not (has_fringe[i] and np.abs(u_b[i]) > 0.0):
                melt = min(c_d * max(m_net[i], 0.0) * dx, Qd)
                Qd -= melt; bed += melt

            # --- supercooling: bed sediment accretes into basal ice (Lawson et al. 1998)
            if m_fr[i] > 0.0:
                acc[i] = min(c_fr * m_fr[i] * dx, bed)
                Qf += acc[i]; bed -= acc[i]

            # --- frozen fringe exchange with the bed: uptake where the equilibrium thickness grows, release where it thins;
            #     grains coarser than D_FRINGE_MAX are not taken up
            Qf_eq = c_fr * np.abs(u_b[i]) * H_f_eq[i]
            if Qf_eq > Qf and eta_fringe > 0.0:
                upt[i] = eta_fringe * min(Qf_eq - Qf, bed)
                Qf += upt[i]; bed -= upt[i]
            elif Qf > Qf_eq:
                rel[i] = min(Qf - Qf_eq, c_fr * max(m_net[i], 0.0) * dx)
                Qf -= rel[i]; bed += rel[i]

            # --- dispersed layer: regelation from the fringe (where a fringe exists)
            if has_fringe[i] and np.abs(u_b[i]) > 0.0:
                H_d_i    = Qd / (c_d * np.abs(u_b[i]))
                v_reg[i] = regelation_rate(theta[i], H_d_i)
                reg      = min(c_d * v_reg[i] * dx, Qf)
                Qf -= reg; Qd += reg
            Q_fr[i] = Qf; Q_d[i] = Qd

            available = susp + bed          # what the channel can take at this node
            Q_run     = Q_fl_in + Q_def_in  # upstream flux, for the till balance below

            # --- bed routing (channel / till deformation / accumulation)
            if i == n - 1:
                Q_fl[i] = min(available, Q_cap[i]); H_till_new[i] = 0.0
                continue
            if available >= Q_cap[i]:
                Q_fl[i]    = Q_cap[i]
                sup_coeff  = prod[i] * W * dx
                # available = Q_run + fresh + net, with net the fringe and dispersed exchange (uptake < 0, release > 0)
                net = available - Q_run - fresh
                excess_inf = Q_run + net + sup_coeff * SHIELD_MIN - Q_cap[i]
                if excess_inf > B_def[i] * H_DEF_MAX:
                    H_till_new[i] = H_TILL_TRAP
                    Q_def[i]      = B_def[i] * H_DEF_MAX
                    Q_acc[i]      = excess_inf - Q_def[i]
                else:
                    H = min(max(H_till[i], 0.0), H_DEF_MAX)
                    for _ in range(30):
                        f  = B_def[i] * min(H, H_DEF_MAX) - (Q_run + net + sup_coeff * _shield_of(H) - Q_cap[i])
                        fp = (B_def[i] if H < H_DEF_MAX else 0.0) \
                             + sup_coeff * (1.0 - SHIELD_MIN) / H_TILL_REF * np.exp(-H / H_TILL_REF)
                        if fp <= 0.0:
                            break
                        H_new = max(0.0, H - f / fp)
                        if abs(H_new - H) < 1e-9:
                            H = H_new; break
                        H = H_new
                    H_till_new[i] = H
                    Q_def[i]      = B_def[i] * min(H, H_DEF_MAX)
            else:
                excess    = Q_cap[i] - available
                till_mass = RHO_B * H_till[i] * W * dx
                remob     = min(excess, till_mass)
                Q_fl[i]   = available + remob
                H_till_new[i] = max(0.0, H_till[i] - remob / (RHO_B * W * dx))
            Q_fl_in, Q_def_in = Q_fl[i], Q_def[i]

        shield_raw = np.exp(-H_till_new / H_TILL_REF); shield_raw[H_till_new <= 0.0] = 1.0
        shield_new = SHIELD_MIN + (1.0 - SHIELD_MIN) * shield_raw
        delta = np.max(np.abs(H_till_new - H_till))
        H_till, shield = H_till_new, shield_new
        if delta < tol:
            break

    e_dot = K_G * np.abs(u_b)**L_EXP * shield
    ub_safe = np.maximum(np.abs(u_b), 1e-12)
    H_f = Q_fr / (c_fr * ub_safe)
    H_d = Q_d / (c_d * ub_safe)
    return dict(Q_fl=Q_fl, Q_cap=Q_cap, Q_def=Q_def, Q_acc=Q_acc, H_till=H_till,
                e_dot=e_dot, shield=shield, Q_fr=Q_fr, Q_d=Q_d, H_f=H_f,
                H_f_eq=H_f_eq, H_d=H_d, theta=theta, N=N, N_fringe=N_fringe,
                p_w=p_w, h_sheet=h_sheet, S_hyd=S_hyd, m_basal=m_bas,
                m_freeze=m_fr, supercooled=supercooled, H_accreted=H_accreted,
                chan_bedload=chan_bedload, v_reg=v_reg, uptake_fr=upt,
                release_fr=rel, accretion_fr=acc, W=W, c_slide=c_slide)


def budget_from_routing(r, dx):
    """
    Sediment budget for a steady_routing() result (budget domain nodes
    0..NX-2, outlets at the penultimate node).  Export pathways: channel,
    till deformation, frozen fringe advected past the terminus, dispersed
    layer advected past the terminus; the remainder is till accumulation at
    trapping nodes.  Storage volumes are grain volumes.
    """
    W      = r.get('W', W_GL)
    Q_ero  = RHO_R * W * np.sum(r['e_dot'][:-1]) * dx
    Q_fl, Q_def, Q_fr, Q_d = r['Q_fl'][-2], r['Q_def'][-2], r['Q_fr'][-2], r['Q_d'][-2]
    Q_acc  = np.sum(r['Q_acc'][:-1])
    Q_out  = Q_fl + Q_def + Q_fr + Q_d
    V_till = RHO_B / RHO_SED * W * np.sum(r['H_till'][:-1]) * dx
    V_fr   = (1.0 - PHI_TILL) * W * np.sum(r['H_f'][:-1]) * dx
    V_disp = RHO_ICE / RHO_SED * C_D * W * np.sum(r['H_d'][:-1]) * dx
    V_stor = V_till + V_fr + V_disp
    Q_release = Q_out / RHO_SED
    e_vol     = Q_ero / RHO_SED
    return dict(Q_ero=Q_ero, Q_fl=Q_fl, Q_def=Q_def, Q_fr=Q_fr, Q_disp=Q_d, Q_acc=Q_acc,
                Q_out=Q_out, T=Q_out / (Q_ero + 1e-30), T_fl=Q_fl / (Q_ero + 1e-30),
                T_def=Q_def / (Q_ero + 1e-30), T_fr=Q_fr / (Q_ero + 1e-30),
                T_disp=Q_d / (Q_ero + 1e-30),
                V_stor=V_stor, V_till=V_till, V_fr=V_fr, V_disp=V_disp,
                Q_release=Q_release, e_vol=e_vol, Da=e_vol / (Q_release + 1e-30),
                tau_stor=V_stor / (Q_release + 1e-30))


def check_mass_conservation(budget, label='', tol=0.01):
    """Print the sediment budget and check that the erosion equals the fluvial, deformation and dispersed outlets plus the
    till accumulation.  Returns True if the relative residual is within tol."""
    b = budget
    Q_fr = b.get('Q_fr', 0.0)
    residual = b['Q_fl'] + b['Q_def'] + Q_fr + b['Q_disp'] + b['Q_acc'] - b['Q_ero']
    rel_err  = abs(residual) / (b['Q_ero'] + 1e-30)
    pct = lambda q: f'({q / b["Q_ero"] * 100:5.1f} % of ero)'

    tag = f'  [{label}] ' if label else '  '
    print(f'\n{tag}--- Mass conservation budget ---')
    print(f'{tag}  Total eroded            : {b["Q_ero"]:.3e} kg/yr')
    print(f'{tag}    → fluvial outlet      : {b["Q_fl"]:.3e} kg/yr  {pct(b["Q_fl"])}')
    print(f'{tag}    → deformation outlet  : {b["Q_def"]:.3e} kg/yr  {pct(b["Q_def"])}')
    if 'Q_fr' in b:
        print(f'{tag}    → fringe outlet       : {Q_fr:.3e} kg/yr  {pct(Q_fr)}')
    print(f'{tag}    → dispersed outlet    : {b["Q_disp"]:.3e} kg/yr  {pct(b["Q_disp"])}')
    print(f'{tag}    → till accumulation   : {b["Q_acc"]:.3e} kg/yr  {pct(b["Q_acc"])}'
          + ('  (trapping nodes present)' if b['Q_acc'] > 0 else ''))
    print(f'{tag}  Transfer ratio T        : {b["T"]:.3f}')
    print(f'{tag}  Stored volume V_stor    : {b["V_stor"]:.3e} m³ sediment '
          f'(till {b["V_till"]:.2e}, fringe {b.get("V_fr", 0.0):.2e}, '
          f'dispersed {b["V_disp"]:.2e})')
    print(f'{tag}  Residence time tau_stor : {b["tau_stor"]:.1f} yr')
    print(f'{tag}  Da_sed                  : {b["Da"]:.3f}')
    print(f'{tag}  Residual (should be 0)  : {residual:.3e} kg/yr  '
          f'(rel. error = {rel_err*100:.4f} %)')
    ok = rel_err < tol
    print(f'{tag}  Conservation check      : {"PASS" if ok else "FAIL"}'
          f'  (tol = {tol*100:.1f} %)')
    return ok


def run_glacier_diagnostics(d_grain=D_GRAIN, tag=None, rows='all'):
    """Steady-state diagnostics for the three glaciers, one column per glacier.  d_grain is the grain diameter (m),
    tag a filename suffix, and rows 'all' or 'sediment' (rows c-e only)."""
    if tag is None:
        tag = f'{d_grain*1000:.2f}mm'.replace('.', 'p') if d_grain >= 1e-4 \
            else f'{d_grain*1e6:.0f}um'

    glaciers = [
        dict(name="Haut Glacier d'Arolla", loader=load_arolla, Qw=QW_REF,
             W=W_AROLLA, Cs=CS_AROLLA, f=SHAPE_AROLLA, nx=NX_AROLLA, label='Arolla reference', vmax=20.0),
        dict(name='Franz Josef Glacier', loader=load_franzjosef, Qw=QW_FJ,
             W=W_FJ, Cs=CS_FJ, f=SHAPE_FJ, nx=NX_FJ, label='Franz Josef reference', vmax=None),
        dict(name='Gornergletscher', loader=load_gorner, Qw=QW_GORNER,
             W=W_GORNER, Cs=CS_GORNER, f=SHAPE_GORNER, nx=NX_GORNER, label='Gornergletscher reference', vmax=None),
    ]

    fig, axes = plt.subplots(5, 3, figsize=(15, 17))

    for col, g in enumerate(glaciers):
        print(f"\nSolving Blatter-Pattyn flow ({g['name']})...")
        x, sig, dx, ds, b, h, hb, dhdx, dbdx, C, d, Z_o, U, Sigma, Tau = \
            bp_solution(g['loader'].__name__, g['loader'], c_slide=g['Cs'], shape=g['f'], nx=g['nx'])
        u_s, u_b = U[:, -1], U[:, 0]
        L = x[-1]

        r = steady_routing(u_b, g['Qw'], dbdx, dx, x, hb, b, d_grain=d_grain,
                           W=g['W'], c_slide=g['Cs'])
        Q_s_fl, Q_s_cap, Q_deform, H_till = r['Q_fl'], r['Q_cap'], r['Q_def'], r['H_till']
        e_dot, shield, Q_accum, H_d, H_f = r['e_dot'], r['shield'], r['Q_acc'], r['H_d'], r['H_f']

        Q_s_ero  = RHO_R * g['W'] * np.cumsum(e_dot) * dx
        bud      = budget_from_routing(r, dx)
        Qs_disp      = bud['Q_disp'] + bud['Q_fr']       # basal-ice export (fringe + dispersed)
        Qs_total_out = bud['Q_out']
        transfer     = bud['T']

        alpha_mean = float(np.mean(np.abs(dhdx[dhdx != 0])))
        H_mean     = float(np.mean(hb[hb > 1.0]))
        print(f'  Mean surface slope:     {np.degrees(alpha_mean):.1f}°')
        print(f'  Mean ice thickness:     {H_mean:.0f} m')
        cs_txt = (f'{g["Cs"]:g}' if np.ndim(g["Cs"]) == 0
                  else f'inverted, {np.min(g["Cs"]):.1f}-{np.max(g["Cs"]):.1f}')
        f_txt = (f'{g["f"]:g}' if np.ndim(g["f"]) == 0
                 else f'inverted, {np.min(g["f"]):.2f}-{np.max(g["f"]):.2f}')
        print(f'  Width / f / C_s / Q_w:  {g["W"]:.0f} m / {f_txt} / {cs_txt} / {g["Qw"]:g} m³/s')
        print(f'  Max surface speed:      {u_s.max():.0f} m/yr  (basal {u_b.max():.0f})')
        print(f'  Mean erosion rate:      {1e3 * np.mean(e_dot[hb > 1]):.2f} mm/yr')
        print(f'  Sheet N:              {r["N"].min()/1e5:.2f}–{r["N"].max()/1e5:.2f} bar '
              f'({int(np.sum(r["N"] <= 0))} nodes at/above overburden); '
              f'fringe N (f_w = {F_W}): {r["N_fringe"].min()/1e5:.2f}–{r["N_fringe"].max()/1e5:.2f} bar')
        print(f'  Fringe nodes:           {int(np.sum(r["H_f_eq"] > 0))}/{len(x)}  '
              f'(max H_f {np.max(H_f[:-1]):.2f} m)')
        sc = r['supercooled']
        print(f'  Supercooled nodes:      {int(sc.sum())}'
              + (f'  at x = {np.round(x[sc] / 1e3, 1).tolist()} km, '
                 f'freeze-on up to {r["m_freeze"].max():.3f} m/yr '
                 f'({r["H_accreted"][-1]:.1f} m accreted ice, upper bound), '
                 f'accretion {np.sum(r["accretion_fr"]):.2e} kg/yr' if sc.any() else ''))
        print(f'  Max H_d:                {np.max(H_d[:-1]):.4f} m')
        print(f'  Max H_till:             {np.max(H_till):.4f} m  '
              f'(shielded frac: {1.0 - np.mean(shield):.3f}; '
              f'trapping nodes: {int(np.sum(Q_accum > 0))})')
        check_mass_conservation(bud, label=g['label'])

        # (a) velocity profile
        ax = axes[0, col]
        ax.plot(x, u_s, 'k-', lw=1.5, label='surface')
        ax.plot(x, u_b, 'b--', lw=1.2, label='basal')
        ax.set_title(f"{g['name']}\n" + r'$Q_w$ = ' + f"{g['Qw']:.0f}" + r' m$^3$ s$^{-1}$',
                     fontsize=20)
        ax.legend(fontsize=11, loc='upper right')
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
        cb.ax.tick_params(labelsize=14)
        cb.set_label('u (m/yr)', fontsize=14)
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
        ax.legend(fontsize=10, loc='upper right')
        ax2b = ax.twinx()
        ax2b.plot(x, H_till, 'saddlebrown', lw=1.0, ls=':')
        ax2b.tick_params(axis='y', labelcolor='saddlebrown')
        ax.set_xlim(0, L)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel(r'$\dot{e}$ (m/yr)')
        if col == 2:
            ax2b.set_ylabel(r'$H_\mathrm{till}$ (m)', color='saddlebrown')

        # (d) basal ice: frozen fringe and dispersed layer; effective pressure
        ax = axes[3, col]
        ax.plot(x[:-1], H_f[:-1], color='teal', lw=1.5, label=r'$H_f$ fringe')
        ax.plot(x[:-1], H_d[:-1], 'g-', lw=1.5, label=r'$H_d$ dispersed')
        ax.set_title(r'$\mathcal{T}$ = ' + f'{transfer:.2f}', fontsize=18)
        ax.legend(fontsize=10, loc='upper left')
        ax3b = ax.twinx()
        ax3b.plot(x, r['N_fringe'] * 1e-5, color='0.4', lw=1.0, ls=':')
        ax3b.tick_params(axis='y', labelcolor='0.4')
        ax.set_xlim(0, L)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel(r'$H_f$, $H_d$ (m ice-eq.)')
        if col == 2:
            ax3b.set_ylabel(r'$N$ (bar)', color='0.4')

        # (e) sediment fluxes
        ax = axes[4, col]
        # channel capacity as a light band on a log axis; zero capacity falls below the axis, unbounded capacity is blank
        ymin = 1e-3 * Q_s_ero[-1] * 1e-3
        cap_t = Q_s_cap * 1e-3
        cap_t = np.where(cap_t > 1e15, np.nan, np.maximum(cap_t, 1e-2 * ymin))
        ax.set_yscale('log')
        ax.fill_between(x, ymin, np.maximum(cap_t, ymin), color='b', alpha=0.08, label='channel capacity')
        ax.plot(x, cap_t, color='b', lw=0.8, alpha=0.5)
        ax.plot(x, Q_s_ero * 1e-3, 'm-',  lw=1.8, label='cumulative erosion')
        ax.plot(x, Q_s_fl  * 1e-3, 'b-',  lw=1.8, label='channel')
        ax.plot(x, Q_deform * 1e-3, color='saddlebrown', lw=1.4, label='till deformation')
        ax.plot(x[:-1], r['Q_fr'][:-1] * 1e-3, color='teal', lw=1.4, label='basal ice (fringe)')
        ax.plot(x[:-1], r['Q_d'][:-1] * 1e-3, 'g--', lw=1.2, label='dispersed')
        ax.set_ylim(ymin, 3.0 * np.nanmax(cap_t))
        if col == 0:
            leg_h, leg_l = ax.get_legend_handles_labels()
        ax.set_xlim(0, L)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Distance along flowline (m)', fontsize=16)
        if col == 0:
            ax.set_ylabel(r'$Q_s$ (t/yr)')

    if rows == 'sediment':
        # drop the two velocity rows; put the glacier titles on the erosion row
        for col in range(3):
            axes[2, col].set_title(axes[0, col].get_title(), fontsize=20)
            axes[0, col].remove(); axes[1, col].remove()
        axes = axes[2:, :]                  # the canvas keeps its five-row height

    panel_labels = 'abcdefghijklmno'
    for row in range(axes.shape[0]):
        for col in range(3):
            lbl = panel_labels[row * 3 + col]
            axes[row, col].text(0.02, 0.97, f'({lbl})', transform=axes[row, col].transAxes,
                                fontsize=20, fontweight='bold', va='top', ha='left',
                                zorder=10,
                                bbox=dict(facecolor='white', edgecolor='none',
                                          alpha=0.85, pad=1.5))

    plt.tight_layout()
    # one legend for the flux panels, below the figure so that it hides no data
    fig.legend(leg_h, leg_l, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=len(leg_l), fontsize=12, frameon=False)
    plt.savefig(FIG_DIR / f'glacier_diagnostics_{tag}.png', dpi=150, bbox_inches='tight')
    plt.savefig(FIG_DIR / f'glacier_diagnostics_{tag}.pdf', bbox_inches='tight')


def compare_glaciers():
    """Export as a fraction of the potential (unshielded) erosion against grain size D, split into channel, till deformation
    and basal ice.  The remainder is stored."""
    D_vals = np.logspace(np.log10(10e-6), np.log10(0.40), 30)
    D_vals = np.sort(np.unique(np.concatenate([D_vals, [D_FINE, D_GRAIN]])))
    D_mm   = D_vals * 1000

    glaciers = [
        ('Arolla',          load_arolla,     QW_REF,    W_AROLLA, CS_AROLLA, SHAPE_AROLLA, NX_AROLLA, '#1b7837'),
        ('Franz Josef',     load_franzjosef, QW_FJ,     W_FJ,     CS_FJ,     SHAPE_FJ,     NX_FJ,     '#2166ac'),
        ('Gornergletscher', load_gorner,     QW_GORNER, W_GORNER, CS_GORNER, SHAPE_GORNER, NX_GORNER, '#b2182b'),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)

    for ax, (name, loader, Qw, W, Cs, f, nx, col) in zip(axes, glaciers):
        print(f'\nSolving BP for {name}...')
        x, _, dx, ds, b, _, hb, dhdx, dbdx, C, d, _, U, _, _ = \
            bp_solution(loader.__name__, loader, c_slide=Cs, shape=f, nx=nx)
        u_b = U[:, 0]

        T_arr, Tfl_arr, Tdef_arr, shield_arr = [], [], [], []
        # potential (unshielded) erosion of the budget domain
        Q_pot = RHO_R * W * np.sum(K_G * np.abs(u_b[:-1])**L_EXP) * dx
        for D in D_vals:
            r   = steady_routing(u_b, Qw, dbdx, dx, x, hb, b, d_grain=D, W=W, c_slide=Cs)
            bud = budget_from_routing(r, dx)
            f_pot = bud['Q_ero'] / Q_pot
            T_arr.append(bud['T'] * f_pot)
            Tfl_arr.append(bud['T_fl'] * f_pot)
            Tdef_arr.append(bud['T_def'] * f_pot)
            shield_arr.append(f_pot)

        T     = np.array(T_arr)
        Tfl   = np.array(Tfl_arr)
        Tdef  = np.array(Tdef_arr)

        ax.fill_between(D_mm, 0,   Tfl, color=col, alpha=0.30, label='fluvial')
        ax.fill_between(D_mm, Tfl, Tfl + Tdef, color='saddlebrown', alpha=0.25,
                        label='till deformation')
        ax.fill_between(D_mm, Tfl + Tdef, T, color=col, alpha=0.10, label='basal ice (fringe + dispersed)')
        ax.plot(D_mm, T,   color=col, lw=2.0, label='total export / potential erosion')
        ax.plot(D_mm, Tfl, color=col, lw=1.2, ls='--', label='fluvial only')
        ax.plot(D_mm, shield_arr, color='saddlebrown', lw=1.2, ls='-.',
                label='actual / potential erosion (till shielding)')
        ax.axvline(D_FINE * 1000, color='darkorange', lw=1.0, ls=':', alpha=0.8,
                   label=f'fine ref.  ({D_FINE*1e6:.0f} $\\mu$m)')
        ax.axvline(D_GRAIN * 1000, color='saddlebrown', lw=1.0, ls=':', alpha=0.8,
                   label=f'coarse ref.  ({D_GRAIN*1000:.0f} mm)')

        ax.set_xscale('log')
        ax.set_ylabel('Export / potential erosion')
        ax.set_title(f'{name}  —  $Q_w$ = {Qw:.0f} m³/s', fontsize=20)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Grain size  $D$  (mm)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=10, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.06))

    for ax, lbl in zip(axes, 'abc'):
        ax.text(0.01, 0.97, f'({lbl})', transform=ax.transAxes,
                fontsize=22, fontweight='bold', va='top')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'glacier_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(FIG_DIR / 'glacier_comparison.pdf', bbox_inches='tight')
    print('Comparison figure saved.')


