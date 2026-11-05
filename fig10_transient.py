"""Figure for the ELA-forced transient (63 um): export, storage-derived fraction, Da_sed, median sliding speed."""
import numpy as np
import matplotlib.pyplot as plt
import blatter_flow as bf

GL = [('arolla', "Haut Glacier d'Arolla", '#1b7837'),
      ('franzjosef', 'Franz Josef Glacier', '#2166ac'),
      ('gorner', 'Gornergletscher', '#b2182b')]


def median_speed(d):
    """Median basal sliding speed over the ice-covered nodes (hb > 1 m) at each year."""
    return np.array([np.median(np.abs(u[h > 1.0])) if np.any(h > 1.0) else np.nan for u, h in zip(d['u_b'], d['hb'])])


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    (a_exp, a_stor), (a_da, a_u) = axes
    for name, label, col in GL:
        try:
            d = np.load(bf.OUT_DIR / f'transient_{name}.npz')
        except FileNotFoundError:
            continue
        t = d['t']; gl = 'fine'
        a_exp.plot(t, d[f'{gl}_Q_out'] / d[f'{gl}_Q_out'][0], color=col, lw=1.8, label=label)
        a_exp.plot(t, d[f'{gl}_Q_ero'] / d[f'{gl}_Q_ero'][0], color=col, lw=1.0, ls=':')
        a_stor.plot(t, d[f'{gl}_f_stor'], color=col, lw=1.8)
        Qo = d[f'{gl}_Q_out']     # residence time from the export; masked where export vanishes
        tau = np.where(Qo > 0.01 * Qo[0], d[f'{gl}_V_stor'] * bf.RHO_SED / np.maximum(Qo, 1e-30), np.nan)
        a_da.plot(t, tau / 50.0, color=col, lw=1.8)
        u = median_speed(d)
        ratio = d[f'{gl}_Q_out'] / d[f'{gl}_Q_ero']
        print(f"{label}: volume loss {100 * (1 - d['V'][-1] / d['V'][0]):.0f} %, retreat {(d['L'][0] - d['L'][-1]) / 1e3:.1f} km, "
              f"erosion end/initial {d[f'{gl}_Q_ero'][-1] / d[f'{gl}_Q_ero'][0]:.2f}, export/erosion {ratio[0]:.2f} at t=0, peak {ratio.max():.2f} at {t[ratio.argmax()]:.0f} yr, "
              f"{ratio[-1]:.2f} at the end, stored fraction {d[f'{gl}_f_stor'][0]:.2f} -> {d[f'{gl}_f_stor'][10]:.2f} (10 yr), "
              f"Da_sed {tau[0] / 50.0:.2f} -> {tau[-1] / 50.0:.2f}, discharge end/initial {d['Qw'][-1] / d['Qw'][0]:.2f}")
        a_u.plot(t, u / u[0], color=col, lw=1.8, label=f'{u[0]:.0f} m/yr initially')
    a_exp.set_ylabel('export / initial'); a_stor.set_ylabel('storage-derived fraction')
    a_stor.axhline(0.35, color='0.4', ls='-.', lw=1.0)
    a_stor.text(52, 0.36, 'De Doncker et al. 2026 (Gorner, 35 %)', fontsize=8, color='0.3')
    a_da.set_ylabel(r'$\mathrm{Da_{sed}}$'); a_da.set_yscale('log'); a_da.axhline(1.0, color='k', ls=':', lw=1.0)
    a_u.set_ylabel('median sliding speed / initial')
    for ax in (a_da, a_u):
        ax.set_xlabel('years (ELA rises 0–50 yr, then held)', fontsize=12)
    for k, ax in enumerate(axes.flat):
        ax.axvspan(0, 50, color='0.9', zorder=0); ax.grid(alpha=.3)
        ax.text(.01, .95, f'({"abcd"[k]})', transform=ax.transAxes, fontsize=13, fontweight='bold', va='top')
    a_exp.legend(fontsize=10, loc='upper left', bbox_to_anchor=(0.09, 0.97))
    a_u.legend(fontsize=9, loc='lower left', title='median speed at t = 0', title_fontsize=9)
    plt.tight_layout()
    plt.savefig(bf.FIG_DIR / 'transient_ela.png', dpi=150, bbox_inches='tight')
    plt.savefig(bf.FIG_DIR / 'transient_ela.pdf', bbox_inches='tight')
    print('saved transient_ela.png/.pdf')


if __name__ == '__main__':
    main()
