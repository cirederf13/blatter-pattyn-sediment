"""Gornergletscher: invert C_s(x) and the shape factor f(x) from the observed 2018 surface speed (Planet imagery, De Doncker et al.
2026).  Writes output/gorner_Cs_f.npy; the coefficients used in the paper are in data/inverted/."""
import numpy as np, blatter_flow as bf
a = np.loadtxt(bf.DATA_DIR / 'gorner' / 'gorner_flowline.csv', delimiter=',', skiprows=1)
(Cs, f), mis, sol = bf.invert_sliding(bf.load_gorner, a[:, 0], a[:, 3], nx=bf.NX_GORNER, shape=1.0, c0=26.0, n_iter=14, f_min=0.6, verbose=False)
print('final misfit factor %.2f; C_s %.1f-%.1f, f %.2f-%.2f' % (mis, Cs.min(), Cs.max(), f.min(), f.max()))
np.save(bf.OUT_DIR / 'gorner_Cs_f.npy', np.array([Cs, f]))
