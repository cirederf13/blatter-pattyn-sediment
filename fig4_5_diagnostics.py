"""Figs. 4 and 5: steady-state solution at 63 um (all rows) and at 10 mm (sediment rows)."""
import blatter_flow as bf
bf.run_glacier_diagnostics(d_grain=bf.D_FINE, tag='fine')
bf.run_glacier_diagnostics(d_grain=bf.D_GRAIN, tag='coarse', rows='sediment')
