# blatter-pattyn-sediment

Coupled Blatter-Pattyn ice-flow and subglacial sediment-transport model.

- `blatter_flow.py` — first-order (Blatter 1995; Pattyn 2003) ice-flow solver along a 1-D flowline, solved via shooting with Anderson acceleration, coupled to Engelund-Hansen bedload transport in a Röthlisberger channel, viscoplastic till remobilisation (Boulton & Hindmarsh 1987), and dispersed-layer sediment storage (Pierce et al. 2026).
- `test_blatter_flow.py` — analytical verification test suite (closed-form checks covering fluvial transport, till shielding, and dispersed-layer storage).

## Usage

```
python3 -m pytest test_blatter_flow.py -q -m "not slow"
```

The diagnostics for the three glaciers discussed in the accompanying paper (Haut Glacier d'Arolla, Franz Josef Glacier, Gornergletscher) require flowline geometry data not included in this repository; contact the authors for access.

## Reference

Herman, F., De Doncker, F. — coupled ice-dynamics and sediment-transport model of glacial erosion-export coupling.
