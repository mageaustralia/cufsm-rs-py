# cufsm (Python)

Python bindings for [cufsm-rs](../cufsm-rs), a Rust port of CUFSM, the finite strip method for
elastic buckling of thin-walled sections (Schafer et al., Johns Hopkins University). This is an
independent port, not affiliated with or endorsed by the CUFSM authors. If you publish results,
cite CUFSM: Schafer, B.W., Ádány, S., Li, Z., Jin, S. CUFSM v5.66. DOI 10.5281/zenodo.17771486.

Status: a local prototype. Nothing has been published.

## Install from source

You need Rust (1.75 or later) and Python 3.9 or later. The crate depends on `../cufsm-rs` by path.

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install maturin pytest numpy
maturin develop --release
pytest -q
```

The build is abi3, so one wheel covers CPython 3.9 and later.

## Models

Models use CUFSM's arrays, in CUFSM's column order and with its 1-based node and material
numbers:

| array | columns |
|---|---|
| `prop` | `[mat#, Ex, Ey, vx, vy, G]` |
| `node` | `[node#, x, z, xdof, zdof, ydof, qdof, stress]` (1 = free) |
| `elem` | `[elem#, nodei, nodej, t, mat#]` (`mat#` optional) |
| `constraints` | `[node#e, dofe, coeff, node#k, dofk]` |
| `springs` | `[#, nodei, nodej, ku, kv, kw, kq, local, discrete, ys]` (nodej 0 = ground) |

`Model.from_dicts(...)` builds the same arrays from lists of dicts with 0-based indices.
`lipped_c`, `lipped_z`, `plain_c` and `template` build models from CUFSM's templates.

## Example: the CUFSM tutorial C section

```python
import numpy as np
import cufsm

xz = [(5, 1), (5, 0), (2.5, 0), (0, 0), (0, 3), (0, 6), (0, 9), (2.5, 9), (5, 9), (5, 8)]
m = cufsm.Model(
    prop=[[100, 29500, 29500, 0.3, 0.3, 11346.15]],
    node=[[i + 1, x, z, 1, 1, 1, 1, 0] for i, (x, z) in enumerate(xz)],
    elem=[[i + 1, i + 1, i + 2, 0.1, 100] for i in range(9)],
)

p = cufsm.section_properties(m)          # A, xcg, zcg, Ixx, Izz, Ixz, thetap, I11, I22, J, xs, zs, Cw, B1, B2, wn
y = cufsm.first_yield(m, fy=50)           # Py=105, Mxx=324.64 at the element faces
yc = cufsm.first_yield(m, 50, extreme_fibre=False)   # Mxx=328.25, Mzz=112.51 at the centreline

mc = cufsm.stress(m, P=y.Py)              # a new Model with the reference stresses set
sig = cufsm.signature(mc)                 # CUFSM signature_ss: 100 log-spaced half-wavelengths
sig.lengths, sig.curve, sig.minima        # numpy arrays; minima rows are [length, load factor]

r = cufsm.strip(mc, np.logspace(0, 3, 60), bc="S-S", neigs=5)
r.load_factors                            # (60, 5)
r.modes                                   # (60, 5, 4*nnodes*nterms), CUFSM DOF order
gdlo = r.classify()                       # (60, 5, 4): cFSM G, D, L, O percent

dist = cufsm.strip(mc, [20.0], spaces="D", neigs=1)   # pure distortional (cFSM)
cc = cufsm.strip(mc, [100.0], m_all=10, bc="C-C")     # general end conditions, terms 1..10
```

Invalid input raises `ValueError`, and the message names the bad row. A model with a mechanism
raises `cufsm.MechanismError`, which is a subclass of `ValueError`. The long solves (`strip`,
`signature`, `classify`) release the GIL, and cufsm-rs spreads the lengths across threads.

## Tests

`tests/test_oracle.py` reads `../cufsm-rs/tests/fixtures/cufsm_octave.json` (set `CUFSM_ORACLE`
to point somewhere else). This fixture holds CUFSM's own MATLAB run under Octave, and it is the
same oracle the Rust parity tests use. The tests check properties, the stresses from stresgen,
centreline and face first yield, bimoment stress, stress_to_action, load factors, mode MAC,
cFSM classification and restricted load factors. The tolerances are the Rust ones.
`tests/test_api.py` covers the tutorial section, the input forms, errors and timing.
