# cufsm-rs-py

CUFSM, the finite strip method for the elastic buckling of thin-walled sections, in Python.

```sh
pip install cufsm-rs-py            # the engine (needs only NumPy)
pip install "cufsm-rs-py[plot]"    # plus matplotlib, for cufsm_rs.plot
```

```python
import cufsm_rs
```

CUFSM is by B.W. Schafer and co-workers at Johns Hopkins University
([www.ce.jhu.edu/cufsm](https://www.ce.jhu.edu/cufsm/)). This package is an independent port,
not affiliated with or endorsed by the CUFSM authors.

## Why this exists

[cufsm-rs](https://github.com/mageaustralia/cufsm-rs) is a Rust port of CUFSM. Its WebAssembly
build runs CUFSM in the browser, in
[CivilKit Buckling](https://mageengineering.com.au/apps/buckling/), with no Python involved.

This package is the other build of the same engine: a native extension for CPython (3.9 and
later), for scripts and notebooks on your own machine, so a script and the web app give the same
numbers. It needs NumPy, and it does not run in the browser, under Pyodide or under MicroPython.
The results are checked against CUFSM's own MATLAB code run under Octave (see [Tests](#tests)).

## Units

The package is unit-agnostic, as CUFSM is: use any consistent set, for example N, mm and MPa
(moments in N mm) or kip, in and ksi. Results come back in the same set.

A **load factor** is a multiplier on the model's reference stresses: the section buckles when
the reference stresses are multiplied by it. Set the reference stresses from the squash load
(`stress(model, P=Py)`) and a load factor reads directly as `Pcr / Py`. Positive stress is
compression.

## Quickstart

```python
import numpy as np
import cufsm_rs as fsm

# CUFSM's tutorial C: 9 x 5 x 1 in, t = 0.1 in (inches and ksi)
xz = [(5, 1), (5, 0), (2.5, 0), (0, 0), (0, 3), (0, 6), (0, 9), (2.5, 9), (5, 9), (5, 8)]
m = fsm.Model(
    prop=[[100, 29500, 29500, 0.3, 0.3, 11346.15]],
    node=[[i + 1, x, z, 1, 1, 1, 1, 0] for i, (x, z) in enumerate(xz)],
    elem=[[i + 1, i + 1, i + 2, 0.1, 100] for i in range(9)],
)

p = fsm.section_properties(m)          # p.A, p.Ixx, p.J, p.Cw, p.xs, ... (also p["Ixx"])
y = fsm.first_yield(m, fy=50)          # y.Py = 105, y.Mxx = 324.64 (element faces, as CUFSM)

mc = fsm.stress(m, P=y.Py)             # a new Model with the reference stresses set
sig = fsm.signature(mc, np.logspace(0, 3, 80))
sig                                    # StripResult(signature, bc=S-S, 80 lengths 1 to 1000, neigs=1, 2 minima)
sig.minima                             # [[7.27, 0.353], [43.9, 0.544]]: [half-wavelength, load factor]
sig.classify_minima()                  # [G, D, L, O] percent of each minimum's mode: L 98%, D 94%

r = fsm.strip(mc, [7.27, 43.9], neigs=3)   # any lengths, several modes
r.load_factors                             # (2, 3)
shape = r.mode_shape(0)                    # the lowest mode at the first length
shape.u, shape.v, shape.w, shape.theta     # each (terms, nodes)
shape.dofs                                 # the raw vector, CUFSM's DOF order
shape.at()                                 # displacements summed over terms at mid-length

dist = fsm.strip(mc, [43.9], spaces="D", neigs=1)                  # pure distortional (cFSM)
cc = fsm.strip(mc, [100.0, 200.0], m_all=10, bc="C-C", neigs=2)    # general end conditions
cc.classify()                                                       # (2, 2, 4): G, D, L, O percent
```

`lipped_c`, `lipped_z`, `plain_c` (outside dimensions and inside radius, default steel in MPa)
and `template` (CUFSM's templatecalc) build models for you:

```python
lc = fsm.lipped_c(200, 76, 15, 1.9, ri=3)    # mm; 37 nodes
```

The notebook [examples/quickstart.ipynb](https://github.com/mageaustralia/cufsm-rs-py/blob/main/examples/quickstart.ipynb) walks through all of this,
with plots.

## Models

Models use CUFSM's arrays, in CUFSM's column order and with its 1-based node and material
numbers. x and z are the cross-section coordinates; y runs along the member.

| array | columns |
|---|---|
| `prop` | `[mat#, Ex, Ey, vx, vy, G]` |
| `node` | `[node#, x, z, xdof, zdof, ydof, qdof, stress]` (1 = free, 0 = fixed) |
| `elem` | `[elem#, nodei, nodej, t, mat#]` (`mat#` optional) |
| `constraints` | `[node#e, dofe, coeff, node#k, dofk]` |
| `springs` | `[#, nodei, nodej, ku, kv, kw, kq, local, discrete, ys]` (nodej 0 = ground) |

`Model.from_dicts(...)` builds the same arrays from lists of dicts with 0-based indices. The
arrays are NumPy and editable in place.

## Results

`strip()` and `signature()` both return a `StripResult`:

| attribute | |
|---|---|
| `lengths` | `(nlengths,)` half-wavelengths (signature, S-S) or member lengths |
| `load_factors` | `(nlengths, neigs)`, smallest first, NaN where fewer were found |
| `curve` | the lowest load factor at each length |
| `modes` | `(nlengths, neigs, ndof)` raw mode vectors, CUFSM's DOF order |
| `m_terms` | the longitudinal terms at each length |
| `minima` | `(n, 2)` `[length, load factor]` at each local minimum (signature only; empty for strip) |
| `kind`, `bc`, `model` | `"signature"` or `"strip"`, the end conditions, the analysed model |

Methods: `mode_shape(i_length, k_mode=0)`, `classify()`, `classify_minima()`.

**Mode shapes.** `mode_shape` returns a `ModeShape` with `u` (along x), `v` (along y, the
member), `w` (along z) and `theta` (rotation, anticlockwise with x right and z up), each of
shape `(nterms, nnodes)`, and the raw vector as `dofs`. The raw vector is CUFSM's order: for
each longitudinal term in turn, a block of `4 * nnodes` entries, first `u` and `v` interleaved
for every node (`u1 v1 u2 v2 ...`), then `w` and `theta` interleaved (`w1 theta1 w2 theta2 ...`).
Modes are scaled so their largest entry is +1. `shape.at(y)` sums the terms with the end
conditions' longitudinal shape functions at `y` along the member.

## Plotting

`pip install "cufsm-rs-py[plot]"`, then:

```python
from cufsm_rs.plot import plot_section, plot_signature, plot_mode

plot_section(mc, node_numbers=True)     # elements to thickness, nodes, stress bands
plot_signature(sig, classify=True)      # log-x curve, minima labelled with G/D/L/O
plot_mode(sig, i_length=20)             # deformed over undeformed cross-section
```

Each takes an optional `ax` and returns the matplotlib Axes. `plot_mode` scales the mode so its
largest in-plane displacement is 10% of the section size, whatever the units (`scale=` to
override). `import cufsm_rs` never imports matplotlib.

## Jupyter

`Model`, `SectionProperties`, `YieldActions` and `StripResult` render as small HTML tables in a
notebook (big models are truncated). There is nothing extra to install.

## Errors and threads

Invalid input raises `ValueError`, and the message names the bad row. A model with a mechanism
raises `cufsm_rs.MechanismError`, a subclass of `ValueError`. The long solves (`strip`,
`signature`, `classify`) release the GIL, and cufsm-rs spreads the lengths across threads.

## Install from source

You need Rust (1.75 or later) and Python 3.9 or later. The Rust engine comes from crates.io
(`cufsm-rs = "0.4.1"`).

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --release --extras dev
pytest -q
```

The build is abi3, so one wheel covers CPython 3.9 and later.

## Tests

`tests/test_oracle.py` compares against CUFSM's own MATLAB run under Octave, the same oracle
fixture the cufsm-rs parity tests use: properties, stresgen stresses, centreline and face first
yield, bimoment stress, stress_to_action, load factors, mode MAC, cFSM classification and
restricted load factors, at the Rust tolerances. The fixture ships with the cufsm-rs source
(`tests/fixtures/cufsm_octave.json`). It is looked for at `../cufsm-rs/tests/fixtures/` next to
this checkout; set `CUFSM_ORACLE` to its path otherwise. Without it those tests skip and say so.
The other tests cover the API, mode shapes, plotting (Agg backend) and the HTML reprs.

## Citation

If you publish results, cite CUFSM as its authors ask:

- Schafer, B.W., Ádány, S., Li, Z., Jin, S. CUFSM v5.66. DOI 10.5281/zenodo.17771486.
- For general end conditions: Schafer, B.W., Li, Z. "Buckling analysis of cold-formed steel
  members with general boundary conditions using CUFSM: conventional and constrained finite
  strip methods." 20th International Specialty Conference on Cold-Formed Steel Structures,
  2010, pp. 17-32.

## Related projects

- [CUFSM](https://www.ce.jhu.edu/cufsm/) itself (MATLAB), by B.W. Schafer and co-workers.
- [pyCUFSM](https://github.com/ClearCalcs/pyCUFSM) by ClearCalcs, a pure-Python port of CUFSM
  and prior work in this space that this project gratefully acknowledges.
- [cufsm-rs](https://github.com/mageaustralia/cufsm-rs), the Rust engine behind this package.

## License

MIT, see [LICENSE](https://github.com/mageaustralia/cufsm-rs-py/blob/main/LICENSE). CUFSM is MIT-licensed, and its copyright notice is kept there.
