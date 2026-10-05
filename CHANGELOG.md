# Changelog

## 0.1.1 (2026-10-05)

Built on cufsm-rs 0.4.2.

- A model with a node that belongs to no element is now refused by every function with
  `ValueError("invalid model: node N (node row i) belongs to no element")`, naming the node by
  its CUFSM node number. Before, a cFSM run on such a model (`strip` with `spaces`, `classify`)
  panicked in the engine and raised `pyo3_runtime.PanicException`, which `except ValueError`
  (or `except Exception`) does not catch.
- The same model also changes elsewhere: a free `strip` or `signature` used to raise
  `MechanismError` ("not positive definite"), and `section_properties`, `stress`, `first_yield`
  and `stress_to_action` used to return results. All now raise the `ValueError` above.
  `MechanismError` is a subclass of `ValueError`, so code that catches `ValueError` sees no
  difference on the analyses.
- Nothing else in cufsm-rs 0.4.2 changes what this package computes: the rest of that release
  is new exports in its C interface, which this package does not use.
- README: says that this package is the CPython build of cufsm-rs, and that a page in the
  browser runs the WebAssembly build, not this package.

## 0.1.0 (2026-10-01)

First release: Python bindings for cufsm-rs 0.4.1, the Rust port of CUFSM.

- Models as CUFSM arrays (`prop`, `node`, `elem`, `constraints`, `springs`), from dicts, or
  from CUFSM's C and Z templates (`template`, `lipped_c`, `lipped_z`, `plain_c`).
- Section properties (grosprop and cutwp_prop2), reference stresses from actions (stresgen and
  warp_stress), first-yield actions (centreline or element faces, and the bimoment), and
  stress_to_action.
- `strip` (stripmain) for all five end conditions with any longitudinal terms, fixed DOFs,
  constraints and springs, and cFSM-restricted solves (`spaces="D"` and so on); `signature` for
  the signature curve and its refined minima; `classify` for cFSM G, D, L, O classification.
- One result class, `StripResult`, for both `strip` and `signature`, with
  `mode_shape(i_length, k_mode)` giving per-node `u`, `v`, `w`, `theta` arrays per longitudinal
  term (the raw CUFSM-order vector stays on `dofs`) and `classify_minima`.
- `cufsm_rs.plot` (optional, `pip install cufsm-rs-py[plot]`): `plot_section`,
  `plot_signature` and `plot_mode`.
- HTML reprs for Jupyter; type hints and `py.typed`.
- abi3 wheels for CPython 3.9 and later.
