# Changelog

## 0.1.0 (unreleased)

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
