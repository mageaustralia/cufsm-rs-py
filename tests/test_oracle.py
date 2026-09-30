"""Cross-checks against the CUFSM/Octave oracle that cufsm-rs's own parity tests use.

The fixture is read from the sibling cufsm-rs checkout (override with CUFSM_ORACLE).
Tolerances follow tests/cufsm_parity.rs and tests/cfsm_parity.rs.
"""

import json
import os
import pathlib
import numpy as np
import pytest

import cufsm_rs as cufsm

ORACLE = pathlib.Path(
    os.environ.get(
        "CUFSM_ORACLE",
        pathlib.Path(__file__).resolve().parents[2] / "cufsm-rs/tests/fixtures/cufsm_octave.json",
    )
)
pytestmark = pytest.mark.skipif(not ORACLE.exists(), reason=f"oracle fixture not found at {ORACLE}")


@pytest.fixture(scope="module")
def cases():
    return {c["name"]: c for c in json.loads(ORACLE.read_text())}


def model_of(c):
    E, nu = c["E"], c["nu"]
    cons = c["constraints"] if isinstance(c["constraints"], list) else None
    spr = c["springs"] if isinstance(c["springs"], list) and c["springs"] else None
    # The oracle's elem rows have no mat# column: every strip is material 1.
    return cufsm.Model([[1, E, E, nu, nu, E / (2 * (1 + nu))]], c["node"], c["elem"], cons, spr)


def restrained(c):
    return c["actions"]["unsymm"] == 0


PROPS_CASES = [
    "lipped-c compression",
    "lipped-c rounded compression",
    "lipped-z unrestrained bending",
    "hat compression",
    "equal angle compression",
    "outside-dims lipped-z compression",
]


@pytest.mark.parametrize("name", PROPS_CASES)
def test_section_properties(cases, name):
    c = cases[name]
    p = cufsm.section_properties(model_of(c))
    for k in ["A", "xcg", "zcg", "Ixx", "Izz", "Ixz", "I11", "I22", "thetap"]:
        w = c["props"][k]
        # as cufsm_parity.rs: 1e-12 relative to max(|w|, A)
        assert abs(p[k] - w) <= 1e-12 * max(abs(w), p.A), f"{k}: {p[k]} vs CUFSM {w}"
    # cutwp_prop2 against the cFSM oracle where present (cfsm_parity.rs holds it to 1e-10)
    if "cutwp" in c.get("cfsm", {}):
        cw = c["cfsm"]["cutwp"]
        for k, w in [("J", cw["J"]), ("xs", cw["xs"]), ("zs", cw["zs"]), ("Cw", cw["Cw"]), ("B2", cw["B2"])]:
            assert abs(p[k] - w) <= 1e-10 * max(abs(w), 1.0), f"{k}: {p[k]} vs {w}"


@pytest.mark.parametrize("name", ["cfsm lipped-c", "cfsm lipped-z", "cfsm hat"])
def test_cutwp_properties(cases, name):
    c = cases[name]
    p = cufsm.section_properties(model_of(c))
    cw = c["cfsm"]["cutwp"]
    for k in ["J", "xs", "zs", "Cw", "B1", "B2"]:
        assert abs(p[k] - cw[k]) <= 1e-10 * max(abs(cw[k]), 1.0), f"{k}: {p[k]} vs {cw[k]}"
    np.testing.assert_allclose(p.wn, cw["wn"], rtol=1e-10, atol=1e-10 * np.max(np.abs(cw["wn"])))


YIELD_CASES = [
    "lipped-c compression",
    "lipped-c rounded major bending",
    "lipped-z unrestrained bending",
    "lipped-z principal bending",
    "hat bending",
    "outside-dims lipped-c compression",
]


def _close(g, w, rel=1e-12):
    return abs(g - w) <= rel * max(abs(w), 1.0)


@pytest.mark.parametrize("name", YIELD_CASES)
def test_first_yield_extreme_fibre(cases, name):
    c = cases[name]
    y = c["yield_ext"]
    got = cufsm.first_yield(model_of(c), y["fy"], restrained=restrained(c), extreme_fibre=True)
    for k in ["Py", "Mxx", "Mzz", "M11", "M22", "B"]:
        w = y[k] if y[k] is not None else float("inf")
        if abs(w) > 1e15:
            assert abs(got[k]) > 1e15 or got[k] == 0.0
            continue
        assert _close(got[k], w), f"{k}: {got[k]} vs CUFSM {w}"


@pytest.mark.parametrize("name", YIELD_CASES[:3])
def test_first_yield_centreline(cases, name):
    c = cases[name]
    y = c["yield"]
    got = cufsm.first_yield(model_of(c), y["fy"], restrained=restrained(c), extreme_fibre=False)
    for k in ["Py", "Mxx", "Mzz", "M11", "M22"]:
        assert _close(got[k], y[k]), f"{k}: {got[k]} vs CUFSM {y[k]}"


@pytest.mark.parametrize("name", ["lipped-c compression", "lipped-c major bending", "lipped-z unrestrained bending"])
def test_stresgen_reproduces_oracle_stresses(cases, name):
    c = cases[name]
    m = model_of(c)
    a = c["actions"]
    s = cufsm.stress(m, P=a["P"], Mxx=a["Mxx"], Mzz=a["Mzz"], M11=a["M11"], M22=a["M22"],
                     restrained=restrained(c), as_array=True)
    want = np.array(c["node"])[:, 7]
    assert np.max(np.abs(s - want)) <= 1e-12 * np.max(np.abs(want))


def test_bimoment_stress_and_stress_to_action(cases):
    c = cases["lipped-c compression"]
    m = model_of(c)
    # stress() rebuilds from the given actions (B only here): the oracle added B w / Cw onto
    # the P stress already in the model, so add that back before comparing.
    s = cufsm.stress(m, B=c["warp"]["B"], as_array=True)
    sp = m.stress
    np.testing.assert_allclose(s + sp, c["warp"]["stress"], rtol=1e-12, atol=1e-12)
    fit = cufsm.stress_to_action(m.with_stress(c["warp"]["stress"]))
    s2a = c["s2a"]
    assert _close(fit.P, s2a["P"], 1e-9)
    assert _close(fit.B, s2a["B"], 1e-9)


def _load_factor_check(c, max_length=None):
    m = model_of(c)
    L = np.atleast_1d(c["lengths"]).astype(float)
    m_all = [np.atleast_1d(x).tolist() for x in c["m_all"]]
    r = cufsm.strip(m, L, m_all, bc=c["bc"], neigs=c["neigs"])
    n = 0
    worst = 0.0
    for i, w in enumerate(c["load_factors_dense"]):
        if max_length is not None and L[i] >= max_length:
            continue
        w = np.atleast_1d(w).astype(float)
        g = r.load_factors[i, : len(w)]
        assert not np.any(np.isnan(g)), f"length {L[i]}: fewer load factors than CUFSM"
        rel = np.abs(g / w - 1)
        # cufsm_parity.rs: (1e-10 + 1e-12 cond) * lf/lf1; the well-conditioned part is held to 1e-9
        tol = 1e-9 * np.maximum(w / w[0], 1.0)
        assert np.all(rel <= tol), f"length {L[i]}: {g} vs CUFSM {w}"
        worst = max(worst, rel.max())
        n += len(w)
    return n, worst


@pytest.mark.parametrize(
    "name",
    ["lipped-c C-C major bending", "lipped-c C-C springs", "cfsm lipped-c", "lipped-c S-C compression", "ss plate compression"],
)
def test_load_factors_every_length(cases, name):
    n, worst = _load_factor_check(cases[name])
    assert n > 0


@pytest.mark.parametrize(
    "name",
    ["lipped-c compression", "lipped-c major bending", "lipped-z unrestrained bending", "hat compression",
     "lipped-c constrained compression", "lipped-c grounded springs", "lipped-c node springs"],
)
def test_load_factors_local_distortional_range(cases, name):
    # Global modes at long lengths are condition-limited (see cufsm-rs analysis.rs); the Rust
    # tests scale the tolerance by cond(K). Below 1000 mm every case here is well conditioned.
    n, worst = _load_factor_check(cases[name], max_length=1000.0)
    assert n > 50


def test_first_mode_shape_mac(cases):
    c = cases["lipped-c compression"]
    m = model_of(c)
    L = np.atleast_1d(c["lengths"]).astype(float)
    r = cufsm.strip(m, L, None, bc="S-S", neigs=c["neigs"])
    compared = 0
    for i, want in enumerate(c["mode1"]):
        w = c["load_factors_dense"][i]
        if not want or abs(w[1] / w[0] - 1) < 1e-6 or abs(c["load_factors"][i][0] / w[0] - 1) > 1e-6:
            continue
        a, b = r.modes[i, 0], np.asarray(want)
        mac = (a @ b) ** 2 / ((a @ a) * (b @ b))
        assert 1 - mac <= 1e-8
        compared += 1
    assert compared > 10


def test_signature_matches_strip_ss(cases):
    c = cases["lipped-c compression"]
    m = model_of(c)
    L = np.atleast_1d(c["lengths"]).astype(float)
    sig = cufsm.signature(m, L, neigs=1)
    w0 = np.array([np.atleast_1d(x)[0] for x in c["load_factors_dense"]])
    short = L < 1000
    np.testing.assert_allclose(sig.curve[short], w0[short], rtol=1e-9)
    assert sig.minima.shape[1] == 2 and len(sig.minima) >= 1


@pytest.mark.parametrize("name", ["cfsm lipped-c", "cfsm lipped-z", "cfsm plain channel"])
def test_classification_matches_cufsm(cases, name):
    c = cases[name]
    m = model_of(c)
    L = np.atleast_1d(c["lengths"]).astype(float)
    r = cufsm.strip(m, L, [np.atleast_1d(x).tolist() for x in c["m_all"]], bc=c["bc"], neigs=3)
    got = r.classify()  # CUFSM defaults: axial, vector, ST
    compared = 0
    for i, want in enumerate(c["cfsm"]["classification"]):
        lfs = r.load_factors[i]
        for q, w in enumerate(want):
            distinct = (q == 0 or abs(lfs[q] / lfs[q - 1] - 1) > 1e-6) and (
                q + 1 >= len(lfs) or abs(lfs[q + 1] / lfs[q] - 1) > 1e-6
            )
            if not distinct:
                continue
            assert np.all(np.abs(got[i, q] - w) < 1e-6), f"L={L[i]} mode {q+1}: {got[i, q]} vs {w}"
            compared += 1
    assert compared >= 6


@pytest.mark.parametrize("space,key", [("G", "lf_G"), ("D", "lf_D"), ("L", "lf_L")])
def test_restricted_load_factors(cases, space, key):
    c = cases["cfsm lipped-c"]
    m = model_of(c)
    L = np.atleast_1d(c["lengths"]).astype(float)
    want = c["cfsm"][key]
    r = cufsm.strip(m, L, None, bc="S-S", neigs=len(want[0]), spaces=space)
    for i, w in enumerate(want):
        w = np.atleast_1d(w)
        np.testing.assert_allclose(r.load_factors[i, : len(w)], w, rtol=1e-8)
