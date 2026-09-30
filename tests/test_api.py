"""The CUFSM tutorial-video C section, the input forms, errors and a timing note."""

import time

import numpy as np
import pytest

import cufsm_rs as cufsm

XZ = [(5, 1), (5, 0), (2.5, 0), (0, 0), (0, 3), (0, 6), (0, 9), (2.5, 9), (5, 9), (5, 8)]
PROP = [[100, 29500, 29500, 0.3, 0.3, 11346.15]]
NODE = [[i + 1, x, z, 1, 1, 1, 1, 0.0] for i, (x, z) in enumerate(XZ)]
ELEM = [[i + 1, i + 1, i + 2, 0.1, 100] for i in range(9)]


def rel(a, b):
    return abs(a / b - 1)


@pytest.fixture
def tutorial():
    return cufsm.Model(PROP, NODE, ELEM)


def test_tutorial_first_yield(tutorial):
    cl = cufsm.first_yield(tutorial, 50, extreme_fibre=False)
    assert rel(cl.Py, 105) <= 1e-12
    assert rel(cl.Mxx, 328.25) <= 1e-9
    assert rel(cl.Mzz, 112.51375) <= 1e-9
    ef = cufsm.first_yield(tutorial, 50)
    assert rel(ef.Py, 105) <= 1e-12
    assert rel(ef.Mxx, 324.642857142857) <= 1e-9


def test_numpy_and_list_inputs_agree(tutorial):
    m2 = cufsm.Model(np.array(PROP), np.array(NODE), np.array(ELEM))
    assert cufsm.section_properties(m2).as_dict().keys() == cufsm.section_properties(tutorial).as_dict().keys()
    assert cufsm.section_properties(m2).Ixx == cufsm.section_properties(tutorial).Ixx


def test_from_dicts_matches_arrays(tutorial):
    m = cufsm.Model.from_dicts(
        nodes=[{"x": x, "z": z, "stress": 0.0} for x, z in XZ],
        elements=[{"i": i, "j": i + 1, "t": 0.1} for i in range(9)],
        materials=[{"Ex": 29500, "Ey": 29500, "vx": 0.3, "vy": 0.3, "G": 11346.15}],
    )
    np.testing.assert_array_equal(m.node[:, 1:], tutorial.node[:, 1:])
    assert cufsm.first_yield(m, 50).Mxx == cufsm.first_yield(tutorial, 50).Mxx


def test_stress_and_back(tutorial):
    m = cufsm.stress(tutorial, P=10.0, Mxx=5.0, restrained=True)
    assert m is not tutorial and np.all(tutorial.stress == 0)
    fit = cufsm.stress_to_action(m)
    assert rel(fit.P, 10.0) < 1e-12 and rel(fit.M11, 5.0) < 1e-12


def test_constraints_and_springs_forms(tutorial):
    m = cufsm.stress(tutorial, P=1.0)
    free = cufsm.strip(m, [5.0, 50.0], neigs=1).curve
    sprung = cufsm.Model.from_dicts(
        nodes=[{"x": x, "z": z, "stress": s} for (x, z), s in zip(XZ, m.stress)],
        elements=[{"i": i, "j": i + 1, "t": 0.1} for i in range(9)],
        E=29500, nu=0.3,
        springs=[{"i": 8, "ku": 10.0, "kw": 10.0}],
        constraints=[{"node_e": 0, "dof_e": "x", "coeff": 1.0, "node_k": 9, "dof_k": "x"}],
    )
    assert np.all(cufsm.strip(sprung, [5.0, 50.0], neigs=1).curve >= free * (1 - 1e-9))


def test_bad_input_raises_value_error(tutorial):
    with pytest.raises(ValueError, match="columns"):
        cufsm.Model(PROP, [r[:7] for r in NODE], ELEM)
    with pytest.raises(ValueError, match="node 42"):
        cufsm.Model(PROP, NODE, ELEM + [[10, 10, 42, 0.1, 100]]).section_properties()
    with pytest.raises(ValueError, match="material 7"):
        cufsm.section_properties(cufsm.Model(PROP, NODE, [[1, 1, 2, 0.1, 7]]))
    with pytest.raises(ValueError, match="boundary condition"):
        cufsm.strip(tutorial, [10.0], bc="X-Y")
    with pytest.raises(ValueError, match="positive length"):
        cufsm.strip(tutorial, [-1.0])
    with pytest.raises(ValueError, match="spaces"):
        cufsm.strip(tutorial, [10.0], spaces="Q")
    with pytest.raises(ValueError):
        cufsm.first_yield(tutorial, -5)


def test_multiple_terms_and_bcs(tutorial):
    m = cufsm.stress(tutorial, P=1.0)
    r = cufsm.strip(m, [50.0, 100.0], m_all=5, bc="C-C", neigs=4)
    assert r.load_factors.shape == (2, 4)
    assert r.modes.shape == (2, 4, 4 * 10 * 5)
    assert np.all(np.diff(r.load_factors, axis=1) >= 0)


def test_templates():
    c = cufsm.lipped_c(200, 76, 15, 1.9, ri=3.0)
    z = cufsm.lipped_z(200, 76, 15, 1.9, ri=3.0)
    assert len(c.node) == len(z.node) == 37  # 12 web + 2x6 flange + 2x2 lip + 4x2 corner strips
    p = cufsm.section_properties(c)
    assert 0 < p.A < 200 * 1.9 * 3
    t = cufsm.template("C", h=9, b1=5, d1=1, t=0.1)  # centreline, sharp
    np.testing.assert_allclose(t.node[[0, -1], 1:3], [[5, 1], [5, 8]])


def test_signature_timing_lipped_c(capsys):
    m = cufsm.stress(cufsm.lipped_c(200, 76, 15, 1.9, ri=3.0), P=1.0)
    L = np.logspace(1, 4, 90)
    t0 = time.perf_counter()
    sig = cufsm.signature(m, L)
    dt = time.perf_counter() - t0
    with capsys.disabled():
        print(f"\n[timing] signature, lipped C 200x76x15x1.9 r3 ({len(m.node)} nodes), 90 lengths: {dt*1e3:.1f} ms; "
              f"minima {np.round(sig.minima, 3).tolist()}")
    assert len(sig.minima) >= 2


def test_elem_always_has_the_material_column():
    c = cufsm.lipped_c(200, 76, 15, 1.9, ri=3.0)
    assert c.elem.shape[1] == 5 and np.all(c.elem[:, 4] == c.prop[0, 0])
    four = cufsm.Model(PROP, NODE, [r[:4] for r in ELEM])
    assert four.elem.shape[1] == 5 and np.all(four.elem[:, 4] == 100)
    assert cufsm.first_yield(four, 50).Mxx == cufsm.first_yield(cufsm.Model(PROP, NODE, ELEM), 50).Mxx


def test_names():
    assert cufsm.__name__ == "cufsm_rs"
    assert cufsm.MechanismError.__module__ == "cufsm_rs"
