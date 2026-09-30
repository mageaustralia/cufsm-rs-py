"""The result class, mode shapes (CUFSM's DOF order) and the package surface."""

import pathlib
import re
import subprocess
import sys

import numpy as np
import pytest

import cufsm_rs as cufsm


def test_signature_and_strip_return_one_class(tutorial_loaded, lengths):
    sig = cufsm.signature(tutorial_loaded, lengths)
    st = cufsm.strip(tutorial_loaded, lengths, neigs=3)
    assert type(sig) is type(st) is cufsm.StripResult
    assert sig.kind == "signature" and st.kind == "strip"
    assert sig.minima.shape[1] == 2 and len(sig.minima) == 2
    assert st.minima.shape == (0, 2)
    assert sig.neigs == 1 and st.neigs == 3
    np.testing.assert_allclose(sig.curve, st.curve, rtol=1e-12)
    assert st.load_factors.shape == (60, 3) and st.modes.shape == (60, 3, 40)


def test_mode_shape_matches_cufsm_dof_order(lipped):
    r = cufsm.strip(lipped, [100.0, 2000.0], m_all=3, bc="C-C", neigs=2)
    n = len(lipped.node)
    for i in range(2):
        for k in range(2):
            ms = r.mode_shape(i, k)
            assert ms.u.shape == ms.v.shape == ms.w.shape == ms.theta.shape == (3, n)
            np.testing.assert_array_equal(ms.dofs, r.modes[i, k])
            assert np.max(np.abs(ms.dofs)) == pytest.approx(1.0)
            for t in range(3):
                o = 4 * n * t
                np.testing.assert_array_equal(ms.u[t], ms.dofs[o : o + 2 * n : 2])
                np.testing.assert_array_equal(ms.v[t], ms.dofs[o + 1 : o + 2 * n : 2])
                np.testing.assert_array_equal(ms.w[t], ms.dofs[o + 2 * n : o + 4 * n : 2])
                np.testing.assert_array_equal(ms.theta[t], ms.dofs[o + 2 * n + 1 : o + 4 * n : 2])
            assert ms.bc == "C-C" and ms.length == r.lengths[i] and ms.load_factor == r.load_factors[i, k]
            np.testing.assert_array_equal(ms.m_terms, [1, 2, 3])


def test_theta_is_the_slope_normal_to_each_strip():
    # A fine mesh, so the chord slope of the normal displacement matches the mean end rotation.
    m = cufsm.stress(cufsm.lipped_c(200, 76, 15, 1.9, ri=3, mesh=60), P=1.0)
    ms = cufsm.strip(m, [150.0], neigs=1).mode_shape(0)
    u, w, th = ms.u[0], ms.w[0], ms.theta[0]
    for e in m.elem:
        i, j = int(e[1]) - 1, int(e[2]) - 1
        dx, dz = m.x[j] - m.x[i], m.z[j] - m.z[i]
        b = np.hypot(dx, dz)
        c, s = dx / b, dz / b
        slope = ((-s * u[j] + c * w[j]) - (-s * u[i] + c * w[i])) / b
        assert slope == pytest.approx((th[i] + th[j]) / 2, abs=2e-4)


def test_at_sums_terms_with_the_shape_functions(tutorial_loaded):
    r = cufsm.strip(tutorial_loaded, [10.0], neigs=1)
    ms = r.mode_shape(0)
    mid = ms.at(5.0)
    np.testing.assert_allclose(mid.u, ms.u[0], atol=1e-15)
    np.testing.assert_allclose(mid.w, ms.w[0], atol=1e-15)
    np.testing.assert_allclose(mid.v, 0.0, atol=1e-12)  # cos(pi/2): v is zero at mid-length
    end = ms.at(0.0)
    np.testing.assert_allclose(end.v, ms.v[0], atol=1e-15)
    assert ms.at().y == pytest.approx(5.0, rel=0.02)
    cc = cufsm.strip(tutorial_loaded, [60.0], m_all=4, bc="C-C", neigs=1).mode_shape(0)
    d = cc.at()
    assert 0 < d.y < 60 and d.u.shape == (10,)
    np.testing.assert_allclose(cc.at(0.0).u, 0.0, atol=1e-12)  # clamped ends do not move


def test_mode_shape_errors(tutorial_loaded):
    r = cufsm.strip(tutorial_loaded, [10.0], neigs=2)
    with pytest.raises(IndexError):
        r.mode_shape(1)
    with pytest.raises(IndexError):
        r.mode_shape(0, 2)
    assert r.mode_shape(-1).length == 10.0


def test_classify_minima(tutorial_loaded, lengths):
    sig = cufsm.signature(tutorial_loaded, lengths)
    c = sig.classify_minima()
    assert c.shape == (2, 4)
    np.testing.assert_allclose(c.sum(axis=1), 100.0, rtol=1e-9)
    assert [cufsm.MODE_CLASSES[k] for k in np.argmax(c, axis=1)] == ["L", "D"]
    assert cufsm.strip(tutorial_loaded, [10.0]).classify_minima().shape == (0, 4)


def test_model_strip_method(tutorial_loaded):
    a = tutorial_loaded.strip([10.0, 20.0], neigs=2)
    b = cufsm.strip(tutorial_loaded, [10.0, 20.0], neigs=2)
    np.testing.assert_array_equal(a.load_factors, b.load_factors)


def test_version_matches_pyproject():
    text = (pathlib.Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert re.search(r'^version = "([^"]+)"', text, re.M).group(1) == cufsm.__version__


def _run(code):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_core_import_does_not_load_matplotlib():
    r = _run("import sys, cufsm_rs; print('matplotlib' in sys.modules, 'cufsm_rs.plot' in sys.modules)")
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["False", "False"]


def test_plot_without_matplotlib_says_how_to_install():
    code = (
        "import sys; sys.modules['matplotlib'] = None\n"
        "import cufsm_rs\n"
        "try:\n    import cufsm_rs.plot\nexcept ImportError as e:\n    print(e)\n"
    )
    r = _run(code)
    assert r.returncode == 0, r.stderr
    assert "pip install cufsm-rs-py[plot]" in r.stdout


def test_plot_attribute_is_lazy():
    r = _run("import cufsm_rs; print(cufsm_rs.plot.plot_section.__name__)")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "plot_section"
