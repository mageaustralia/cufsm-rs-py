"""The plot functions, on the Agg backend: they run, return the Axes and draw what they should."""

import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

import cufsm_rs as cufsm  # noqa: E402
from cufsm_rs.plot import plot_mode, plot_section, plot_signature  # noqa: E402


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _polys(ax):
    return [c for c in ax.collections if isinstance(c, PolyCollection)]


def test_plot_section(tutorial_loaded):
    ax = plot_section(tutorial_loaded, node_numbers=True)
    assert isinstance(ax, plt.Axes)
    polys = _polys(ax)
    section = polys[-1]
    assert len(section.get_paths()) == 9  # one strip per element, drawn to its thickness
    assert sum(len(p.get_paths()) for p in polys[:-1]) == 9  # all compression, one band each
    assert len(ax.texts) == 10
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["compression"]


def test_plot_section_bending_splits_at_zero(tutorial):
    m = cufsm.stress(tutorial, Mxx=100.0)
    ax = plot_section(m)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["compression", "tension"]
    comp, tens = _polys(ax)[:2]
    # the web's middle element is split where the stress crosses zero
    assert len(comp.get_paths()) + len(tens.get_paths()) == 10


def test_plot_section_without_stress_on_given_axes(tutorial):
    fig, ax = plt.subplots()
    out = plot_section(tutorial, ax=ax, stress=True)  # zero stress: nothing to overlay
    assert out is ax and len(_polys(ax)) == 1 and ax.get_legend() is None
    ax2 = plot_section(cufsm.stress(tutorial, P=1.0), stress=False)
    assert len(_polys(ax2)) == 1 and not ax2.texts


def test_plot_signature(tutorial_loaded, lengths):
    sig = cufsm.signature(tutorial_loaded, lengths)
    ax = plot_signature(sig)
    assert isinstance(ax, plt.Axes) and ax.get_xscale() == "log"
    assert len(ax.lines) == 1 + len(sig.minima)
    assert len(ax.texts) == 2 and ax.get_xlabel() == "half-wavelength"
    assert ax.get_ylim()[1] < np.nanmax(sig.curve)  # the short-length spike is clipped
    ax = plot_signature(sig, classify=True)
    labels = [t.get_text() for t in ax.texts]
    assert labels[0].startswith("L ") and labels[1].startswith("D ")


def test_plot_signature_general_bc_labels_length(tutorial_loaded):
    r = cufsm.strip(tutorial_loaded, [50.0, 100.0, 200.0], m_all=3, bc="C-C", neigs=1)
    ax = plot_signature(r, classify=True)
    assert ax.get_xlabel() == "length" and not ax.texts


@pytest.mark.parametrize("which", ["inch", "mm"])
def test_plot_mode_auto_scale_is_relative_to_the_section(which, tutorial_loaded, lipped):
    m = tutorial_loaded if which == "inch" else lipped
    L = 7.0 if which == "inch" else 150.0
    r = cufsm.strip(m, [L], neigs=1)
    ax = plot_mode(r, 0)
    assert isinstance(ax, plt.Axes)
    nel = len(m.elem)
    assert len(ax.lines) == 2 * nel + 1  # undeformed and deformed per element, then the nodes
    xy = ax.lines[-1].get_xydata()
    moved = np.hypot(xy[:, 0] - m.x, xy[:, 1] - m.z).max()
    size = max(np.ptp(m.x), np.ptp(m.z))
    assert moved == pytest.approx(0.1 * size, rel=1e-9)
    assert "load factor" in ax.get_title()


def test_plot_mode_explicit_scale_and_general_bc(lipped):
    r = cufsm.strip(lipped, [1500.0], m_all=5, bc="C-C", neigs=2)
    ax = plot_mode(r, 0, k_mode=1, scale=5.0)
    d = r.mode_shape(0, 1).at()
    xy = ax.lines[-1].get_xydata()
    np.testing.assert_allclose(xy[:, 0], lipped.x + 5.0 * d.u)
    np.testing.assert_allclose(xy[:, 1], lipped.z + 5.0 * d.w)
    assert ax.get_title().startswith("mode 2")
