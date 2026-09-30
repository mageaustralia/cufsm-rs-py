"""__repr__ and _repr_html_ for Jupyter."""

import cufsm_rs as cufsm


def test_model_repr(tutorial):
    assert repr(tutorial) == "Model(10 nodes, 9 elements, 1 materials, 0 constraints, 0 springs)"
    h = tutorial._repr_html_()
    assert h.count("<table>") == 3 and "10 nodes, 9 elements" in h
    assert "..." not in h  # small enough to show whole
    assert h.count("<tr>") == 1 + 10 + 1 + 9 + 1 + 1


def test_big_model_html_is_truncated():
    m = cufsm.lipped_c(200, 76, 15, 1.9, ri=3, mesh=40)
    h = m._repr_html_()
    assert "..." in h and h.count("<tr>") < 40
    assert f"{len(m.node)} nodes" in h


def test_properties_and_actions_html(tutorial):
    p = cufsm.section_properties(tutorial)
    h = p._repr_html_()
    for k in ["Ixx", "Cw", "J", "St Venant torsion constant"]:
        assert k in h
    assert "wn" not in repr(p)  # the per-node array stays out of the repr
    y = cufsm.first_yield(tutorial, 50)
    hy = y._repr_html_()
    assert "First-yield" in hy and "Py" in hy and "105" in hy
    fit = cufsm.stress_to_action(cufsm.stress(tutorial, P=1.0))
    assert "err" in fit._repr_html_()


def test_result_repr_and_html(tutorial_loaded, lengths):
    sig = cufsm.signature(tutorial_loaded, lengths)
    assert repr(sig) == "StripResult(signature, bc=S-S, 60 lengths 1 to 1000, neigs=1, 2 minima)"
    h = sig._repr_html_()
    assert "Signature curve" in h and h.count("<tr>") == 3  # header and two minima
    st = cufsm.strip(tutorial_loaded, lengths, m_all=2, bc="c-c", neigs=2)
    assert repr(st) == "StripResult(strip, bc=C-C, 60 lengths 1 to 1000, neigs=2)"
    hs = st._repr_html_()
    assert "Strip analysis" in hs and "..." in hs and "terms per length: 2" in hs
