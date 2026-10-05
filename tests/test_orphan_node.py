"""A node that belongs to no element is refused with ValueError.

Up to 0.1.0 a cFSM run on such a model (strip restricted to spaces, classify) panicked inside
the engine and surfaced as a PanicException. cufsm-rs 0.4.2 refuses the model, and this package
names the node by its CUFSM node number (the engine's message gives its 0-based index).
"""

import dataclasses

import numpy as np
import pytest

import cufsm_rs as cufsm

from conftest import ELEM, NODE, PROP

# The tutorial C plus node 11, which no element uses.
ORPHAN = NODE + [[11, 7.5, 4.5, 1, 1, 1, 1, 0.0]]


@pytest.fixture
def orphan():
    return cufsm.Model(PROP, ORPHAN, ELEM)


@pytest.fixture
def result(tutorial_loaded):
    """A free strip result on the tutorial C, to classify against the orphan model."""
    return cufsm.strip(tutorial_loaded, [5.0, 50.0], neigs=1)


def assert_refused(excinfo):
    # Not PanicException (a BaseException, so pytest.raises(ValueError) would not catch it)
    # and not MechanismError: the model is invalid, not a mechanism.
    assert type(excinfo.value) is ValueError
    msg = str(excinfo.value)
    assert "invalid model" in msg
    assert "node 11 (node row 10) belongs to no element" in msg


def test_strip_restricted_to_spaces_refuses_the_orphan_node(orphan):
    with pytest.raises(ValueError) as e:
        cufsm.strip(orphan, [5.0, 50.0], neigs=1, spaces="D")
    assert_refused(e)


def test_classify_refuses_the_orphan_node(orphan, result):
    # strip refuses the orphan model, so take the valid model's result, point it at the orphan
    # model and pad each mode with the orphan node's four DOFs, so the modes have the length
    # this model needs and only the orphan node is wrong. This reached the panic in 0.1.0.
    modes = np.concatenate([result.modes, np.zeros(result.modes.shape[:2] + (4,))], axis=2)
    with pytest.raises(ValueError) as e:
        cufsm.classify(dataclasses.replace(result, model=orphan, modes=modes))
    assert_refused(e)


@pytest.mark.parametrize("call", [
    lambda m: cufsm.strip(m, [5.0, 50.0], neigs=1),
    lambda m: cufsm.signature(m, [5.0, 50.0]),
    lambda m: cufsm.section_properties(m),
])
def test_every_entry_point_refuses_the_orphan_node(orphan, call):
    with pytest.raises(ValueError) as e:
        call(orphan)
    assert_refused(e)
