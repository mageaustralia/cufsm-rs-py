import numpy as np
import pytest

import cufsm_rs as cufsm

XZ = [(5, 1), (5, 0), (2.5, 0), (0, 0), (0, 3), (0, 6), (0, 9), (2.5, 9), (5, 9), (5, 8)]
PROP = [[100, 29500, 29500, 0.3, 0.3, 11346.15]]
NODE = [[i + 1, x, z, 1, 1, 1, 1, 0.0] for i, (x, z) in enumerate(XZ)]
ELEM = [[i + 1, i + 1, i + 2, 0.1, 100] for i in range(9)]


@pytest.fixture
def tutorial():
    """The CUFSM tutorial C, 9 x 5 x 1 in, t = 0.1 in, no stress."""
    return cufsm.Model(PROP, NODE, ELEM)


@pytest.fixture
def tutorial_loaded(tutorial):
    """The tutorial C in compression, reference stress fy = 50 ksi."""
    return cufsm.stress(tutorial, P=cufsm.first_yield(tutorial, 50).Py)


@pytest.fixture
def lipped():
    """200 x 76 x 15 x 1.9 mm lipped C, 3 mm inside radius, in compression at 1 MPa."""
    return cufsm.stress(cufsm.lipped_c(200, 76, 15, 1.9, ri=3.0), P=1.0)


@pytest.fixture(scope="session")
def lengths():
    return np.logspace(0, 3, 60)
