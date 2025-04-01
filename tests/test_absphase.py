import pytest
import os

import pint.models
import pint.toa
from pinttestdata import datadir


@pytest.fixture(scope="module")
def model_and_toas():
    parfile = os.path.join(datadir, "NGC6440E.par")
    timfile = os.path.join(datadir, "zerophase.tim")
    return pint.models.get_model_and_toas(parfile, timfile)


def test_phase_zero(model_and_toas):
    # Check that model phase is 0.0 for a TOA at exactly the TZRMJD
    model, toas = model_and_toas
    ph = model.phase(toas, abs_phase=True)
    # Check that integer and fractional phase values are very close to 0.0
    assert ph.int.value == pytest.approx(0.0)
    assert ph.frac.value == pytest.approx(0.0)


def test_tzr_attr(model_and_toas):
    model, toas = model_and_toas
    assert not toas.tzr
    assert model.components["AbsPhase"].get_TZR_toa(toas).tzr
