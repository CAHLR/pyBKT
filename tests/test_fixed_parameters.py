"""Fixed parameters keep their values, and parameters left free (-1) are still fitted."""
import numpy as np
import pytest

from helpers import simulate_attempts
from pyBKT.models import Model


@pytest.mark.parametrize("parameter", ["guesses", "slips"])
def test_one_fixed_and_one_free_guess_slip_class(parameter):
    """With multigs, fixing one class's guess or slip still fits the other class's value."""
    attempts = simulate_attempts(
        seed=5, students=300, attempts=10, prior=0.3, learn=0.15,
        guess={"easy": 0.4, "hard": 0.1}, slip=0.1, templates=("easy", "hard"),
    )
    model = Model(seed=0, num_fits=1, parallel=False)

    model.fit(data=attempts, multigs="template_id", fixed={"skill": {parameter: np.array([0.3, -1.0])}})

    fitted = model.coef_["skill"][parameter]
    assert fitted[0] == pytest.approx(0.3)
    assert 0 < fitted[1] < 1 and fitted[1] != pytest.approx(0.3)
