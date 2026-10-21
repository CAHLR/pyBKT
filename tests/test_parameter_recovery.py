"""Fitting data simulated from known parameters recovers those parameters."""
import pytest

from helpers import IS_COMPILED, simulate_attempts
from pyBKT.models import Model

# Pure Python fits about 75 times slower; these tests run on the compiled build.
pytestmark = pytest.mark.skipif(not IS_COMPILED, reason="too slow on the pure-Python build")


def fitted_values(model, parameter):
    """Return a dict from class name to fitted value for one parameter."""
    values = model.params().loc["skill"].loc[parameter]["value"]
    return {name: float(value) for name, value in values.items()}


def test_recovers_the_standard_bkt_parameters():
    """prior, learn, guess and slip come back within 0.03 of the values that generated the data."""
    truth = dict(prior=0.3, learn=0.15, guess=0.2, slip=0.1)
    attempts = simulate_attempts(seed=3, students=1500, attempts=12, **truth)

    model = Model(seed=0, num_fits=3, parallel=False)
    model.fit(data=attempts)

    assert fitted_values(model, "prior")["default"] == pytest.approx(truth["prior"], abs=0.03)
    assert fitted_values(model, "learns")["default"] == pytest.approx(truth["learn"], abs=0.03)
    assert fitted_values(model, "guesses")["default"] == pytest.approx(truth["guess"], abs=0.03)
    assert fitted_values(model, "slips")["default"] == pytest.approx(truth["slip"], abs=0.03)


def test_recovers_a_guess_rate_for_each_template():
    """With multigs, each template's guess rate comes back within 0.05 of its true value."""
    guess = {"easy": 0.4, "hard": 0.1}
    attempts = simulate_attempts(
        seed=4, students=1500, attempts=12, prior=0.3, learn=0.15, guess=guess, slip=0.1,
        templates=("easy", "hard"),
    )

    model = Model(seed=0, num_fits=3, parallel=False)
    model.fit(data=attempts, multigs=True)

    fitted = fitted_values(model, "guesses")
    assert fitted["easy"] == pytest.approx(guess["easy"], abs=0.05)
    assert fitted["hard"] == pytest.approx(guess["hard"], abs=0.05)
