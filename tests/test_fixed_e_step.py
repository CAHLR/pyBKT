"""The E-step uses fixed parameter values from its first iteration, not the starting model's."""
import numpy as np
import pytest

from helpers import IS_COMPILED, simulate_attempts
from pyBKT.fit import EM_fit
from pyBKT.generate import random_model_uni
from pyBKT.models import Model
from pyBKT.util import data_helper


def one_em_step(start_learn, fixed):
    attempts = simulate_attempts(seed=3, students=200, attempts=8, prior=0.3, learn=0.15, guess=0.2, slip=0.1)
    data = data_helper.convert_data(attempts, "skill")["skill"]
    start = random_model_uni.random_model_uni(1, 1, rand=np.random.RandomState(0))
    start["learns"] = np.array([start_learn])
    start["As"][0, 1, 0], start["As"][0, 0, 0] = start_learn, 1 - start_learn
    fitted, log_likelihoods = EM_fit.EM_fit(start, data, tol=-1, maxiter=1, parallel=False, fixed=fixed)
    return [float(np.ravel(fitted[name])[0]) for name in ("prior", "guesses", "slips")], float(np.ravel(log_likelihoods)[0])


def test_a_fixed_learn_rate_is_used_in_the_first_e_step():
    """Starting from learn 0.6 with learn fixed at 0.15 gives the same first step as starting from 0.15."""
    from_other_start = one_em_step(0.6, {"learns": np.array([0.15])})
    from_fixed_value = one_em_step(0.15, {"learns": np.array([0.15])})

    assert from_other_start == from_fixed_value


@pytest.mark.skipif(not IS_COMPILED, reason="checks the C++ E-step's input checks")
def test_a_fixed_array_shorter_than_the_model_raises():
    """A fixed guess array with fewer entries than guess/slip classes raises ValueError."""
    with pytest.raises(ValueError, match="fixed guesses has 0 entries"):
        one_em_step(0.15, {"guesses": np.array([])})


def test_one_fixed_guess_applies_to_every_guess_slip_class():
    """Under multigs, a single fixed guess is used for every class, as the M-step broadcasts it."""
    attempts = simulate_attempts(
        seed=4, students=200, attempts=8, prior=0.3, learn=0.15,
        guess={"easy": 0.4, "hard": 0.1}, slip=0.1, templates=("easy", "hard"),
    )
    model = Model(seed=1, num_fits=1, parallel=False)

    model.fit(data=attempts, multigs="template_id", fixed={"skill": {"guesses": np.array([0.25])}})

    np.testing.assert_array_equal(model.coef_["skill"]["guesses"], [0.25, 0.25])
