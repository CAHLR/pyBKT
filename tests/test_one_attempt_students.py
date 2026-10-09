"""A student with a single attempt still gives evidence about guess and slip."""
import numpy as np
import pandas as pd
import pytest

from helpers import FAILS_ON_NUMPY2, simulate_attempts
from pyBKT.fit import EM_fit
from pyBKT.generate import random_model_uni
from pyBKT.util import data_helper


def guess_after_one_em_step(attempts):
    data = data_helper.convert_data(attempts.copy(), "skill")["skill"]
    start = random_model_uni.random_model_uni(1, 1, rand=np.random.RandomState(0))
    fitted, _ = EM_fit.EM_fit(start, data, tol=-1, maxiter=1, parallel=False)
    return float(fitted["guesses"][0])


@FAILS_ON_NUMPY2
def test_one_correct_answer_raises_the_guess_estimate():
    """After one EM step, adding a student with one correct answer raises the guess estimate."""
    attempts = simulate_attempts(seed=2, students=50, attempts=6, prior=0.3, learn=0.2, guess=0.2, slip=0.1)
    one_attempt_student = pd.DataFrame(
        [{"user_id": "single", "skill_name": "skill", "order_id": 10_000, "template_id": "t", "correct": 1}]
    )
    with_student = pd.concat([attempts, one_attempt_student], ignore_index=True)

    assert guess_after_one_em_step(with_student) > guess_after_one_em_step(attempts)
