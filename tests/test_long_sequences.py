"""The C++ E-step fits a student with a very long answer sequence."""
import numpy as np
import pytest

from helpers import IS_COMPILED
from pyBKT.fit import EM_fit
from pyBKT.generate import random_model_uni


@pytest.mark.skipif(not IS_COMPILED, reason="checks the C++ E-step's working memory")
def test_a_student_with_a_million_answers_fits():
    """One EM iteration on a single 1,000,000-answer sequence finishes with valid parameters.

    A sequence this long needs more than the main-thread stack on Linux (8 MiB) or in
    Python on macOS (16 MiB), so the E-step's per-sequence arrays must not live on the stack.
    """
    answers = 1_000_000
    rng = np.random.default_rng(0)
    data = {
        "data": (rng.random((1, answers)) < 0.6).astype(np.int32) + 1,
        "starts": np.array([1], dtype=np.int64),
        "lengths": np.array([answers], dtype=np.int64),
        "resources": np.ones(answers, dtype=np.int64),
    }
    start = random_model_uni.random_model_uni(1, 1, rand=np.random.RandomState(0))

    fitted, _ = EM_fit.EM_fit(start, data, tol=-1, maxiter=1, parallel=False)

    for name in ("prior", "learns", "forgets", "guesses", "slips"):
        values = np.ravel(fitted[name])
        assert np.all(np.isfinite(values)) and np.all((0 <= values) & (values <= 1)), name
