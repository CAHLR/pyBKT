"""The compiled and pure-Python E-steps must fit the same parameters.

Each case runs 20 EM iterations, with no early stop, from the same starting
model. The expected values in ``reference/em_parity.json`` come from the
pure-Python build; regenerate them with ``make_reference.py``.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from helpers import simulate_attempts
from pyBKT.fit import EM_fit
from pyBKT.generate import random_model_uni
from pyBKT.util import data_helper

REFERENCE_FILE = Path(__file__).parent / "reference" / "em_parity.json"

#              multilearn, multiprior, multipair, multigs
DEFAULT_MODEL = [False, False, False, False]
MULTILEARN = [True, False, False, False]
MULTIGS = [False, False, False, True]

# Under EM, a forget rate that starts at 0 stays at 0, so only the "forgets" case starts above 0.
CASES = {
    "default": dict(
        simulation=dict(prior=0.3, learn=0.2, guess=0.2, slip=0.1),
        model_type=DEFAULT_MODEL,
        start_forget=0.0,
    ),
    "forgets": dict(
        simulation=dict(prior=0.3, learn=0.2, guess=0.2, slip=0.1, forget=0.05),
        model_type=DEFAULT_MODEL,
        start_forget=0.1,
    ),
    "multilearn": dict(
        simulation=dict(prior=0.3, learn={"a": 0.1, "b": 0.35}, guess=0.2, slip=0.1, templates=("a", "b")),
        model_type=MULTILEARN,
        start_forget=0.0,
    ),
    "multigs": dict(
        simulation=dict(prior=0.3, learn=0.2, guess={"a": 0.1, "b": 0.4}, slip=0.1, templates=("a", "b")),
        model_type=MULTIGS,
        start_forget=0.0,
    ),
}

def fit_em(case):
    """Run 20 EM iterations for one case and return the fitted parameters."""
    attempts = simulate_attempts(seed=1, students=200, attempts=8, **CASES[case]["simulation"])
    data = data_helper.convert_data(attempts, "skill", model_type=CASES[case]["model_type"])["skill"]
    start = random_model_uni.random_model_uni(
        len(data["resource_names"]), len(data["gs_names"]), rand=np.random.RandomState(0)
    )
    start["forgets"] = np.full(len(data["resource_names"]), CASES[case]["start_forget"])
    fitted, _ = EM_fit.EM_fit(start, data, tol=-1, maxiter=20, parallel=False)
    return {
        "prior": [float(fitted["prior"])],
        "learns": np.ravel(fitted["learns"]).tolist(),
        "forgets": np.ravel(fitted["forgets"]).tolist(),
        "guesses": np.ravel(fitted["guesses"]).tolist(),
        "slips": np.ravel(fitted["slips"]).tolist(),
    }


@pytest.mark.parametrize("case", CASES)
def test_em_matches_reference(case):
    """Both builds fit the same parameters from the same data and starting model."""
    expected_values = json.loads(REFERENCE_FILE.read_text())[case]
    fitted = fit_em(case)
    for parameter, expected in expected_values.items():
        np.testing.assert_allclose(fitted[parameter], expected, rtol=1e-9, atol=1e-12, err_msg=parameter)
