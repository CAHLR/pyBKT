"""A seeded fit and its predictions stay the same from one change to the next.

The expected values in ``reference/model_outputs.json`` are kept for each build,
because the two builds stop EM at different tolerances. Regenerate them with
``make_reference.py`` only when a change is meant to alter results.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from helpers import IS_COMPILED, simulate_attempts
from pyBKT.models import Model

BUILD = "compiled" if IS_COMPILED else "python"
REFERENCE_FILE = Path(__file__).parent / "reference" / "model_outputs.json"

CASES = {
    "default": dict(),
    "forgets": dict(forgets=True),
}


def fit_and_predict(case):
    """Fit a seeded model for one case and return its parameters and predictions."""
    attempts = simulate_attempts(seed=6, students=30, attempts=5, prior=0.3, learn=0.2, guess=0.2, slip=0.1, forget=0.05)
    model = Model(seed=0, num_fits=2, parallel=False)
    model.fit(data=attempts.copy(), **CASES[case])
    # predict() sorts the frame it is given, so pass a copy and restore the input order.
    predictions = model.predict(data=attempts.copy()).loc[attempts.index]
    return {
        "params": model.params()["value"].tolist(),
        "correct_predictions": predictions["correct_predictions"].tolist(),
        "state_predictions": predictions["state_predictions"].tolist(),
    }


@pytest.mark.parametrize("case", CASES)
def test_fit_and_predictions_match_reference(case):
    """The fitted parameters and the predictions match the stored values for this build."""
    expected_values = json.loads(REFERENCE_FILE.read_text())[BUILD][case]
    outputs = fit_and_predict(case)
    for name, expected in expected_values.items():
        np.testing.assert_allclose(outputs[name], expected, rtol=1e-7, atol=1e-10, err_msg=name)
