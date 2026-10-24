"""Data that uses only some of a fitted model's guess/slip classes (#47)."""
import numpy as np
import pandas as pd

from helpers import simulate_attempts
from pyBKT.models import Model


def fitted_multigs_model(attempts):
    model = Model(seed=0, num_fits=1, parallel=False)
    model.fit(data=attempts, multigs=True)
    return model


def test_predict_on_a_subset_of_classes_matches_predict_with_all_classes():
    """Adding a student who answers other classes doesn't change the first students' predictions."""
    attempts = simulate_attempts(
        seed=0, students=60, attempts=9, prior=0.3, learn=0.2, guess=0.2, slip=0.1, templates=("a", "b", "c")
    )
    model = fitted_multigs_model(attempts)

    only_c = attempts[attempts["template_id"] == "c"].copy()
    other_student = attempts[(attempts["user_id"] == "student0") & (attempts["template_id"] != "c")].copy()
    other_student["user_id"] = "other"
    with_other = pd.concat([only_c, other_student])

    alone = model.predict(data=only_c)
    together = model.predict(data=with_other).loc[only_c.index]

    for column in ["correct_predictions", "state_predictions"]:
        np.testing.assert_allclose(alone[column], together[column])


def test_evaluate_on_a_subset_of_classes():
    attempts = simulate_attempts(
        seed=1, students=60, attempts=9, prior=0.3, learn=0.2, guess=0.2, slip=0.1, templates=("a", "b", "c")
    )
    model = fitted_multigs_model(attempts)

    rmse = model.evaluate(data=attempts[attempts["template_id"] == "c"])

    assert 0 <= rmse <= 1
