"""Model.fit and Model.predict leave the caller's DataFrame unchanged."""
import warnings

import pandas as pd

from helpers import simulate_attempts
from pyBKT.models import Model


def shuffled_attempts():
    """Simulated attempts in a shuffled row order, with order_id as text, so sorting and conversion would show."""
    attempts = simulate_attempts(seed=9, students=40, attempts=6, prior=0.3, learn=0.2, guess=0.2, slip=0.1)
    attempts = attempts.sample(frac=1, random_state=1)
    return attempts.assign(order_id=attempts["order_id"].astype(str))


def test_fit_and_predict_leave_the_callers_frame_unchanged():
    """The caller's rows keep their order, columns and dtypes through fit and predict."""
    attempts = shuffled_attempts()
    original = attempts.copy()
    model = Model(seed=0, num_fits=1, parallel=False)

    model.fit(data=attempts)
    predictions = model.predict(data=attempts)

    pd.testing.assert_frame_equal(attempts, original)
    assert set(predictions.index) == set(attempts.index)
    assert {"correct_predictions", "state_predictions"} <= set(predictions.columns)


def test_convert_data_raises_no_warnings_of_its_own():
    """Fitting emits no warning from data_helper, such as the pandas 4 deprecations or SettingWithCopyWarning."""
    first, second = shuffled_attempts(), shuffled_attempts()
    second = second.assign(skill_name="other", user_id="b" + second["user_id"])
    attempts = pd.concat([first, second], ignore_index=True).assign(original=1)
    model = Model(seed=0, num_fits=1, parallel=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(data=attempts)
        model.predict(data=attempts)

    assert [str(w.message) for w in caught if w.filename.endswith("data_helper.py")] == []
