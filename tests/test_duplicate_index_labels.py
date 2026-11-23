"""predict works on frames whose index repeats labels, as pd.concat gives without ignore_index=True."""
import numpy as np
import pandas as pd

from helpers import simulate_attempts
from pyBKT.models import Model


def two_skills():
    first = simulate_attempts(seed=0, students=20, attempts=6, prior=0.3, learn=0.2, guess=0.2, slip=0.1)
    second = simulate_attempts(seed=1, students=20, attempts=6, prior=0.4, learn=0.1, guess=0.3, slip=0.1)
    second["skill_name"] = "other"
    second["user_id"] = second["user_id"] + "_other"
    return first, second


def test_predict_with_repeated_labels_matches_unique_labels():
    first, second = two_skills()
    repeated = pd.concat([first, second])
    unique = repeated.reset_index(drop=True)
    assert not repeated.index.is_unique

    model = Model(seed=0, num_fits=1, parallel=False)
    model.fit(data=unique)
    with_repeats = model.predict(data=repeated)
    without_repeats = model.predict(data=unique)

    for column in ["correct_predictions", "state_predictions"]:
        np.testing.assert_array_equal(with_repeats[column].to_numpy(), without_repeats[column].to_numpy())
    assert sorted(with_repeats.index) == sorted(repeated.index)


def test_each_row_gets_its_own_prediction():
    first, second = two_skills()
    repeated = pd.concat([first, second])
    model = Model(seed=0, num_fits=1, parallel=False)
    model.fit(data=repeated.reset_index(drop=True))

    key = ["skill_name", "user_id", "order_id"]
    alone = pd.concat([model.predict(data=frame) for frame in (first, second)]).set_index(key)
    together = model.predict(data=repeated).set_index(key)

    pd.testing.assert_series_equal(
        together["correct_predictions"].sort_index(), alone["correct_predictions"].sort_index(), check_exact=True
    )
