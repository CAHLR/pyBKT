"""With multigs, a missing answer is predicted with its own guess/slip class."""
import numpy as np
import pytest

from helpers import simulate_attempts
from pyBKT.models import Model
from pyBKT.util import crossvalidate
from pyBKT.util.data_helper import convert_data

DEFAULTS = {"multigs": "template_id", "folds": "folds"}


def attempts():
    return simulate_attempts(
        seed=0, students=60, attempts=8, prior=0.3, learn=0.2,
        guess={"a": 0.1, "b": 0.5, "c": 0.3}, slip=0.1, templates=("a", "b", "c"),
    )


@pytest.fixture(scope="module")
def fitted():
    data = attempts()
    model = Model(seed=0, num_fits=1, parallel=False)
    model.fit(data=data, multigs=True)
    return model, data


@pytest.mark.parametrize("row", [0, 1, 2, 7])
def test_missing_answer_keeps_its_prediction(fitted, row):
    model, data = fitted
    missing = data.copy()
    missing.loc[row, "correct"] = -1

    answered = model.predict(data=data).loc[row, "correct_predictions"]
    unanswered = model.predict(data=missing).loc[row, "correct_predictions"]

    assert unanswered == pytest.approx(answered, rel=0, abs=1e-12)


def test_fold_builders_keep_each_columns_class():
    """A missing answer changes an answer code, never a class, so folds built with and without one agree."""
    complete = attempts().assign(folds=lambda frame: frame["user_id"].str[-1:])
    missing = complete.copy()
    missing.loc[::5, "correct"] = -1
    full, gaps = (
        convert_data(frame, ".*", defaults=DEFAULTS, model_type=[False, False, False, True], folds=True)["skill"]
        for frame in (complete, missing)
    )
    label = full["folds"][0]
    count = np.count_nonzero(full["folds"] == label)
    every_other = np.arange(0, len(full["starts"]), 2)

    full_train, full_test = crossvalidate.fix_data_specified(full, label, count)
    gaps_train, gaps_test = crossvalidate.fix_data_specified(gaps, label, count)
    pairs = [
        (full_train, gaps_train),
        (full_test, gaps_test),
        (crossvalidate.fix_data(full, every_other), crossvalidate.fix_data(gaps, every_other)),
    ]
    for full_fold, gaps_fold in pairs:
        np.testing.assert_array_equal(full_fold["gs_classes"], (full_fold["data"] != 0).argmax(axis=0))
        np.testing.assert_array_equal(gaps_fold["gs_classes"], full_fold["gs_classes"])
