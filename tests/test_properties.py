"""Properties every BKT fit should have, checked on many simulated datasets."""
import numpy as np
import pandas as pd
import pytest
from hypothesis import Phase, assume, given, settings
from hypothesis import strategies as st

from helpers import IS_COMPILED, NUMPY_MAJOR, simulate_attempts
from pyBKT.fit import EM_fit
from pyBKT.generate import random_model_uni
from pyBKT.models import Model
from pyBKT.util import data_helper, metrics

# The pure-Python fit fails on NumPy 2 (issue #65), and test_backend_parity.py
# already reports that, so skip these slower tests there.
pytestmark = pytest.mark.skipif(
    not IS_COMPILED and NUMPY_MAJOR >= 2, reason="pure-Python fit fails on NumPy 2, see issue #65"
)

# Each example fits a model. derandomize=True makes every run use the same 20
# examples, and skipping the shrink phase keeps a failing run short.
FITS = settings(
    max_examples=20,
    deadline=None,
    derandomize=True,
    phases=[Phase.explicit, Phase.reuse, Phase.generate],
)


def vary_lengths(attempts):
    """Keep a different number of attempts for each student, from 1 up to all of them."""
    student_number = attempts["user_id"].str.removeprefix("student").astype(int)
    attempt_number = attempts.groupby("user_id").cumcount()
    longest = attempt_number.max() + 1
    return attempts[attempt_number < 1 + student_number % longest].reset_index(drop=True)


datasets = st.builds(
    simulate_attempts,
    seed=st.integers(0, 10_000),
    students=st.integers(5, 30),
    attempts=st.integers(2, 8),
    prior=st.floats(0.1, 0.9),
    learn=st.floats(0.05, 0.4),
    guess=st.floats(0.05, 0.35),
    slip=st.floats(0.05, 0.25),
).map(vary_lengths)


def fitted_parameters(attempts, **fit_options):
    model = Model(seed=0, num_fits=1, parallel=False)
    model.fit(data=attempts.copy(), **fit_options)
    return model, model.params()["value"].to_numpy()


def run_em(attempts, iterations=10):
    """Run EM with no early stop from a fixed starting model."""
    data = data_helper.convert_data(attempts.copy(), "skill")["skill"]
    start = random_model_uni.random_model_uni(1, 1, rand=np.random.RandomState(0))
    return EM_fit.EM_fit(start, data, tol=-1, maxiter=iterations, parallel=False)


@FITS
@given(datasets)
def test_parameters_and_predictions_are_probabilities(attempts):
    """Every fitted parameter and every prediction lies between 0 and 1."""
    model, parameters = fitted_parameters(attempts, forgets=True)
    predictions = model.predict(data=attempts.copy())

    assert np.all((parameters >= 0) & (parameters <= 1))
    for column in ("correct_predictions", "state_predictions"):
        assert predictions[column].between(0, 1).all()


@FITS
@given(datasets)
def test_em_never_lowers_the_log_likelihood(attempts):
    """Each EM iteration leaves the log-likelihood the same or higher."""
    _, log_likelihoods = run_em(attempts)
    steps = np.diff(log_likelihoods.ravel())
    assert np.all(steps >= -1e-9 * np.abs(log_likelihoods[:-1].ravel()))


@FITS
@given(datasets, st.integers(0, 2**32 - 1))
def test_row_order_does_not_change_the_fit(attempts, shuffle_seed):
    """Shuffling the rows gives the same fit, because pyBKT orders attempts by order_id."""
    shuffled = attempts.sample(frac=1, random_state=shuffle_seed)
    _, original = fitted_parameters(attempts)
    _, reordered = fitted_parameters(shuffled)
    np.testing.assert_array_equal(original, reordered)


@FITS
@given(datasets)
def test_student_names_do_not_change_the_fit(attempts):
    """Renaming every student gives the same fit."""
    # Numbering the students in reverse also reverses the order pyBKT sorts them in.
    number = attempts["user_id"].str.removeprefix("student").astype(int)
    renamed = attempts.assign(user_id="student" + (999 - number).astype(str))
    _, original = fitted_parameters(attempts)
    _, after_rename = fitted_parameters(renamed)
    np.testing.assert_allclose(original, after_rename, rtol=1e-9)


@FITS
@given(datasets)
def test_duplicating_every_student_does_not_change_the_fit(attempts):
    """Copying every student leaves each EM step unchanged.

    Doubling the data doubles every expected count, so each M-step computes the
    same ratios. This runs fixed EM steps, because the stopping rule in
    Model.fit compares log-likelihoods, which also double.
    """
    copies = attempts.assign(
        user_id="copy-" + attempts["user_id"], order_id=attempts["order_id"] + len(attempts)
    )
    doubled = pd.concat([attempts, copies], ignore_index=True)
    once, _ = run_em(attempts)
    twice, _ = run_em(doubled)
    for parameter in ("prior", "learns", "guesses", "slips"):
        np.testing.assert_allclose(once[parameter], twice[parameter], rtol=1e-8, err_msg=parameter)


@FITS
@given(datasets, st.floats(0.05, 0.5))
def test_a_fixed_learn_rate_stays_fixed(attempts, learn):
    """A learn rate passed through fixed= comes back unchanged."""
    model, _ = fitted_parameters(attempts, fixed={"skill": {"learns": np.array([learn])}})
    assert model.coef_["skill"]["learns"][0] == learn


@FITS
@given(datasets)
def test_a_correct_answer_never_lowers_predicted_mastery(attempts):
    """With no forgetting and guess + slip < 1, a correct answer never lowers the next mastery estimate."""
    model, _ = fitted_parameters(attempts)
    coefficients = model.coef_["skill"]
    assume(coefficients["guesses"][0] + coefficients["slips"][0] < 1)

    predictions = model.predict(data=attempts.copy()).sort_values("order_id")
    for _, student in predictions.groupby("user_id"):
        mastery = student["state_predictions"].to_numpy()
        correct = student["correct"].to_numpy()
        after_correct = mastery[1:][correct[:-1] == 1]
        before = mastery[:-1][correct[:-1] == 1]
        assert np.all(after_correct >= before - 1e-12)


# Predictions are whole percentages, so squaring or reversing them cannot
# round two different predictions into a tie.
percent = st.integers(1, 99)


@settings(derandomize=True)
@given(st.lists(st.tuples(st.integers(0, 1), percent), max_size=48), percent, percent)
def test_auc_depends_only_on_the_order_of_predictions(answers, incorrect_score, correct_score):
    """Squaring the predictions keeps their order, so AUC does not change; reversing them gives 1 - AUC."""
    answers = [(0, incorrect_score), (1, correct_score)] + answers
    labels = np.array([label for label, _ in answers])
    predictions = np.array([score / 100 for _, score in answers])
    auc = metrics.auc(labels, predictions)

    assert np.isclose(metrics.auc(labels, predictions**2), auc)
    assert np.isclose(metrics.auc(labels, 1 - predictions), 1 - auc)
