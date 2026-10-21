"""The evaluation metrics give the textbook answer on small, hand-checked inputs."""
import math

import numpy as np
import pytest

from helpers import simulate_attempts
from pyBKT.models import Model
from pyBKT.util import metrics

LABELS = np.array([0, 0, 1, 1])


def test_auc_is_one_when_predictions_rank_every_correct_answer_first():
    assert metrics.auc(LABELS, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


def test_auc_is_zero_when_predictions_rank_every_correct_answer_last():
    assert metrics.auc(LABELS, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)


def test_auc_is_one_half_when_every_prediction_is_tied():
    assert metrics.auc(LABELS, np.full(4, 0.5)) == pytest.approx(0.5)


def test_auc_is_undefined_when_every_answer_has_the_same_label():
    assert math.isnan(metrics.auc(np.array([1, 1, 1]), np.array([0.2, 0.5, 0.9])))


def test_rmse():
    assert metrics.rmse(LABELS, np.array([0.0, 0.4, 0.5, 1.0])) == pytest.approx(math.sqrt(0.1025))


def test_accuracy_counts_a_prediction_of_one_half_as_correct():
    assert metrics.accuracy(LABELS, np.array([0.0, 0.4, 0.5, 1.0])) == pytest.approx(1.0)


def test_evaluate_rejects_an_unknown_metric_name():
    attempts = simulate_attempts(seed=5, students=20, attempts=4, prior=0.3, learn=0.2, guess=0.2, slip=0.1)
    model = Model(seed=0, num_fits=1, parallel=False)
    model.fit(data=attempts)
    with pytest.raises(ValueError, match="metric must be one of"):
        model.evaluate(data=attempts, metric="not_a_metric")
