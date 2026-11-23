"""evaluate and crossvalidate score only real answers: no missing answers, no multiprior placeholder rows."""
import numpy as np
import pytest

from helpers import simulate_attempts
from pyBKT.models import Model


def attempts():
    return simulate_attempts(
        seed=0, students=40, attempts=6, prior=0.3, learn=0.2, guess=0.2, slip=0.1, templates=("a", "b")
    )


def mean_absolute_error(true, predicted):
    return float(np.mean(np.abs(true - predicted)))


@pytest.mark.parametrize("options", [{}, {"multiprior": "template_id"}, {"multigs": True}])
def test_custom_metric_gets_one_label_per_answer(options):
    data = attempts()
    data.loc[1, "correct"] = -1
    model = Model(seed=0, num_fits=1, parallel=False)
    model.fit(data=data, **options)

    given = []

    def recording_metric(true, predicted):
        given.append(np.asarray(true))
        return mean_absolute_error(true, predicted)

    reported = model.evaluate(data=data, metric=recording_metric)
    predicted = model.predict(data=data)
    answered = predicted[predicted["correct"] != -1]

    assert set(given[0]) <= {0, 1}
    assert len(given[0]) == len(answered)
    assert reported == pytest.approx(mean_absolute_error(answered["correct"], answered["correct_predictions"]))


def test_crossvalidate_gives_custom_metrics_only_answers():
    model = Model(seed=0, num_fits=1, parallel=False)
    labels = []

    def recording_metric(true, predicted):
        labels.extend(np.asarray(true).tolist())
        return mean_absolute_error(true, predicted)

    data = attempts()
    data.loc[::7, "correct"] = -1
    model.crossvalidate(data=data, multiprior="template_id", folds=2, metric=recording_metric)

    assert labels
    assert set(labels) <= {0, 1}


@pytest.mark.parametrize("metric", ["rmse", "accuracy", "auc"])
def test_built_in_metrics_are_nan_when_no_answer_is_present(metric):
    data = attempts()
    model = Model(seed=0, num_fits=1, parallel=False)
    model.fit(data=data)

    assert np.isnan(model.evaluate(data=data.assign(correct=-1), metric=metric))
