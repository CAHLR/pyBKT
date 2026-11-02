"""A fit or partial_fit that raises leaves the model as it was before the call."""
import copy

import pandas as pd
import pytest

from helpers import simulate_attempts
from pyBKT.models import Model


def two_skills():
    first = simulate_attempts(seed=0, students=20, attempts=6, prior=0.3, learn=0.2, guess=0.2, slip=0.1)
    second = simulate_attempts(seed=1, students=20, attempts=6, prior=0.4, learn=0.1, guess=0.3, slip=0.1)
    second["skill_name"] = "other"
    return pd.concat([first, second], ignore_index=True)


def snapshot(model, data):
    settings = {name: getattr(model, name) for name in ("forgets", "skills", "manual_param_init", "num_fits")}
    return settings, model.params(), model.predict(data=data)


def fail_on_second_skill(monkeypatch):
    calls = []
    fit = Model._fit

    def failing_fit(self, *args, **kwargs):
        calls.append(args[1])
        if len(calls) == 2:
            raise RuntimeError("second skill failed")
        return fit(self, *args, **kwargs)

    monkeypatch.setattr(Model, "_fit", failing_fit)


@pytest.mark.parametrize("method", ["fit", "partial_fit"])
@pytest.mark.parametrize("failure, error", [("answer code", ValueError), ("missing column", KeyError), ("second skill", RuntimeError)])
def test_failed_call_leaves_the_model_unchanged(method, failure, error, monkeypatch):
    data = two_skills()
    model = Model(seed=0, num_fits=1, parallel=False)
    model.fit(data=data)
    settings, params, predictions = snapshot(model, data)

    bad = data
    if failure == "answer code":
        bad = data.assign(correct=7)
    elif failure == "missing column":
        bad = data.drop(columns="correct")
    else:
        fail_on_second_skill(monkeypatch)

    never_failed = copy.deepcopy(model)
    with pytest.raises(error):
        getattr(model, method)(data=bad, forgets=True, num_fits=2)
    monkeypatch.undo()

    after_settings, after_params, after_predictions = snapshot(model, data)
    assert after_settings == settings
    pd.testing.assert_frame_equal(after_params, params, check_exact=True)
    pd.testing.assert_frame_equal(after_predictions, predictions, check_exact=True)

    # the random state is restored too, so the next call fits as if the failed one never ran.
    # num_fits and parallel are passed again because a second call drops them (#120).
    getattr(model, method)(data=data, num_fits=1, parallel=False)
    getattr(never_failed, method)(data=data, num_fits=1, parallel=False)
    pd.testing.assert_frame_equal(model.params(), never_failed.params(), check_exact=True)
