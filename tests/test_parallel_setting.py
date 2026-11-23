"""predict and evaluate pass the model's parallel setting to the prediction kernel."""
import pytest

from helpers import simulate_attempts
from pyBKT.fit import predict_onestep
from pyBKT.models import Model


@pytest.mark.parametrize("parallel", [False, True])
def test_predict_and_evaluate_pass_parallel(parallel, monkeypatch):
    data = simulate_attempts(seed=0, students=20, attempts=6, prior=0.3, learn=0.2, guess=0.2, slip=0.1)
    model = Model(seed=0, num_fits=1, parallel=parallel)
    model.fit(data=data)

    given = []
    run = predict_onestep.run

    def recording_run(*args, **kwargs):
        given.append(kwargs.get("parallel"))
        return run(*args, **kwargs)

    monkeypatch.setattr(predict_onestep, "run", recording_run)
    model.predict(data=data)
    model.evaluate(data=data)

    assert given == [parallel, parallel]
