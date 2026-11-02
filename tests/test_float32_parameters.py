"""Float32 parameters give the results of float64 parameters holding the same values."""
import numpy as np
import pandas as pd

from helpers import simulate_attempts
from pyBKT.generate import synthetic_data
from pyBKT.models import Model

PARAMETERS = {"learns": 0.15, "forgets": 0.02, "guesses": 0.2, "slips": 0.1}


def parameters(dtype):
    """The same values in both dtypes: float32 rounding first, so float64 holds them exactly."""
    values = {name: np.array([value], dtype=np.float32).astype(dtype) for name, value in PARAMETERS.items()}
    return {"prior": 0.3, **values}


def fitted(dtype, data):
    model = Model(seed=0, num_fits=1, parallel=False)
    model.coef_ = {"skill": parameters(dtype)}
    model.fit(data=data, forgets=True)
    return model


def test_fit_and_predict_match_float64():
    data = simulate_attempts(seed=0, students=20, attempts=6, prior=0.3, learn=0.2, guess=0.2, slip=0.1)
    single, double = fitted(np.float32, data), fitted(np.float64, data)

    # the pure Python build computes in float32 when given float32 arrays, so allow float32 rounding
    pd.testing.assert_frame_equal(single.params(), double.params(), rtol=1e-5)
    pd.testing.assert_frame_equal(single.predict(data=data), double.predict(data=data), rtol=1e-5)


def test_synthetic_data_reads_float32_parameters():
    # a student who starts unknown and learns after one item, never guessing or slipping,
    # answers the first item of each sequence wrong (1) and every later one right (2)
    rates = {"learns": 1.0, "forgets": 0.0, "guesses": 0.0, "slips": 0.0}
    model = {"prior": 0.0, **{name: np.array([value], dtype=np.float32) for name, value in rates.items()}}
    generated = synthetic_data.synthetic_data(model, np.array([6, 4]), np.ones(10, dtype=np.int64))

    np.testing.assert_array_equal(generated["data"][0], [1, 2, 2, 2, 2, 2, 1, 2, 2, 2])
