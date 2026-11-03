"""synthetic_data rejects resource IDs outside 1 to the number of learn rates."""
import numpy as np
import pytest

from helpers import IS_COMPILED
from pyBKT.generate import synthetic_data

MODEL = {"prior": 0.3, "learns": np.array([0.15, 0.25]), "forgets": np.array([0.02, 0.05]),
         "guesses": np.array([0.2]), "slips": np.array([0.1])}


@pytest.mark.parametrize("resources", [np.full(8, 3), np.zeros(8, dtype=int), np.ones(5, dtype=int)],
                         ids=["too large", "zero", "too few"])
def test_bad_resources_raise(resources):
    with pytest.raises(ValueError, match="resources"):
        synthetic_data.synthetic_data(dict(MODEL), np.array([8]), resources=resources)


def test_valid_resources_generate_data():
    generated = synthetic_data.synthetic_data(dict(MODEL), np.array([5, 3]), resources=np.array([1, 2] * 4))

    assert generated["data"].shape == (1, 8)
    assert set(np.unique(generated["data"])) <= {1, 2}


@pytest.mark.skipif(not IS_COMPILED, reason="the C++ helper exists only in the compiled build")
def test_helper_rejects_bad_resources_when_called_directly():
    from pyBKT.generate import synthetic_data_helper

    with pytest.raises(ValueError, match="resources"):
        synthetic_data_helper.create_synthetic_data(
            dict(MODEL), np.array([1], dtype=np.int64), np.array([8], dtype=np.int64), np.full(8, 3, dtype=np.int64)
        )
