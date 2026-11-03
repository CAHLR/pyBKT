"""The C++ E-step rejects inputs it would otherwise crash on or count wrongly."""
import numpy as np
import pytest

from helpers import IS_COMPILED

pytestmark = pytest.mark.skipif(not IS_COMPILED, reason="checks the C++ E-step's input checks")

MODEL = {"prior": 0.3, "learns": np.array([0.15]), "forgets": np.array([0.02]),
         "guesses": np.array([0.2]), "slips": np.array([0.1])}
# two students with four answers each; starts count from 1
DATA = {"data": np.array([[1, 2, 2, 1, 2, 1, 2, 2]], dtype=np.int32), "resources": np.ones(8, dtype=np.int64),
        "starts": np.array([1, 5]), "lengths": np.array([4, 4])}


def run(data=DATA, model=MODEL):
    from pyBKT.fit import E_step

    return E_step.run(data, model, 1, 0, {})


@pytest.mark.parametrize("key", ["prior", "learns", "forgets", "guesses", "slips"])
def test_a_missing_model_key_raises(key):
    with pytest.raises(KeyError, match=key):
        run(model={name: value for name, value in MODEL.items() if name != key})


@pytest.mark.parametrize("key", ["data", "resources", "starts", "lengths"])
def test_a_missing_data_key_raises(key):
    with pytest.raises(KeyError, match=key):
        run(data={name: value for name, value in DATA.items() if name != key})


@pytest.mark.parametrize("code, dtype", [(2**32 + 1, np.int64), (3, np.int32), (-1, np.int32)])
def test_an_answer_code_other_than_0_1_or_2_raises(code, dtype):
    answers = DATA["data"].astype(dtype)
    answers[0, 0] = code
    with pytest.raises(ValueError, match=f"answer code {code}"):
        run(data=dict(DATA, data=answers))


def test_64_bit_answer_codes_give_the_32_bit_result():
    wide = dict(DATA, data=DATA["data"].astype(np.int64))

    assert run(wide)["total_loglike"] == run()["total_loglike"]


@pytest.mark.parametrize("starts, lengths, message", [
    ([1, 4], [5, 5], "more than one sequence covers data column 4"),
    ([1, 5], [3, 4], "no sequence covers data column 4"),
])
def test_histories_must_cover_each_column_once(starts, lengths, message):
    with pytest.raises(ValueError, match=message):
        run(data=dict(DATA, starts=np.array(starts), lengths=np.array(lengths)))


def test_histories_may_come_in_any_order():
    reordered = dict(DATA, starts=np.array([5, 1]), lengths=np.array([4, 4]))

    assert run(reordered)["total_loglike"] == pytest.approx(run()["total_loglike"], rel=1e-15)
