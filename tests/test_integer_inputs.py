"""The E-step reads integer arrays the same way whatever their width, and never past their end."""
import numpy as np
import pytest

from helpers import IS_COMPILED, simulate_attempts
from pyBKT.fit import EM_fit, predict_onestep
from pyBKT.generate import random_model_uni
from pyBKT.models import Model, Roster
from pyBKT.util import crossvalidate, data_helper


def fitted_parameters(data):
    start = random_model_uni.random_model_uni(1, 1, rand=np.random.RandomState(0))
    fitted, _ = EM_fit.EM_fit(start, data, tol=-1, maxiter=5, parallel=False)
    return [fitted[name] for name in ("prior", "learns", "guesses", "slips")]


def test_64_bit_answers_fit_like_32_bit_answers():
    """EM gives the same fit when the answer array is int64 instead of int32."""
    attempts = simulate_attempts(seed=3, students=40, attempts=6, prior=0.3, learn=0.2, guess=0.2, slip=0.1)
    data = data_helper.convert_data(attempts, "skill")["skill"]
    wide = dict(data, data=data["data"].astype(np.int64))

    for expected, actual in zip(fitted_parameters(data), fitted_parameters(wide)):
        np.testing.assert_array_equal(actual, expected)


def test_roster_matches_predict_on_the_same_answers():
    """Roster, which builds int64 answers, gives the mastery Model.predict gives for the same answers."""
    attempts = simulate_attempts(seed=4, students=40, attempts=6, prior=0.3, learn=0.2, guess=0.2, slip=0.1)
    model = Model(seed=0, num_fits=1, parallel=False)
    model.fit(data=attempts)
    answers = [0, 1, 1, 0, 1, 1]

    roster = Roster(students=["new"], skills="skill", model=model)
    roster.update_state("skill", "new", np.array(answers))
    # predict gives mastery before each answer, so one extra row reads the mastery after the last one.
    new = simulate_attempts(seed=5, students=1, attempts=7, prior=0.3, learn=0.2, guess=0.2, slip=0.1)
    new["correct"] = answers + [1]
    expected = model.predict(data=new)["state_predictions"].iloc[-1]

    assert roster.get_mastery_prob("skill", "new") == pytest.approx(expected, rel=1e-12)


def multilearn_inputs():
    """Return a fitted multilearn model and one student's data with three answers."""
    attempts = simulate_attempts(
        seed=6, students=40, attempts=6, prior=0.3, learn={"a": 0.1, "b": 0.3}, guess=0.2, slip=0.1,
        templates=("a", "b"),
    )
    model = Model(seed=0, num_fits=1, parallel=False)
    model.fit(data=attempts, multilearn="template_id")
    one_student = attempts[attempts.user_id == "student0"].head(3)
    data = data_helper.convert_data(one_student, "skill", model_type=[True, False, False, False],
                                    resource_refs=model.fit_model)["skill"]
    return model.fit_model["skill"], data


def test_a_short_resources_array_raises():
    """The E-step raises when resources is shorter than a sequence, as a Roster multilearn update can make it."""
    model, data = multilearn_inputs()
    short = dict(data, resources=data["resources"][:1])

    with pytest.raises(ValueError if IS_COMPILED else IndexError, match="resources" if IS_COMPILED else None):
        predict_onestep.run(model, short, parallel=False)


@pytest.mark.skipif(not IS_COMPILED, reason="checks the C++ E-step's input conversion")
def test_non_integer_starts_raise():
    """The C++ E-step rejects a starts array that doesn't hold integers instead of misreading it."""
    model, data = multilearn_inputs()
    floats = dict(data, starts=data["starts"].astype(float))

    with pytest.raises(ValueError, match="starts must hold integers"):
        predict_onestep.run(model, floats, parallel=False)


BAD_INPUTS = {
    "a start before the data": lambda model, data: (model, dict(data, starts=np.array([0]))),
    "an empty sequence": lambda model, data: (model, dict(data, lengths=np.array([0]))),
    "a sequence past the data": lambda model, data: (model, dict(data, lengths=np.array([10**6]))),
    "more lengths than starts": lambda model, data: (model, dict(data, lengths=np.array([3, 3]))),
    "a resource with no learn rate": lambda model, data: (model, dict(data, resources=np.full(3, 99))),
    "one-dimensional data": lambda model, data: (model, dict(data, data=data["data"][0])),
    "too few guesses": lambda model, data: (dict(model, guesses=np.array([])), data),
}


@pytest.mark.skipif(not IS_COMPILED, reason="checks the C++ E-step's input checks")
@pytest.mark.parametrize("case", BAD_INPUTS)
def test_inputs_the_c_extension_cannot_read_raise(case):
    """The C++ E-step raises ValueError for inputs it would otherwise read out of bounds."""
    model, data = BAD_INPUTS[case](*multilearn_inputs())

    with pytest.raises(ValueError):
        predict_onestep.run(model, data, parallel=False)


def test_crossvalidate_with_a_folds_column_matches_32_bit_answers(monkeypatch):
    """crossvalidate with a folds column, which builds int64 answers, scores the same as with int32 answers."""
    attempts = simulate_attempts(seed=7, students=60, attempts=8, prior=0.3, learn=0.2, guess=0.2, slip=0.1)
    attempts["fold"] = attempts["user_id"].str.removeprefix("student").astype(int) % 3

    def auc():
        model = Model(seed=0, num_fits=1, parallel=False)
        return model.crossvalidate(data=attempts.copy(), folds="fold", metric="auc")["auc"].iloc[0]

    wide = auc()
    narrow_data = crossvalidate.fix_data_specified
    monkeypatch.setattr(crossvalidate, "fix_data_specified",
                        lambda *args: [dict(fold, data=fold["data"].astype(np.int32)) for fold in narrow_data(*args)])

    assert wide == auc()
