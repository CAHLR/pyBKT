"""Roster updates give the mastery of the forward pass behind Model.predict and never change the model.

Runs against the compiled build, or the pure-Python build with ``PYTHONPATH=source-py``.
"""
import copy
import sys

import numpy as np
import pytest

from pyBKT.fit import predict_onestep
from pyBKT.models import Model, Roster

PARAMS = {
    "prior": 0.3,
    "learns": np.array([0.1, 0.25]),
    "forgets": np.array([0.02, 0.05]),
    "guesses": np.array([0.2, 0.3, 0.1]),
    "slips": np.array([0.1, 0.05, 0.2]),
}
GS_NAMES = {"a": 0, "b": 1, "c": 2}
LEARN_NAMES = {"x": 1, "y": 2}


def model_with_params(gs_classes=3, learn_classes=2, multilearn=None):
    """A Model holding PARAMS as if fitted, keeping the first gs_classes guess/slip and learn_classes learn classes.
    It counts as fitted with multilearn when it has more than one learn class, unless multilearn says otherwise."""
    params = copy.deepcopy(PARAMS)
    params["guesses"], params["slips"] = params["guesses"][:gs_classes], params["slips"][:gs_classes]
    params["learns"], params["forgets"] = params["learns"][:learn_classes], params["forgets"][:learn_classes]
    model = Model()
    model.fit_model = {"skill": dict(params, pi_0=np.array([[1 - params["prior"]], [params["prior"]]]),
                                     gs_names=dict(list(GS_NAMES.items())[:gs_classes]),
                                     resource_names=dict(list(LEARN_NAMES.items())[:learn_classes]))}
    is_multilearn = learn_classes > 1 if multilearn is None else multilearn
    model.model_type = [is_multilearn, False, False, gs_classes > 1]
    return model


def forward_pass_mastery(model, answers):
    """P(known) after the answers, from predict_onestep with int32 data as Model.predict passes it.
    Each answer is (correct, guess/slip class, learn class); correct -1 means no observation."""
    params = model.fit_model["skill"]
    data = np.zeros((len(params["guesses"]), len(answers) + 1), dtype=np.int32)
    for t, (correct, gs, _) in enumerate(answers):
        if correct != -1:
            data[params["gs_names"][gs], t] = correct + 1
    resources = [params["resource_names"][lf] for _, _, lf in answers] + [1]
    run_input = {"data": data, "resources": np.array(resources, dtype=np.int64),
                 "starts": np.array([1], dtype=np.int64), "lengths": np.array([len(answers) + 1], dtype=np.int64)}
    _, state_predictions = predict_onestep.run(dict(params), run_input, parallel=False)
    return state_predictions[1][-1]


def test_multigs_updates_match_the_forward_pass():
    """Each multigs and multilearn update gives the forward pass's mastery after that answer."""
    model = model_with_params()
    roster = Roster(["A"], "skill", model=model)
    answers = []
    for correct, gs, lf in [(1, "a", "x"), (0, "c", "y"), (1, "b", "y"), (1, "a", "x"), (0, "b", "x")]:
        roster.update_state("skill", "A", correct, multigs=gs, multilearn=lf)
        answers.append((correct, gs, lf))
        assert roster.get_mastery_prob("skill", "A") == pytest.approx(forward_pass_mastery(model, answers), abs=1e-14)


def test_answer_array_with_unobserved_attempt_matches_the_forward_pass():
    """An answer array with an unobserved attempt gives the forward pass's mastery."""
    model = model_with_params(gs_classes=1, learn_classes=1)
    roster = Roster(["A"], "skill", model=model)
    corrects = [1, 0, -1, 1, 1]

    roster.update_state("skill", "A", np.array(corrects))

    expected = forward_pass_mastery(model, [(c, "a", "x") for c in corrects])
    assert roster.get_mastery_prob("skill", "A") == pytest.approx(expected, abs=1e-14)


def test_single_class_updates_match_the_forward_pass():
    model = model_with_params(gs_classes=1, learn_classes=1)
    roster = Roster(["A", "B"], "skill", model=model)
    answers = {"A": [], "B": []}
    for i, correct in enumerate([1, 1, 0, 1, 0, 0, 1, 1]):
        student = "AB"[i % 2]
        roster.update_state("skill", student, correct)
        answers[student].append((correct, "a", "x"))
    for student in "AB":
        assert roster.get_mastery_prob("skill", student) == pytest.approx(forward_pass_mastery(model, answers[student]), abs=1e-14)


def test_updates_leave_the_model_unchanged():
    model = model_with_params()
    before = copy.deepcopy(model.fit_model)
    roster = Roster(["A"], "skill", model=model)

    for correct, gs in [(1, "a"), (0, "c"), (1, "b")]:
        roster.update_state("skill", "A", correct, multigs=gs, multilearn="y")

    for key, value in before["skill"].items():
        np.testing.assert_array_equal(model.fit_model["skill"][key], value)


def test_update_that_raises_leaves_the_model_unchanged(monkeypatch):
    """An update that fails part way through the answers leaves the shared model as it was."""
    model = model_with_params(gs_classes=1)
    roster = Roster(["A"], "skill", model=model)
    roster.update_state("skill", "A", 1, multilearn="x")
    before = copy.deepcopy(model.fit_model)
    roster_module = sys.modules[Roster.__module__]
    calls = []

    def fail_on_second_answer(*args, **kwargs):
        calls.append(args)
        if len(calls) == 2:
            raise RuntimeError("update failed")
        return roster_module.update_mastery.__wrapped__(*args, **kwargs)

    fail_on_second_answer.__wrapped__ = roster_module.update_mastery
    monkeypatch.setattr(roster_module, "update_mastery", fail_on_second_answer)
    with pytest.raises(RuntimeError, match="update failed"):
        roster.update_state("skill", "A", np.array([1, 0, 1]), multilearn=["x", "y", "y"])

    for key, value in before["skill"].items():
        np.testing.assert_array_equal(model.fit_model["skill"][key], value)


def test_answer_without_a_guess_slip_class_raises():
    roster = Roster(["A"], "skill", model=model_with_params(learn_classes=1))

    with pytest.raises(ValueError, match=r"skill 'skill' has guess/slip classes \['a', 'b', 'c'\]; pass multigs="):
        roster.update_state("skill", "A", 1)


def test_unknown_guess_slip_class_raises():
    roster = Roster(["A"], "skill", model=model_with_params(learn_classes=1))

    with pytest.raises(ValueError, match=r"guess/slip class 'd' is not in the fitted model for skill 'skill'"):
        roster.update_state("skill", "A", 1, multigs="d")


def test_fewer_guess_slip_classes_than_answers_raises():
    roster = Roster(["A"], "skill", model=model_with_params(learn_classes=1))

    with pytest.raises(ValueError, match=r"got 3 answers but 1 guess/slip classes"):
        roster.update_state("skill", "A", np.array([1, 0, 1]), multigs="a")


def test_answer_without_a_learn_class_raises():
    """Without the check every answer silently used the first learn class."""
    roster = Roster(["A"], "skill", model=model_with_params(gs_classes=1))

    with pytest.raises(ValueError, match=r"skill 'skill' has learn classes \['x', 'y'\]; pass multilearn="):
        roster.update_state("skill", "A", 1)


def test_answer_without_a_learn_class_is_allowed_for_one_learn_class():
    model = model_with_params(gs_classes=1, learn_classes=1)
    roster = Roster(["A"], "skill", model=model)

    roster.update_state("skill", "A", 1)

    assert roster.get_mastery_prob("skill", "A") == pytest.approx(forward_pass_mastery(model, [(1, "a", "x")]), abs=1e-14)


def test_unknown_learn_class_raises():
    roster = Roster(["A"], "skill", model=model_with_params(gs_classes=1))

    with pytest.raises(ValueError, match=r"learn class 'z' is not in the fitted model for skill 'skill'; expected one of \['x', 'y'\]"):
        roster.update_state("skill", "A", 1, multilearn="z")


def test_fewer_learn_classes_than_answers_raises():
    """One learn class is not broadcast across an answer array, matching the guess/slip rule."""
    roster = Roster(["A"], "skill", model=model_with_params(gs_classes=1))

    with pytest.raises(ValueError, match=r"got 3 answers but 1 learn classes; pass one class per answer"):
        roster.update_state("skill", "A", np.array([1, 1, 1]), multilearn="x")


def test_answer_array_with_one_learn_class_per_answer_matches_the_forward_pass():
    model = model_with_params(gs_classes=1)
    roster = Roster(["A"], "skill", model=model)
    corrects, learn_classes = [1, 0, 1], ["x", "y", "y"]

    roster.update_state("skill", "A", np.array(corrects), multilearn=learn_classes)

    expected = forward_pass_mastery(model, [(c, "a", lf) for c, lf in zip(corrects, learn_classes)])
    assert roster.get_mastery_prob("skill", "A") == pytest.approx(expected, abs=1e-14)


def test_multiprior_model_updates_without_a_learn_class():
    """A model whose extra learn classes come from multiprior uses Default for every answer, as before."""
    model = model_with_params(gs_classes=1, multilearn=False)
    model.model_type = [False, True, False, False]
    roster = Roster(["A"], "skill", model=model)

    roster.update_state("skill", "A", 1)
    roster.update_state("skill", "A", 0)

    expected = forward_pass_mastery(model, [(1, "a", "x"), (0, "a", "x")])
    assert roster.get_mastery_prob("skill", "A") == pytest.approx(expected, abs=1e-14)


def test_set_model_accepts_a_multigs_model():
    """Roster.set_model with a multigs and multilearn model keeps each student's mastery at the prior."""
    model = model_with_params()
    roster = Roster(["A", "B"], "skill")

    roster.set_model(model)

    for student in "AB":
        assert roster.get_mastery_prob("skill", student) == pytest.approx(PARAMS["prior"])
