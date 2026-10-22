"""update_mastery gives the same mastery as the forward pass behind Model.predict, one answer at a time.

Runs against the compiled build, or the pure-Python build with ``PYTHONPATH=source-py``.
"""
from fractions import Fraction

import numpy as np
import pytest

from pyBKT.fit import predict_onestep
from pyBKT.fit.update_mastery import transition_mastery, update_mastery, update_mastery_array

# Three guess/slip classes and two learn/forget classes, so every index is exercised.
PARAMS = {
    "prior": 0.3,
    "learns": np.array([0.1, 0.25]),
    "forgets": np.array([0.02, 0.05]),
    "guesses": np.array([0.2, 0.3, 0.1]),
    "slips": np.array([0.1, 0.05, 0.2]),
}


def simulate_students(seed, students=40, attempts=25):
    """Each answer is (correct, guess/slip class, learn/forget class) drawn under the BKT model.
    About one answer in ten is unobserved, with correct set to None."""
    rng = np.random.default_rng(seed)
    sequences = []
    for _ in range(students):
        known = rng.random() < PARAMS["prior"]
        answers = []
        for _ in range(rng.integers(1, attempts)):
            gs, lf = int(rng.integers(3)), int(rng.integers(2))
            if known:
                correct = rng.random() >= PARAMS["slips"][gs]
                known = rng.random() >= PARAMS["forgets"][lf]
            else:
                correct = rng.random() < PARAMS["guesses"][gs]
                known = rng.random() < PARAMS["learns"][lf]
            answers.append((None if rng.random() < 0.1 else bool(correct), gs, lf))
        sequences.append(answers)
    return sequences


def as_predict_input(sequences):
    """Packs sequences in the layout predict_onestep.run reads: one data row per guess/slip class,
    2 for correct, 1 for incorrect and 0 for no observation."""
    answers = [a for sequence in sequences for a in sequence]
    data = np.zeros((3, len(answers)), dtype=np.int32)
    for t, (correct, gs, _) in enumerate(answers):
        if correct is not None:
            data[gs, t] = 2 if correct else 1
    lengths = np.array([len(s) for s in sequences], dtype=np.int64)
    return {
        "data": data,
        "resources": np.array([lf + 1 for _, _, lf in answers], dtype=np.int64),
        "lengths": lengths,
        "starts": np.cumsum(lengths) - lengths + 1,
    }


def test_stepping_through_answers_reproduces_predict_state_predictions():
    sequences = simulate_students(seed=0)
    _, state_predictions = predict_onestep.run(dict(PARAMS), as_predict_input(sequences), parallel=False)

    stepped = []
    for sequence in sequences:
        mastery = PARAMS["prior"]
        for correct, gs, lf in sequence:
            stepped.append(mastery)
            learn, forget = PARAMS["learns"][lf], PARAMS["forgets"][lf]
            if correct is None:
                mastery = transition_mastery(mastery, learn, forget)
            else:
                mastery = update_mastery(mastery, correct, PARAMS["guesses"][gs], PARAMS["slips"][gs], learn, forget)

    assert any(correct is None for sequence in sequences for correct, _, _ in sequence)
    np.testing.assert_allclose(stepped, state_predictions[1], rtol=0, atol=1e-12)


def test_array_form_matches_scalar_form_exactly():
    rng = np.random.default_rng(1)
    n = 1000
    mastery = np.concatenate([rng.random(n - 4), [1e-300, 1e-12, 1 - 1e-12, 1.0]])
    correct = rng.integers(0, 2, n)
    guess, slip = rng.uniform(0.01, 0.5, n), rng.uniform(0.01, 0.5, n)
    learn, forget = rng.uniform(0, 0.5, n), rng.uniform(0, 0.2, n)

    vectorized = update_mastery_array(mastery, correct, guess, slip, learn, forget)
    scalar = [update_mastery(*args) for args in zip(mastery, correct, guess, slip, learn, forget)]
    np.testing.assert_array_equal(vectorized, scalar)

    transitioned = transition_mastery(mastery, learn, forget)
    np.testing.assert_array_equal(transitioned, [transition_mastery(*args) for args in zip(mastery, learn, forget)])


@pytest.mark.parametrize("mastery", [1e-12, 1 - 1e-12])
@pytest.mark.parametrize("correct", [True, False])
def test_update_keeps_full_precision_near_zero_and_one(mastery, correct):
    guess, slip = 0.2, 0.1
    m, g, s = Fraction(mastery), Fraction(guess), Fraction(slip)
    known, unknown = (m * (1 - s), (1 - m) * g) if correct else (m * s, (1 - m) * (1 - g))
    exact = float(known / (known + unknown))

    result = update_mastery(mastery, correct, guess, slip, learn=0.0)

    assert result == pytest.approx(exact, rel=1e-15, abs=0)


@pytest.mark.parametrize("correct", [True, False])
def test_a_guess_of_one_or_a_slip_of_zero_matches_predict(correct):
    """With guess 1 and slip 0, an answer has probability 0 in one state; the forward pass reads that 0 as 1."""
    params = dict(PARAMS, guesses=np.array([1.0]), slips=np.array([0.0]), learns=np.array([0.15]), forgets=np.array([0.0]))
    data = {
        "data": np.array([[2, 2 if correct else 1, 1]], dtype=np.int32),
        "resources": np.ones(3, dtype=np.int64),
        "lengths": np.array([3], dtype=np.int64),
        "starts": np.array([1], dtype=np.int64),
    }
    _, state_predictions = predict_onestep.run(dict(params), data, parallel=False)

    scalar, vectorized = PARAMS["prior"], np.array([PARAMS["prior"]])
    for answer in (True, correct):
        scalar = update_mastery(scalar, answer, 1.0, 0.0, 0.15)
        vectorized = update_mastery_array(vectorized, np.array([answer]), 1.0, 0.0, 0.15)

    assert scalar == pytest.approx(state_predictions[1][2], abs=1e-14)
    np.testing.assert_array_equal(vectorized, [scalar])
