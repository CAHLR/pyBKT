"""Changes to Model's input that must not change its output, checked on generated data."""
import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pyBKT.models import Model, Roster

DEFAULTS = {"multigs": "template_id", "multilearn": "template_id", "multiprior": "template_id"}
every_option = pytest.mark.parametrize(
    "options",
    [{}, {"forgets": True}, {"multigs": True}, {"multilearn": True}, {"multiprior": True}],
    ids=["default", "forgets", "multigs", "multilearn", "multiprior"],
)
generated = settings(max_examples=25, deadline=None, derandomize=True, suppress_health_check=[HealthCheck.too_slow])


@st.composite
def frames(draw):
    """One skill: two to six students, each with one to six answers to items of up to three templates."""
    rows = []
    for student in range(draw(st.integers(2, 6))):
        for _ in range(draw(st.integers(1, 6))):
            correct = draw(st.sampled_from([0, 1, 1, 0, -1]))
            rows.append((f"s{student}", len(rows), draw(st.sampled_from("abc")), correct))
    frame = pd.DataFrame(rows, columns=["user_id", "order_id", "template_id", "correct"]).assign(skill_name="math")
    frame.loc[frame.index[:2], "correct"] = [0, 1]
    return frame


def fit(frame, options):
    model = Model(seed=0, num_fits=1, parallel=False)
    model.fit(data=frame, defaults=DEFAULTS, **options)
    return model


def predictions(model, frame):
    return model.predict(data=frame).loc[frame.index, "correct_predictions"].to_numpy()


@every_option
@generated
@given(frame=frames())
def test_predictions_are_probabilities(options, frame):
    values = predictions(fit(frame, options), frame)
    assert np.all(np.isfinite(values))
    assert np.all((values >= -1e-12) & (values <= 1 + 1e-12))


@every_option
@generated
@given(frame=frames(), seed=st.integers(0, 2**16))
def test_row_order_and_index_labels_do_not_matter(options, frame, seed):
    model = fit(frame, options)
    expected = predictions(model, frame)
    shuffled = frame.sample(frac=1, random_state=seed)
    np.testing.assert_allclose(predictions(model, shuffled)[np.argsort(shuffled.index)], expected, atol=1e-12)
    np.testing.assert_allclose(predictions(model, frame.set_axis(frame.index[::-1] + 100)), expected, atol=1e-12)
    pd.testing.assert_frame_equal(fit(shuffled, options).params(), model.params(), rtol=1e-12, atol=1e-12)


@every_option
@generated
@given(frame=frames(), other=frames())
def test_another_skill_does_not_change_this_one(options, frame, other):
    model = fit(frame, options)
    other = other.assign(skill_name="reading", order_id=other["order_id"] + 1000)
    together = pd.concat([frame, other], ignore_index=True)
    np.testing.assert_allclose(predictions(model, together)[: len(frame)], predictions(model, frame), atol=1e-12)


@every_option
@generated
@given(frame=frames(), position=st.integers(0, 100))
def test_a_missing_answer_keeps_its_own_prediction(options, frame, position):
    """A row's prediction comes before its answer, so hiding the answer can't change it."""
    model = fit(frame, options)
    row = frame.index[position % len(frame)]
    missing = frame.copy()
    missing.loc[row, "correct"] = -1
    before = model.predict(data=frame).loc[row, "correct_predictions"]
    assert model.predict(data=missing).loc[row, "correct_predictions"] == pytest.approx(before, abs=1e-12)


@every_option
@generated
@given(frame=frames())
def test_evaluate_matches_predict(options, frame):
    model = fit(frame, options)
    predicted = model.predict(data=frame)
    answered = predicted[predicted["correct"] != -1]
    expected = np.sqrt(np.mean((answered["correct"] - answered["correct_predictions"]) ** 2))
    assert model.evaluate(data=frame, metric="rmse") == pytest.approx(expected, rel=1e-9)


@generated
@given(frame=frames())
def test_integer_and_string_user_ids_agree(frame):
    model = fit(frame, {})
    numbered = frame.assign(user_id=frame["user_id"].str[1:].astype(int))
    np.testing.assert_allclose(predictions(model, numbered), predictions(model, frame), atol=1e-12)


@generated
@given(frame=frames())
def test_roster_follows_predict(frame):
    model = fit(frame, {})
    student = frame["user_id"].iloc[0]
    history = frame[(frame["user_id"] == student) & (frame["correct"] != -1)]
    roster = Roster([student], "math", model=model)
    for answer in history["correct"]:
        roster.update_state("math", student, int(answer))
    next_row = history.iloc[[-1]].assign(order_id=history["order_id"].max() + 1, correct=-1)
    expected = model.predict(data=pd.concat([history, next_row], ignore_index=True))["state_predictions"].iloc[-1]
    assert roster.get_mastery_prob("math", student) == pytest.approx(expected, rel=1e-9)
