"""Tests that a Model keeps the settings it was created with across calls."""

import numpy as np
import pandas as pd

from helpers import simulate_attempts
from pyBKT.models import Model


def two_skills():
    first = simulate_attempts(0, students=30, attempts=6, prior=0.3, learn=0.2, guess=0.2, slip=0.1)
    second = simulate_attempts(1, students=30, attempts=6, prior=0.4, learn=0.1, guess=0.25, slip=0.1)
    second["skill_name"] = "other"
    second["order_id"] += len(first)
    return pd.concat([first, second], ignore_index=True)


def test_settings_survive_a_second_fit():
    """seed, num_fits and parallel given to Model stay set after more than one fit."""
    df = two_skills()
    model = Model(seed=0, num_fits=3, parallel=False)

    model.fit(data=df.copy())
    model.fit(data=df.copy())

    assert (model.seed, model.num_fits, model.parallel) == (0, 3, False)


def test_crossvalidate_repeats_with_the_same_seed():
    """Two crossvalidate calls on one seeded Model give the same scores."""
    df = two_skills()
    model = Model(seed=0, num_fits=3, parallel=False)

    first = model.crossvalidate(data=df.copy(), folds=3)
    second = model.crossvalidate(data=df.copy(), folds=3)

    assert (model.seed, model.num_fits, model.parallel) == (0, 3, False)
    np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())


def test_a_setting_passed_to_a_call_stays_for_later_calls():
    """A value passed to fit or crossvalidate replaces the one given to Model, and later calls keep it."""
    df = two_skills()
    model = Model(seed=0, num_fits=3, parallel=False)

    model.fit(data=df.copy(), num_fits=2)
    model.fit(data=df.copy())
    model.crossvalidate(data=df.copy(), folds=3, seed=7)
    model.crossvalidate(data=df.copy(), folds=3)

    assert (model.seed, model.num_fits, model.parallel) == (7, 2, False)


def test_fit_options_still_apply_to_one_call():
    """forgets is an option of a single fit, so the next fit without it fits no forgetting."""
    df = two_skills()
    model = Model(seed=0, num_fits=3, parallel=False)

    model.fit(data=df.copy(), forgets=True)
    model.fit(data=df.copy())

    assert model.forgets is False
