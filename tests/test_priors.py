"""Beta priors add pseudo-counts to the M-step, and only where the caller asks for them."""
import numpy as np
import pandas as pd
import pytest

from helpers import simulate_attempts
from pyBKT.fit import M_step
from pyBKT.models import Model


def m_step(priors):
    """Run one M-step on fixed softcounts for one resource and one subpart."""
    model = {"As": np.zeros((1, 2, 2))}
    trans = np.array([[[6.0, 1.0], [4.0, 9.0]]])
    emission = np.array([[[8.0, 2.0], [1.0, 9.0]]])
    initial = np.array([[7.0], [3.0]])
    return M_step.run(model, trans, emission, initial, priors=priors)


def test_each_parameter_becomes_counts_plus_pseudo_counts():
    """Each parameter is (count_yes + a) / (count_yes + count_no + a + b)."""
    fitted = m_step({"prior": (1, 2), "learns": (2, 3), "forgets": (1, 1), "guesses": (4, 6), "slips": (3, 1)})

    assert fitted["prior"] == pytest.approx((3 + 1) / (10 + 3))
    assert fitted["learns"][0] == pytest.approx((4 + 2) / (10 + 5))
    assert fitted["forgets"][0] == pytest.approx((1 + 1) / (10 + 2))
    assert fitted["guesses"][0] == pytest.approx((2 + 4) / (10 + 10))
    assert fitted["slips"][0] == pytest.approx((1 + 3) / (10 + 4))


def test_no_priors_is_maximum_likelihood():
    """Without priors, each parameter is count_yes / (count_yes + count_no)."""
    fitted = m_step(None)

    assert fitted["prior"] == pytest.approx(0.3)
    assert fitted["learns"][0] == pytest.approx(0.4)
    assert fitted["forgets"][0] == pytest.approx(0.1)
    assert fitted["guesses"][0] == pytest.approx(0.2)
    assert fitted["slips"][0] == pytest.approx(0.1)


def two_skills():
    first = simulate_attempts(seed=5, students=60, attempts=8, prior=0.3, learn=0.15, guess=0.2, slip=0.1)
    second = simulate_attempts(seed=6, students=60, attempts=8, prior=0.3, learn=0.15, guess=0.2, slip=0.1)
    second = second.assign(skill_name="other", user_id="b" + second.user_id, order_id=second.order_id + 10_000)
    return pd.concat([first, second], ignore_index=True)


def fit(data, **kwargs):
    model = Model(seed=0, num_fits=1, parallel=False)
    model.fit(data=data.copy(), **kwargs)
    return model


def test_priors_apply_only_to_the_named_skill():
    """A strong guess prior pulls that skill's guess to a / (a + b) and leaves other skills unchanged."""
    data = two_skills()
    plain = fit(data)
    with_prior = fit(data, priors={"skill": {"guesses": (5000, 5000)}})

    assert with_prior.params().loc[("skill", "guesses", "default"), "value"] == pytest.approx(0.5, abs=0.01)
    pd.testing.assert_frame_equal(with_prior.params().loc["other"], plain.params().loc["other"])


def test_a_forgets_prior_without_forgets_warns_and_is_ignored():
    """With forgets=False, a forgets prior warns and leaves forgetting off."""
    data = two_skills()
    plain = fit(data)
    with pytest.warns(UserWarning, match="forgets"):
        with_prior = fit(data, priors={"skill": {"forgets": (50, 50)}})

    pd.testing.assert_frame_equal(with_prior.params(), plain.params())


def test_multiprior_applies_the_prior_to_each_template():
    """Under multiprior, a prior pseudo-count moves each template's prior and a learns one only the learn rate."""
    attempts = simulate_attempts(seed=7, students=120, attempts=8, prior=0.3, learn=0.15, guess=0.2, slip=0.1)
    attempts["first"] = np.where(attempts.user_id.str[-1].isin(list("01234")), "low", "high")

    model = fit(attempts, multiprior=True, defaults={"multiprior": "first"},
                priors={"skill": {"prior": (9000, 1000), "learns": (1, 9999)}})
    fitted = model.fit_model["skill"]
    learns = dict(zip(fitted["resource_names"], fitted["learns"]))

    assert fitted["prior"] == 0
    assert learns["Default"] == pytest.approx(0.0001, abs=0.001)
    assert learns["low"] == pytest.approx(0.9, abs=0.01)
    assert learns["high"] == pytest.approx(0.9, abs=0.01)


def test_one_pair_applies_to_every_template():
    """Under multigs, one guess pair pulls every template's guess rate."""
    attempts = simulate_attempts(seed=8, students=120, attempts=8, prior=0.3, learn=0.15,
                                 guess={"easy": 0.4, "hard": 0.1}, slip=0.1, templates=("easy", "hard"))

    model = fit(attempts, multigs=True, priors={"skill": {"guesses": (5000, 5000)}})
    guesses = model.params().loc["skill"].loc["guesses"]["value"]

    assert guesses["easy"] == pytest.approx(0.5, abs=0.01)
    assert guesses["hard"] == pytest.approx(0.5, abs=0.01)


def test_fixed_values_win_over_priors():
    """A fixed parameter keeps its value when it also has a prior."""
    model = fit(two_skills(), priors={"skill": {"guesses": (5000, 5000)}},
                fixed={"skill": {"guesses": np.array([0.3])}})

    assert model.params().loc[("skill", "guesses", "default"), "value"] == pytest.approx(0.3)


def test_crossvalidate_uses_priors():
    """crossvalidate fits each fold with the priors it is given."""
    data = two_skills()
    plain = Model(seed=0, num_fits=1, parallel=False).crossvalidate(data=data.copy(), folds=2)
    with_prior = Model(seed=0, num_fits=1, parallel=False).crossvalidate(
        data=data.copy(), folds=2, priors={"skill": {"guesses": (5000, 5000)}})

    assert with_prior.loc["other"].equals(plain.loc["other"])
    assert not with_prior.loc["skill"].equals(plain.loc["skill"])


def test_priors_for_an_unknown_skill_warn():
    """A priors key that matches no skill in the data warns; pyBKT turns skill names into strings."""
    with pytest.warns(UserWarning, match="matches no skill"):
        fit(two_skills(), priors={1: {"guesses": (2, 8)}})


@pytest.mark.parametrize("priors", [
    {"guesses": (2, 8)},
    {"skill": {"guess": (2, 8)}},
    {"skill": {"slips": (-1, 9)}},
    {"skill": {"slips": (1, 2, 3)}},
    {"skill": {"slips": (np.inf, 1)}},
])
def test_badly_shaped_priors_raise(priors):
    """priors must map skill names to parameter names to finite, non-negative (a, b) pairs."""
    with pytest.raises(ValueError, match="priors"):
        fit(two_skills(), priors=priors)
