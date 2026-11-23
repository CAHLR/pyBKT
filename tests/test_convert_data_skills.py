"""Converting many skills at once gives each skill the same data as converting it alone."""
import numpy as np
import pandas as pd
import pytest

from helpers import simulate_attempts
from pyBKT.util import data_helper


@pytest.fixture
def interleaved_skills():
    """Three skills whose rows are mixed together, as in a real log of answers."""
    frames = []
    for number, skill in enumerate(["fractions", "decimals", "percents"]):
        attempts = simulate_attempts(
            seed=number, students=20, attempts=5, prior=0.3, learn=0.2, guess=0.2, slip=0.1, templates=("a", "b")
        )
        attempts["skill_name"] = skill
        attempts["order_id"] = attempts["order_id"] * 3 + number
        frames.append(attempts)
    return pd.concat(frames).sample(frac=1, random_state=0).reset_index(drop=True)


@pytest.mark.parametrize("model_type", [None, [True, False, False, True]], ids=["plain", "multilearn_multigs"])
def test_each_skill_matches_converting_it_alone(interleaved_skills, model_type):
    together = data_helper.convert_data(interleaved_skills.copy(), ".*", model_type=model_type)

    assert set(together) == {"fractions", "decimals", "percents"}
    for skill, data in together.items():
        alone = data_helper.convert_data(interleaved_skills.copy(), skill, model_type=model_type)[skill]
        for key in ["data", "starts", "lengths", "resources", "index"]:
            np.testing.assert_array_equal(data[key], alone[key])
        assert data["resource_names"] == alone["resource_names"]
        assert data["gs_names"] == alone["gs_names"]
