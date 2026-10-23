"""Model and Roster raise TypeError for an argument of the wrong type, and data loading keeps its fallbacks."""
import pytest

from helpers import simulate_attempts
from pyBKT.models import Model, Roster


def fitted_model():
    """A one-skill model fitted on a few simulated students."""
    attempts = simulate_attempts(seed=3, students=20, attempts=5, prior=0.3, learn=0.2, guess=0.2, slip=0.1)
    model = Model(seed=0, num_fits=1, parallel=False)
    model.fit(data=attempts)
    return model, attempts


def test_fit_without_data_raises_type_error():
    with pytest.raises(TypeError, match="no data specified"):
        Model().fit()


def test_fit_with_both_data_and_a_path_raises_type_error():
    attempts = simulate_attempts(seed=3, students=2, attempts=2, prior=0.3, learn=0.2, guess=0.2, slip=0.1)
    with pytest.raises(TypeError, match="cannot specify both"):
        Model().fit(data=attempts, data_path="attempts.csv")


def test_a_missing_data_file_still_raises_value_error():
    with pytest.raises(ValueError, match="file not found"):
        Model().fit(data_path="no-such-file.csv")


def test_roster_constructor_rejects_a_model_that_is_not_a_pybkt_model():
    with pytest.raises(TypeError, match="must be of type pyBKT.models.Model"):
        Roster(["A"], "skill", model=object())


def test_roster_rejects_a_model_that_is_not_a_pybkt_model():
    roster = Roster(["A"], "skill")
    with pytest.raises(TypeError, match="must be of type pyBKT.models.Model"):
        roster.set_model(object())


def test_roster_update_rejects_an_answer_that_is_not_an_int_or_array():
    model, _ = fitted_model()
    roster = Roster(["A"], "skill", model=model)
    with pytest.raises(TypeError, match="need to pass int or np.ndarray"):
        roster.update_state("skill", "A", "1")


def test_corrects_that_are_not_integers_raise_value_error():
    attempts = simulate_attempts(seed=3, students=2, attempts=2, prior=0.3, learn=0.2, guess=0.2, slip=0.1)
    attempts["correct"] = attempts["correct"].astype(object)
    attempts.loc[0, "correct"] = "yes"
    with pytest.raises(ValueError, match="Invalid Data In Specified Corrects Column"):
        Model(num_fits=1, parallel=False).fit(data=attempts)


def test_an_infinite_correct_value_raises_value_error():
    attempts = simulate_attempts(seed=3, students=2, attempts=2, prior=0.3, learn=0.2, guess=0.2, slip=0.1)
    attempts["correct"] = attempts["correct"].astype(float)
    attempts.loc[0, "correct"] = float("inf")
    with pytest.raises(ValueError, match="Invalid Data In Specified Corrects Column"):
        Model(num_fits=1, parallel=False).fit(data=attempts)


def test_a_tab_separated_file_loads_when_its_fields_contain_commas(tmp_path):
    """Rows with and without a comma make the comma parse fail, so loading falls back to tabs."""
    attempts = simulate_attempts(seed=3, students=10, attempts=4, prior=0.3, learn=0.2, guess=0.2, slip=0.1)
    attempts["skill_name"] = ["Mean"] * 20 + ["Box, Whisker"] * 20
    path = tmp_path / "attempts.tsv"
    attempts.to_csv(path, sep="\t", index=False)
    model = Model(seed=0, num_fits=1, parallel=False)
    model.fit(data_path=str(path))
    assert sorted(model.coef_) == ["Box, Whisker", "Mean"]


def test_convert_data_reads_a_csv_downloaded_from_a_url(tmp_path, monkeypatch):
    """The download arrives as bytes, which read_csv only accepts wrapped in a buffer."""
    from pyBKT.util import data_helper

    attempts = simulate_attempts(seed=3, students=4, attempts=3, prior=0.3, learn=0.2, guess=0.2, slip=0.1)

    class Response:
        content = attempts.to_csv(index=False).encode("latin")

    monkeypatch.setattr(data_helper.requests, "get", lambda url: Response())
    monkeypatch.chdir(tmp_path)
    result = data_helper.convert_data("https://example.invalid/attempts.csv", "skill")
    assert list(result) == ["skill"]
    assert (tmp_path / "attempts.csv").exists()
