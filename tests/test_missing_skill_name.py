"""A missing skill name becomes the skill 'nan', whatever the pandas version and dtypes."""
import importlib.util

import numpy as np
import pandas as pd
import pytest

from helpers import simulate_attempts
from pyBKT.models import Model
from pyBKT.util.data_helper import convert_data


def attempts():
    data = simulate_attempts(seed=0, students=30, attempts=8, prior=0.3, learn=0.2, guess=0.2, slip=0.1)
    data["skill_name"] = ["s" if i % 2 else None for i in range(len(data))]
    return data


def skill_columns():
    names = attempts()["skill_name"]
    columns = {
        "object": names.astype(object),
        "string": names.astype("string"),
        "category": names.astype("category"),
    }
    if int(pd.__version__.split(".")[0]) >= 3:
        # before pandas 3, astype("str") writes the text 'None' rather than keeping the value missing
        columns["str"] = names.astype("str")
    if importlib.util.find_spec("pyarrow") is not None:
        columns["pyarrow"] = names.astype("string[pyarrow]")
    return columns


@pytest.mark.parametrize("dtype", list(skill_columns()))
def test_missing_skill_name_is_nan_for_every_dtype(dtype):
    data = attempts().assign(skill_name=skill_columns()[dtype])
    assert sorted(convert_data(data, ".*")) == ["nan", "s"]


def test_predictions_do_not_depend_on_the_dtype():
    model = Model(seed=0, num_fits=1, parallel=False)
    model.fit(data=attempts().assign(skill_name=skill_columns()["object"]))
    expected = model.predict(data=attempts())["correct_predictions"].to_numpy()
    for column in skill_columns().values():
        actual = model.predict(data=attempts().assign(skill_name=column))["correct_predictions"].to_numpy()
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("old_name", ["None", "<NA>"])
def test_load_renames_an_old_missing_skill(old_name, tmp_path):
    data = attempts()
    model = Model(seed=0, num_fits=1, parallel=False)
    model.fit(data=data)
    expected = model.predict(data=data)
    model.fit_model[old_name] = model.fit_model.pop("nan")
    model.skills = [old_name if skill == "nan" else skill for skill in model.skills]
    model.save(tmp_path / "model.pkl")

    loaded = Model()
    loaded.load(tmp_path / "model.pkl")

    assert sorted(loaded.fit_model) == ["nan", "s"]
    pd.testing.assert_frame_equal(loaded.predict(data=data), expected, check_exact=True)
