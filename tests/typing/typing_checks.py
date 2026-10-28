"""Checks that pyBKT's type hints accept valid calls and reject common mistakes.

The lint job runs mypy on this file against each package copy; nothing runs it.
Each mistake carries the `type: ignore` code mypy reports for it. If a type
loosens and mypy stops reporting one, strict mode flags the unused ignore and
the job fails.
"""

from collections.abc import Hashable

import numpy as np
import pandas as pd
from typing_extensions import assert_type

from pyBKT.fit import EM_fit, M_step
from pyBKT.generate import random_model_uni, synthetic_data
from pyBKT.models import Model, Roster, State
from pyBKT.util import data_helper
from pyBKT.util.structures import (
    FloatArray,
    InitialParams,
    Params,
    SequenceData,
    SkillData,
)

frame = pd.DataFrame()
data: SequenceData = {
    "data": np.zeros((1, 4), dtype=np.int64),
    "starts": np.array([1], dtype=np.int64),
    "lengths": np.array([4], dtype=np.int64),
    "resources": np.ones(4, dtype=np.int64),
}
model: Params = random_model_uni.random_model_uni(1, 1)

# Valid calls and the types they return.
bkt = Model(seed=42, num_fits=1, multigs=True)
bkt.fit(data=frame, skills="Box and Whisker", forgets=True, fixed={"Box and Whisker": {"learns": True}})
assert_type(bkt.predict(data=frame), pd.DataFrame)
assert_type(bkt.evaluate(data=frame, metric=("rmse", "auc")), float | list[float])
metric_names = ["rmse", "auc"]
assert_type(bkt.evaluate(data=frame, metric=metric_names), float | list[float])
bkt.fit(data=frame, skills=("s1", "s2"), defaults=None, fixed=None)
assert_type(bkt.crossvalidate(data=frame, folds=3), pd.DataFrame)
assert_type(bkt.coef_, dict[str, InitialParams])
bkt.coef_ = {"Box and Whisker": {"prior": 0.5, "learns": np.array([0.1])}}
assert_type(EM_fit.EM_fit(model, data, fixed={"prior": 0.4}), tuple[Params, FloatArray])
assert_type(data_helper.convert_data(frame, "s", None, None, None, None, True), tuple[dict[str, SkillData], pd.DataFrame])
assert_type(data_helper.convert_data(frame, "s"), dict[str, SkillData])
synthetic_data.synthetic_data({"prior": 0.5, "learns": np.array([0.1]), "forgets": np.array([0.0]),
                               "guesses": np.array([0.2]), "slips": np.array([0.1])}, np.array([4]))
roster = Roster(["Morgan", "Bob"], "Box and Whisker", model=bkt)
assert_type(roster.update_state("Box and Whisker", "Morgan", 1), State)
assert_type(roster.get_mastery_probs("Box and Whisker"), dict[Hashable, float])

roster.update_state("Box and Whisker", "Bob", -1)
roster.update_state("Box and Whisker", "Bob", True)
Model(defaults={"user_id": "student", "correct": "right"})

# Mistakes the types must reject.
Model(num_fit=5)  # type: ignore[call-arg]
bkt.fit(data=frame, fixed={"Box and Whisker": {"learn": True}})  # type: ignore[typeddict-unknown-key]
EM_fit.EM_fit(model, data, fixed={"learn": np.array([0.1])})  # type: ignore[arg-type]
missing_lengths: SequenceData = {"data": data["data"], "starts": data["starts"], "resources": data["resources"]}  # type: ignore[typeddict-item]
EM_fit.EM_fit({"prior": 0.5, "learns": np.array([0.1]), "forgets": np.array([0.0]),  # type: ignore[typeddict-item]
               "guesses": np.array([0.2]), "slips": np.array([0.1])}, data)
counts = np.zeros((1, 2, 2), dtype=np.int64)
M_step.run(model, counts, counts, np.zeros((2, 1)))  # type: ignore[arg-type]
roster.update_state("Box and Whisker", "Morgan", "1")  # type: ignore[arg-type]
roster.update_state("Box and Whisker", "Morgan", 2)  # type: ignore[arg-type]
Model(defaults={"user": "student"})  # type: ignore[arg-type]
