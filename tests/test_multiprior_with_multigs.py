"""multiprior and multigs together: each answer stays in its own column, next to the multiprior placeholders."""
import numpy as np
import pandas as pd

from helpers import simulate_attempts
from pyBKT.models import Model
from pyBKT.util.data_helper import convert_data

DEFAULTS = {"multiprior": "template_id", "multigs": "template_id"}


def attempts():
    return simulate_attempts(
        seed=0, students=30, attempts=6, prior=0.3, learn=0.2, guess=0.2, slip=0.1, templates=("a", "b", "c")
    )


def test_multigs_columns_line_up_with_multiprior_placeholders():
    data = attempts()
    both = convert_data(data, ".*", defaults=DEFAULTS, model_type=[False, True, False, True])["skill"]
    prior_only = convert_data(data, ".*", defaults=DEFAULTS, model_type=[False, True, False, False])["skill"]

    np.testing.assert_array_equal(both["data"].sum(axis=0), prior_only["data"][0])
    np.testing.assert_array_equal(both["resources"], prior_only["resources"])
    np.testing.assert_array_equal(both["data"][:, prior_only["multiprior_index"]], 0)

    # each answer sits in its own class's row, as it does with multigs alone
    gs_only = convert_data(data, ".*", defaults=DEFAULTS, model_type=[False, False, False, True])["skill"]
    answers = np.delete(both["data"], prior_only["multiprior_index"], axis=1)
    np.testing.assert_array_equal(answers, gs_only["data"])


def test_one_guess_slip_class_fits_like_multiprior_alone():
    data = attempts().assign(gs_class="only")
    defaults = {"multiprior": "template_id", "multigs": "gs_class"}
    both = Model(seed=0, num_fits=1, parallel=False)
    both.fit(data=data, defaults=defaults, multiprior=True, multigs=True)
    prior_only = Model(seed=0, num_fits=1, parallel=False)
    prior_only.fit(data=data, defaults=defaults, multiprior=True)

    pd.testing.assert_frame_equal(both.predict(data=data), prior_only.predict(data=data), check_exact=True)
