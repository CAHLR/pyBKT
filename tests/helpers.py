"""Helpers shared by the pyBKT tests.

The same tests run against the compiled C++ build and the pure-Python build.
To test pure Python, run pytest with ``PYTHONPATH=source-py`` and without
installing pyBKT.
"""
from importlib.util import find_spec

import numpy as np
import pandas as pd

IS_COMPILED = find_spec("pyBKT.fit.E_step") is not None


def simulate_attempts(
    seed, students, attempts, prior, learn, guess, slip, forget=0.0, templates=("t",)
):
    """Simulate students answering one skill under the standard BKT model.

    Each student starts in the known state with probability ``prior``. Attempt
    ``t`` uses template ``templates[t % len(templates)]``. A known student
    answers correctly with probability ``1 - slip``; an unknown student guesses
    correctly with probability ``guess``. After each attempt, an unknown student
    learns with probability ``learn``, and a known student forgets with
    probability ``forget``. ``learn``, ``guess`` and ``slip`` may be a number or
    a dict from template to number.
    """
    rng = np.random.default_rng(seed)

    def value(parameter, template):
        return parameter[template] if isinstance(parameter, dict) else parameter

    rows = []
    for student in range(students):
        known = rng.random() < prior
        for t in range(attempts):
            template = templates[t % len(templates)]
            if known:
                correct = rng.random() >= value(slip, template)
            else:
                correct = rng.random() < value(guess, template)
            rows.append(
                {
                    "user_id": f"student{student}",
                    "skill_name": "skill",
                    "order_id": student * attempts + t,
                    "template_id": template,
                    "correct": int(correct),
                }
            )
            if known:
                known = rng.random() >= forget
            else:
                known = rng.random() < value(learn, template)
    return pd.DataFrame(rows)
