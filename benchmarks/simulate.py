"""Seeded BKT data for the benchmarks.

``tests/helpers.simulate_attempts`` builds one row at a time, which takes
minutes at a million rows. This version steps every sequence forward at once.
"""

import numpy as np
import pandas as pd

ATTEMPTS = 20
SKILLS_PER_STUDENT = 5


def simulate(rows, skills=100, templates=3, seed=0):
    """Simulate about ``rows`` answers under the standard BKT model.

    Each student practices ``SKILLS_PER_STUDENT`` different skills for
    ``ATTEMPTS`` attempts each, with the skills interleaved in time. Each skill
    draws its own prior, learn, guess and slip. Each answer gets a random
    ``template_id``, for the multigs and multilearn benchmarks.
    """
    rng = np.random.default_rng(seed)
    sequences = rows // ATTEMPTS
    student = np.arange(sequences) // SKILLS_PER_STUDENT
    slot = np.arange(sequences) % SKILLS_PER_STUDENT
    # Spacing the slots apart keeps a student's skills distinct.
    skill = (student + slot * (skills // SKILLS_PER_STUDENT)) % skills

    prior = rng.uniform(0.1, 0.5, skills)[skill]
    learn = rng.uniform(0.05, 0.3, skills)[skill]
    guess = rng.uniform(0.1, 0.3, skills)[skill]
    slip = rng.uniform(0.05, 0.2, skills)[skill]

    known = rng.random(sequences) < prior
    correct = np.empty((sequences, ATTEMPTS), dtype=np.int64)
    for t in range(ATTEMPTS):
        correct[:, t] = rng.random(sequences) < np.where(known, 1 - slip, guess)
        known |= rng.random(sequences) < learn

    attempt = np.arange(ATTEMPTS)
    order = (student[:, None] * ATTEMPTS + attempt) * SKILLS_PER_STUDENT + slot[:, None]
    # Index into one label per student, skill and template; formatting a
    # string per row takes seconds at a million rows.
    student_names = np.array([f"student{i}" for i in range(student[-1] + 1)], dtype=object)
    skill_names = np.array([f"skill{i}" for i in range(skills)], dtype=object)
    template_names = np.array([f"t{i}" for i in range(templates)], dtype=object)
    return pd.DataFrame(
        {
            "user_id": student_names[np.repeat(student, ATTEMPTS)],
            "skill_name": skill_names[np.repeat(skill, ATTEMPTS)],
            "order_id": order.ravel(),
            "template_id": template_names[rng.integers(0, templates, sequences * ATTEMPTS)],
            "correct": correct.ravel(),
        }
    )
