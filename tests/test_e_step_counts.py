"""The E-step's expected counts match the counts from enumerating every hidden path."""
from itertools import product

import numpy as np
import pytest

from helpers import IS_COMPILED


def e_step(data, model):
    """Runs one E-step on whichever build is installed, serially."""
    if IS_COMPILED:
        from pyBKT.fit import E_step

        return E_step.run(data, model, 1, 0, {})
    from pyBKT.fit import EM_fit

    num_resources, num_subparts = len(model["learns"]), data["data"].shape[0]
    return EM_fit.run(
        data, model, np.zeros((num_resources, 2, 2)), np.zeros((num_subparts, 2, 2)), np.zeros((2, 1)), 1, parallel=False
    )


def enumerated_counts(data, model):
    """Expected counts and log-likelihood from every hidden path of every sequence, with no forward-backward."""
    transitions = np.zeros((len(model["learns"]), 2, 2))
    emissions = np.zeros((len(model["guesses"]), 2, 2))
    initial = np.zeros((2, 1))
    loglike = 0.0
    for start, length in zip(data["starts"], data["lengths"]):
        first = int(start) - 1
        paths = []
        for states in product((0, 1), repeat=int(length)):
            weight = model["prior"] if states[0] else 1 - model["prior"]
            for t, state in enumerate(states):
                for subpart, answer in enumerate(data["data"][:, first + t]):
                    if answer:
                        correct = 1 - model["slips"][subpart] if state else model["guesses"][subpart]
                        weight *= correct if answer == 2 else 1 - correct
                if t + 1 < length:
                    resource = data["resources"][first + t] - 1
                    known_next = 1 - model["forgets"][resource] if state else model["learns"][resource]
                    weight *= known_next if states[t + 1] else 1 - known_next
            paths.append((states, weight))
        total = sum(weight for _, weight in paths)
        loglike += np.log(total)
        for states, weight in paths:
            posterior = weight / total
            initial[states[0], 0] += posterior
            for t, state in enumerate(states):
                for subpart, answer in enumerate(data["data"][:, first + t]):
                    if answer:
                        emissions[subpart, answer - 1, state] += posterior
                if t + 1 < length:
                    transitions[data["resources"][first + t] - 1, state, states[t + 1]] += posterior
    return {
        "all_trans_softcounts": transitions,
        "all_emission_softcounts": emissions,
        "all_initial_softcounts": initial,
        "total_loglike": loglike,
    }


@pytest.mark.parametrize("seed", range(8))
def test_e_step_counts_match_path_enumeration(seed):
    rng = np.random.default_rng(seed)
    model = {
        "prior": float(rng.uniform(0.1, 0.9)),
        "learns": rng.uniform(0.01, 0.3, 2),
        "forgets": rng.uniform(0.01, 0.3, 2),
        "guesses": rng.uniform(0.1, 0.4, 3),
        "slips": rng.uniform(0.1, 0.4, 3),
    }
    lengths = np.array([1, 2, 5, 3], dtype=np.int64)
    columns = int(lengths.sum())
    data = {
        "data": rng.integers(0, 3, (3, columns), dtype=np.int32),
        "resources": rng.integers(1, 3, columns, dtype=np.int64),
        "starts": np.r_[1, 1 + np.cumsum(lengths)[:-1]].astype(np.int64),
        "lengths": lengths,
    }

    expected = enumerated_counts(data, model)
    actual = e_step(data, model)

    for key, value in expected.items():
        np.testing.assert_allclose(actual[key], value, rtol=1e-12, atol=1e-12)
