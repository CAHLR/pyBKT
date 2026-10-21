
"""Tests for preserving response order when order IDs are repeated."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_data_helper(source_dir):
    """Load the data helper from the selected implementation."""
    path = REPO_ROOT / source_dir / "pyBKT" / "util" / "data_helper.py"
    spec = importlib.util.spec_from_file_location(
        f"{source_dir.replace('-', '_')}_data_helper", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def response_frames():
    rng = np.random.default_rng(1)
    rows = []

    for student in range(40):
        for order_id in range(8):
            for probability in (0.7, 0.4):
                rows.append({
                    "user_id": f"u{student}",
                    "skill_name": "s",
                    "order_id": order_id,
                    "correct": int(rng.random() < probability),
                })

    grouped = pd.DataFrame(rows)
    interleaved = grouped.sort_values(
        "order_id", kind="mergesort"
    ).reset_index(drop=True)

    return grouped, interleaved


@pytest.mark.parametrize("source_dir", ["source-cpp", "source-py"])
def test_convert_data_preserves_tied_order_ids(source_dir, response_frames):
    helper = load_data_helper(source_dir)
    grouped, interleaved = response_frames

    first = helper.convert_data(grouped.copy(), "s")["s"]
    second = helper.convert_data(interleaved.copy(), "s")["s"]

    np.testing.assert_array_equal(first["data"], second["data"])
    np.testing.assert_array_equal(first["starts"], second["starts"])
    np.testing.assert_array_equal(first["lengths"], second["lengths"])


def test_model_fit_is_consistent(response_frames, monkeypatch):
    # Use the pure-Python implementation for the fitting test.-
    monkeypatch.syspath_prepend(str(REPO_ROOT / "source-py"))

    from pyBKT.models import Model

    grouped, interleaved = response_frames
    parameters = []

    for frame in (grouped, interleaved):
        model = Model(seed=0, num_fits=1, parallel=False)
        model.fit(data=frame.copy())
        parameters.append(model.params().values.ravel())

    np.testing.assert_allclose(
        parameters[0], parameters[1], rtol=1e-7, atol=1e-9
    )
