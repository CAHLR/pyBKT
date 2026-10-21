"""Regenerate the expected values in tests/reference/.

Run it only when a change is meant to alter results, and say why in the pull
request. Run it twice, once per build, from the repository root:

    PYTHONPATH=source-py python tests/make_reference.py   # pure Python
    python tests/make_reference.py                         # installed C++ build

The EM parity values come from the pure-Python build only, because its E-step
follows the BKT equations line by line.
"""
import json
from pathlib import Path

import test_backend_parity
import test_model_outputs
from helpers import IS_COMPILED

REFERENCE_DIR = Path(__file__).parent / "reference"


def write(name, data):
    """Write nested dicts one key per line, with each list of numbers on one line."""

    def format_value(value, depth):
        if not isinstance(value, dict):
            return json.dumps(value)
        pad = "  " * (depth + 1)
        items = [f"{pad}{json.dumps(key)}: {format_value(item, depth + 1)}" for key, item in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + "  " * depth + "}"

    path = REFERENCE_DIR / name
    path.write_text(format_value(data, 0) + "\n")
    print(f"wrote {path}")


if not IS_COMPILED:
    write("em_parity.json", {case: test_backend_parity.fit_em(case) for case in test_backend_parity.CASES})

model_outputs_path = REFERENCE_DIR / "model_outputs.json"
model_outputs = json.loads(model_outputs_path.read_text()) if model_outputs_path.exists() else {}
model_outputs[test_model_outputs.BUILD] = {
    case: test_model_outputs.fit_and_predict(case) for case in test_model_outputs.CASES
}
write("model_outputs.json", model_outputs)
