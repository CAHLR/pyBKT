from typing import Any

import numpy as np

def run(data: dict[str, Any], model: dict[str, Any], forward_messages: np.ndarray, parallel: int, /) -> np.ndarray: ...
