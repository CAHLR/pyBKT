from typing import Literal

from pyBKT.util.structures import (
    CompiledEStepResult,
    CoreParams,
    FixedParams,
    SequenceData,
)

def run(data: SequenceData, model: CoreParams, num_outputs: Literal[1], parallel: bool, fixed: FixedParams, /) -> CompiledEStepResult: ...
