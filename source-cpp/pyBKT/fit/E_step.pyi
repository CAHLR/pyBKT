from pyBKT.util.structures import (
    CompiledEStepResult,
    CoreParams,
    FixedParams,
    SequenceData,
)

def run(data: SequenceData, model: CoreParams, num_outputs: int, parallel: int, fixed: FixedParams, /) -> CompiledEStepResult: ...
