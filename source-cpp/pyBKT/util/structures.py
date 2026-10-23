"""Types for the dictionaries pyBKT passes between its fitting functions.

Arrays of parameters hold float64. Answer data holds integers of any width
(0 for no answer, 1 for wrong, 2 for right), with one row per guess/slip
class. Starts, lengths and resources hold integers too, and starts and
resources count from 1.
"""
from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeGuard,
    get_args,
)

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import pandas as pd

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.integer[Any]]
AnswerArray: TypeAlias = npt.NDArray[np.integer[Any]]

# Name maps are keyed by the values of a data column, which can be strings or numbers.
NameMap: TypeAlias = dict[Any, int]


ParamName: TypeAlias = Literal["prior", "learns", "forgets", "guesses", "slips", "As", "emissions", "pi_0",
                              "resource_names", "gs_names"]
FixedName: TypeAlias = Literal["prior", "learns", "forgets", "guesses", "slips"]
PARAM_NAMES: Final = frozenset(get_args(ParamName))
FIXED_NAMES: Final = frozenset(get_args(FixedName))


def is_param_name(name: str) -> TypeGuard[ParamName]:
    """Whether name is a key of Params."""
    return name in PARAM_NAMES


def is_fixed_name(name: str) -> TypeGuard[FixedName]:
    """Whether name is a key of FixedParams."""
    return name in FIXED_NAMES


class RandomSource(Protocol):
    """The random draws pyBKT makes: numpy.random itself or a numpy.random.RandomState."""

    def gamma(self, shape: npt.NDArray[Any], scale: float = ..., size: None = ...) -> FloatArray: ...
    def random(self, size: None = ...) -> float: ...
    def rand(self, d0: int, /) -> FloatArray: ...


class CoreParams(TypedDict):
    """The BKT parameters every model carries."""

    prior: float
    learns: FloatArray
    forgets: FloatArray
    guesses: FloatArray
    slips: FloatArray


class _ModelMatrices(CoreParams):
    """The core parameters plus the matrix forms the M-step reads and writes."""

    As: FloatArray  # resources x 2 x 2 transition matrices
    emissions: FloatArray  # guess/slip classes x 2 x 2
    pi_0: FloatArray  # 2 x 1 initial state distribution


class NameRefs(TypedDict, total=False):
    """The class names a fit records, which convert_data reuses for later data."""

    resource_names: NameMap
    gs_names: NameMap


class Params(_ModelMatrices, NameRefs):
    """A model EM_fit can fit: random_model's output, or a fit with its class names."""


class SyntheticModel(CoreParams, total=False):
    """A model to draw data from. synthetic_data fills in As and pi_0 when they are missing."""

    As: FloatArray
    pi_0: FloatArray


class FixedParams(TypedDict, total=False):
    """Parameters held fixed during a fit. An array entry below 0 leaves that class free."""

    prior: float
    learns: FloatArray
    forgets: FloatArray
    guesses: FloatArray
    slips: FloatArray


class InitialParams(TypedDict, total=False):
    """Starting values that Model.coef_ sets for one skill. The next fit replaces them."""

    prior: float
    learns: FloatArray
    forgets: FloatArray
    guesses: FloatArray
    slips: FloatArray


class FixedSpec(TypedDict, total=False):
    """One skill's entry in Model's fixed option.

    True holds the value coef_ set; a value holds that value. An array entry
    below 0 leaves that class free.
    """

    prior: bool | float
    learns: bool | FloatArray
    forgets: bool | FloatArray
    guesses: bool | FloatArray
    slips: bool | FloatArray


class ModelOptions(TypedDict, total=False):
    """Keyword arguments to Model(). A model variant takes True, or the name of its data column."""

    parallel: bool
    num_fits: int
    seed: int
    defaults: dict[str, str] | None
    multilearn: bool | str
    multiprior: bool | str
    multipair: bool | str
    multigs: bool | str


class FitOptions(TypedDict, total=False):
    """Keyword arguments to Model.fit and Model.partial_fit."""

    skills: str | Sequence[str]
    num_fits: int
    defaults: dict[str, str] | None
    fixed: Literal[True] | dict[str, FixedSpec] | None
    parallel: bool
    forgets: bool
    preload: bool
    multilearn: bool | str
    multiprior: bool | str
    multipair: bool | str
    multigs: bool | str


class CrossvalidateOptions(FitOptions, total=False):
    """Keyword arguments to Model.crossvalidate. folds is a number of folds, or the name of a folds column."""

    folds: int | str
    seed: int


class SequenceData(TypedDict):
    """Answer sequences in the layout the E-step reads."""

    data: AnswerArray
    starts: IntArray
    lengths: IntArray
    resources: IntArray


class SkillData(SequenceData, total=False):
    """One skill's data as convert_data builds it."""

    resource_names: NameMap
    gs_names: NameMap
    index: "pd.Index"
    multiprior_index: IntArray | None
    folds: npt.NDArray[Any]


class SyntheticData(SequenceData):
    """Data drawn from a model, with the hidden states that produced it."""

    stateseqs: npt.NDArray[np.int32]


class SyntheticDraws(TypedDict):
    """Hidden states and answers drawn from a model.

    create_synthetic_data returns answers as 0 for wrong and 1 for right, and
    synthetic_data shifts them to 1 and 2.
    """

    stateseqs: npt.NDArray[np.int32]
    data: npt.NDArray[np.int32]


class Softcounts(TypedDict):
    """Expected counts an E-step returns for the M-step."""

    all_trans_softcounts: FloatArray
    all_emission_softcounts: FloatArray
    all_initial_softcounts: FloatArray


class EStepResult(Softcounts):
    """The pure-Python E-step's result."""

    total_loglike: float
    alpha_out: FloatArray


class CompiledEStepResult(Softcounts):
    """The C++ E-step's result."""

    total_loglike: float
    alpha: FloatArray
