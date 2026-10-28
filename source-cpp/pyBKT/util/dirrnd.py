from typing import Any

import numpy as np
import numpy.typing as npt

from pyBKT.util.structures import FloatArray, RandomSource


def dirrnd(alphavec: npt.NDArray[Any], rand: RandomSource = np.random) -> FloatArray:
    dimsum = 1
    if(alphavec.ndim == 3 or alphavec.ndim == 2): dimsum = alphavec.ndim - 2
    a = rand.gamma(alphavec, 1)
    a = a / np.sum(a, axis=dimsum, keepdims=True)
    return(a)