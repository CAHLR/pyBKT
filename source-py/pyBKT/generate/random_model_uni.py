from typing import Any

import numpy as np
import numpy.typing as npt

from pyBKT.util import dirrnd
from pyBKT.util.structures import Params, RandomSource


def random_model_uni(num_resources: int | None = None, num_subparts: int | None = None, trans_prior: npt.NDArray[Any] | None = None,
                     given_notknow_prior: npt.NDArray[Any] | None = None, given_know_prior: npt.NDArray[Any] | None = None,
                     pi_0_prior: npt.NDArray[Any] | None = None,
                     rand: RandomSource = np.random) -> Params:

    if num_resources is None: num_resources = 1
    if num_subparts is None: num_subparts = 1

    if trans_prior is None:
        trans_prior = np.tile(np.transpose([[20, 4], [1, 20]]), (num_resources, 1)).reshape((num_resources, 2, 2))
    if given_notknow_prior is None:
        given_notknow_prior = np.tile([[5], [0.5]], (1, num_subparts))
    if given_know_prior is None:
        given_know_prior = np.tile([[0.5], [5]], (1, num_subparts))
    if pi_0_prior is None:
        pi_0_prior = np.array([[100], [1]])

    As = dirrnd.dirrnd(trans_prior, rand)
    given_notknow = dirrnd.dirrnd(given_notknow_prior, rand)
    given_know = dirrnd.dirrnd(given_know_prior, rand)
    #emissions = np.dstack((given_notknow.reshape((num_subparts, 2, 1)), given_know.reshape((num_subparts, 2, 1))))
    emissions = np.stack((np.transpose(given_notknow.reshape((2, num_subparts))), np.transpose(given_know.reshape((2, num_subparts)))), axis=1)
    pi_0 = dirrnd.dirrnd(pi_0_prior, rand)

    prior = rand.random()
    As[:, 1, 0] = rand.rand(num_resources) * 0.40
    As[:, 1, 1] = 1 - As[:, 1, 0]
    As[:, 0, 1] = 0
    As[:, 0, 0] = 1
    given_notknow[1, :] = rand.rand(num_subparts) * 0.40
    given_know[0, :] = rand.rand(num_subparts) * 0.30
    modelstruct: Params = {'prior': prior, 'learns': As[:, 1, 0], 'forgets': As[:, 0, 1], 'guesses': given_notknow[1, :],
                           'slips': given_know[0, :], 'As': As, 'emissions': emissions, 'pi_0': pi_0}

    return(modelstruct)