import random
import numpy as np
from pyBKT.util import dirrnd

def random_model_uni(num_resources=None, num_subparts=None, trans_prior=None, given_notknow_prior=None, given_know_prior=None, pi_0_prior=None):

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

    As = dirrnd.dirrnd(trans_prior)
    given_notknow = dirrnd.dirrnd(given_notknow_prior)
    given_know = dirrnd.dirrnd(given_know_prior)
    #emissions = np.dstack((given_notknow.reshape((num_subparts, 2, 1)), given_know.reshape((num_subparts, 2, 1))))
    emissions = np.stack((np.transpose(given_notknow.reshape((2, num_subparts))), np.transpose(given_know.reshape((2, num_subparts)))), axis=1)
    pi_0 = dirrnd.dirrnd(pi_0_prior)

    modelstruct = {}
    modelstruct['prior'] = random.random()
    As[:, 1, 0] = np.random.rand(num_resources) * 0.40
    As[:, 1, 1] = 1 - As[:, 1, 0]
    As[:, 0, 1] = 0
    As[:, 0, 0] = 1
    modelstruct['learns'] = As[:, 1, 0]
    modelstruct['forgets'] = As[:, 0, 1]
    given_notknow[1, :] = np.random.rand(num_subparts) * 0.40
    modelstruct['guesses'] = given_notknow[1, :]
    given_know[0, :] = np.random.rand(num_subparts) * 0.30
    modelstruct['slips'] = given_know[0, :]

    modelstruct['As'] = As
    modelstruct['emissions'] = emissions
    modelstruct['pi_0'] = pi_0

    return(modelstruct)