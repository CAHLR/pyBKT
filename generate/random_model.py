import numpy as np
from pyBKT.util import dirrnd

def random_model(num_resources=None, num_subparts=None, trans_prior=None, given_notknow_prior=None, given_know_prior=None, pi_0_prior=None):

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

    #fixing data for testing purposes
    # As = np.zeros((num_resources, 2, 2), dtype=np.float_)
    # As[0, :, :] = np.transpose([[0.87, 0.13], [0.04, 0.96]])
    # As[1, :, :] = np.transpose([[0.84, 0.16], [0.01, 0.99]])
    #
    # given_know[0] = [0.01, 0.04, 0.01, 0.14]
    # given_know[1] = [0.99, 0.96, 0.99, 0.86]
    #
    # given_notknow[0] = [0.84, 0.85, 0.97, 0.93]
    # given_notknow[1] = [0.16, 0.15, 0.03, 0.07]
    #
    # pi_0 = np.array([[0.99], [0.01]])
    #
    # emissions[0, :, :] = np.transpose([[0.84, 0.01], [0.16, 0.99]])
    # emissions[1, :, :] = np.transpose([[0.85, 0.04], [0.15, 0.96]])
    # emissions[2, :, :] = np.transpose([[0.97, 0.01], [0.03, 0.99]])
    # emissions[3, :, :] = np.transpose([[0.93, 0.14], [0.07, 0.86]])

    modelstruct = {}

    modelstruct['prior'] = pi_0[1][0]
    modelstruct['learns'] = As[:, 1, 0]
    modelstruct['forgets'] = As[:, 0, 1]
    modelstruct['guesses'] = given_notknow[1, :]
    modelstruct['slips'] = given_know[0, :]

    modelstruct['As'] = As
    modelstruct['emissions'] = emissions
    modelstruct['pi_0'] = pi_0

    return(modelstruct)