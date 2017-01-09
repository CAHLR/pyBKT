import numpy as np
from util import check_data
from fit import E_step
from fit import M_step

def EM_fit(model, data, tol = None, maxiter = None):
    if tol is None: tol = 1e-3
    if maxiter is None: maxiter = 100

    check_data.check_data(data)

    num_subparts = data["data"].shape[0] #mmm the first dimension of data represents each subpart?? interesting.
    num_resources = len(model["learns"])

    trans_softcounts = np.zeros((2, 2, num_resources))
    emission_softcounts = np.zeros((2, 2, num_subparts))
    init_softcounts = np.zeros((2, 1))
    log_likelihoods = np.zeros((maxiter, 1))

    # print(data)
    # print(model)
    # print(trans_softcounts)
    # print(emission_softcounts)
    # print(init_softcounts)
    # result = E_step.run(data, model, trans_softcounts, emission_softcounts, init_softcounts, 1)

    for i in range(maxiter):
        result = E_step.run(data, model, trans_softcounts, emission_softcounts, init_softcounts, 1)
        log_likelihoods[i] = result['total_loglike']

        if(i > 1 and abs(log_likelihoods[i] - log_likelihoods[i-1]) < tol):
            break

        model = M_step.run(model, trans_softcounts, emission_softcounts, init_softcounts)

    return(model, log_likelihoods[1:i])