import numpy as np
from pyBKT.util import check_data
from pyBKT.fit import E_step
from pyBKT.fit import M_step
import os

def EM_fit(model, data, tol = None, maxiter = None, parallel = True, fixed = {}):

    if tol is None: 
        tol = 1e-3
    if maxiter is None:
        maxiter = 100

    num_subparts = data["data"].shape[0] #mmm the first dimension of data represents each subpart?? interesting.
    num_resources = len(model["learns"])
    log_likelihoods = np.zeros((maxiter, 1))

    for i in range(maxiter):
        result = E_step.run(data, model, 1, int(parallel), fixed)
        for j in range(num_resources):
            result['all_trans_softcounts'][j] = result['all_trans_softcounts'][j].transpose()
        for j in range(num_subparts):
            result['all_emission_softcounts'][j] = result['all_emission_softcounts'][j].transpose()

        log_likelihoods[i][0] = result['total_loglike']

        if(i > 1 and abs(log_likelihoods[i][0] - log_likelihoods[i-1][0]) < tol):
            break

        model = M_step.run(model, result['all_trans_softcounts'], result['all_emission_softcounts'], result['all_initial_softcounts'], fixed)

    return(model, log_likelihoods[:i+1])
