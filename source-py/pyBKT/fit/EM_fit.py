#########################################
# EM_fit.py                             #
# EM_fit                                #
#                                       #
# @author Anirudhan Badrinath           #
# @author Christian Garay               #
# Last edited: 20 March 2020            #
#########################################

import numpy as np
from time import time
from pyBKT.util import check_data
from pyBKT.fit import M_step
from multiprocessing import Pool, cpu_count

gs = globals()

def EM_fit(model, data, tol = 0.005, maxiter = 100, parallel = True, fixed = {}):

    check_data.check_data(data)

    num_subparts = data["data"].shape[0] #mmm the first dimension of data represents each subpart?? interesting.
    num_resources = len(model["learns"])

    trans_softcounts = np.zeros((num_resources,2,2))
    emission_softcounts = np.zeros((num_subparts,2,2))
    init_softcounts = np.zeros((2, 1))
    log_likelihoods = np.zeros((maxiter, 1))

    result = {}
    result['all_trans_softcounts'] = trans_softcounts
    result['all_emission_softcounts'] = emission_softcounts
    result['all_initial_softcounts'] = init_softcounts

    for i in range(maxiter):
        result = run(data, model, result['all_trans_softcounts'], result['all_emission_softcounts'], result['all_initial_softcounts'], 1, parallel, fixed = fixed)
        for j in range(num_resources):
            result['all_trans_softcounts'][j] = result['all_trans_softcounts'][j].transpose()
        for j in range(num_subparts):
            result['all_emission_softcounts'][j] = result['all_emission_softcounts'][j].transpose()

        log_likelihoods[i][0] = result['total_loglike']
        if(i > 1 and abs(log_likelihoods[i][0] - log_likelihoods[i-1][0]) <= tol):
            break

        model = M_step.run(model, result['all_trans_softcounts'], result['all_emission_softcounts'], result['all_initial_softcounts'], fixed = fixed)

    return(model, log_likelihoods[:i+1])

def run(data, model, trans_softcounts, emission_softcounts, init_softcounts, num_outputs, parallel = True, fixed = {}):

    # Processed Parameters
    alldata = data["data"]
    bigT, num_subparts = len(alldata[0]), len(alldata)
    allresources, starts, learns, forgets, guesses, slips, lengths = \
            data["resources"], data["starts"], model["learns"], model["forgets"], model["guesses"], model["slips"], data["lengths"]

    prior, num_sequences, num_resources = model["prior"], len(starts), len(learns)
    normalizeLengths = False
    
    if 'prior' in fixed:
        prior = fixed['prior']
    initial_distn = np.empty((2, ), dtype = 'float')
    initial_distn[0] = 1 - prior
    initial_distn[1] = prior
    
    if 'learns' in fixed:
        learns = learns * (fixed['learns'] < 0) + fixed['learns'] * (fixed['learns'] >= 0)
    if 'forgets' in fixed:
        forgets = forgets * (fixed['forgets'] < 0) + fixed['forgets'] * (fixed['forgets'] >= 0)
    As = np.empty((2, 2 * num_resources))
    interleave(As[0], 1 - learns, forgets.copy())
    interleave(As[1], learns.copy(), 1 - forgets)

    if 'guesses' in fixed:
        guesses = fixed['guesses'] * (fixed['guesses'] < 0) + fixed['guesses'] * (fixed['guesses'] >= 0)
    if 'slips' in fixed:
        slips = fixed['slips'] * (fixed['slips'] < 0) + fixed['slips'] * (fixed['slips'] >= 0)
    Bn = np.empty((2, 2 * num_subparts))
    interleave(Bn[0], 1 - guesses, guesses.copy())
    interleave(Bn[1], slips.copy(), 1 - slips)

    # Outputs
    all_trans_softcounts = np.zeros((2, 2 * num_resources))
    all_emission_softcounts = np.zeros((2, 2 * num_subparts))
    all_initial_softcounts = np.zeros((2, 1))

    alpha_out = np.zeros((2, bigT))

    total_loglike = np.empty((1,1))
    total_loglike.fill(0)

    input = {"As": As, "Bn": Bn, "initial_distn": initial_distn, 'allresources': allresources, \
             'starts': starts,
             'lengths': lengths, \
             'num_resources': num_resources, 'num_subparts': num_subparts, \
             'alldata': alldata, 'normalizeLengths': normalizeLengths, 'alpha_out': alpha_out}

    num_threads = cpu_count() if parallel else 1
    thread_counts = [None for i in range(num_threads)]
    for thread_num in range(num_threads):
        blocklen = 1 + ((num_sequences - 1) // num_threads)
        sequence_idx_start = int(blocklen * thread_num)
        sequence_idx_end = min(sequence_idx_start+blocklen, num_sequences)
        thread_counts[thread_num] = {'sequence_idx_start': sequence_idx_start, 'sequence_idx_end': sequence_idx_end}
        thread_counts[thread_num].update(input)

    p = Pool(len(thread_counts))
    x = p.map(inner, thread_counts)
    p.close()

    for i in x:
        total_loglike += i[3]
        all_trans_softcounts += i[0] # + all_trans_softcounts
        all_emission_softcounts += i[1] #+ all_emission_softcounts
        all_initial_softcounts += i[2] #+ all_initial_softcounts
        for sequence_start, T, alpha in i[4]:
            alpha_out[:, sequence_start: sequence_start + T] += alpha
    all_trans_softcounts = all_trans_softcounts.flatten(order = 'F')
    all_emission_softcounts = all_emission_softcounts.flatten(order = 'F')
    result = {}
    result["total_loglike"] = total_loglike;
    result["all_trans_softcounts"] = np.reshape(all_trans_softcounts, (num_resources, 2, 2), order = 'C')
    result["all_emission_softcounts"] = np.reshape(all_emission_softcounts, (num_subparts, 2, 2), order = 'C')
    result["all_initial_softcounts"] = all_initial_softcounts
    result["alpha_out"] = alpha_out.flatten(order = 'F').reshape(alpha_out.shape, order = 'C')

    return result

def interleave(m, v1, v2):
    m[0::2], m[1::2] = v1, v2

def inner(x):
    As, Bn, initial_distn, allresources, starts, lengths, num_resources, num_subparts, alldata, normalizeLengths, alpha_out, sequence_idx_start, sequence_idx_end = \
            x['As'], x['Bn'], x['initial_distn'], x['allresources'], x['starts'], x['lengths'], x['num_resources'], x['num_subparts'], x['alldata'], x['normalizeLengths'], \
            x['alpha_out'], x['sequence_idx_start'], x['sequence_idx_end']
    N_R, N_S = 2 * num_resources, 2 * num_subparts
    trans_softcounts_temp = np.zeros((2, N_R))
    emission_softcounts_temp = np.zeros((2, N_S))
    init_softcounts_temp = np.zeros((2, 1))
    loglike = 0

    alphas = []
    dot, sum, log = np.dot, np.sum, np.log

    for sequence_index in range(sequence_idx_start, sequence_idx_end):
        sequence_start = starts[sequence_index] - 1
        T = lengths[sequence_index]

        # likelihood calculation
        likelihoods = np.ones((2, T))
        alpha = np.empty((2, T))
        for t in range(min(2, T)):
            for n in range(num_subparts):
                data_temp = alldata[n][sequence_start + t]
                if data_temp:
                    sl = Bn[:, 2 * n + int(data_temp == 2)]
                    likelihoods[:,t] *= np.where(sl == 0, 1, sl)
        
        # setup for alpha, included in loop for efficiency (to keep it as one loop)
        alpha[:,0] = initial_distn * likelihoods[:,0]
        norm = sum(alpha[:,0])
        alpha[:,0] /= norm
        contribution = log(norm) / (T if normalizeLengths else 1)
        loglike += contribution

        # combined with t = 2 for efficiency, otherwise we need another loop
        if T >= 2:
            resources_temp = allresources[sequence_start]
            k = 2 * (resources_temp - 1)
            alpha[:, 1] = dot(As[0:2, k: k + 2], alpha[:, 0]) * \
                      likelihoods[:, 1]
            norm = sum(alpha[:, 1])
            alpha[:, 1] /= norm
            contribution = log(norm) / (T if normalizeLengths else 1)
            loglike += contribution

        for t in range(2, T):
            for n in range(num_subparts):
                data_temp = alldata[n][sequence_start + t]
                if data_temp:
                    sl = Bn[:, 2 * n + int(data_temp == 2)]
                    likelihoods[:,t] *= np.where(sl == 0, 1, sl)
            # general loop for alpha calculations
            resources_temp = allresources[sequence_start + t - 1]
            k = 2 * (resources_temp - 1)
            alpha[:, t] = dot(As[0:2, k: k + 2], alpha[:, t - 1]) * \
                      likelihoods[:, t]
            norm = sum(alpha[:, t])
            alpha[:, t] /= norm
            loglike += log(norm) / (T if normalizeLengths else 1)

        # backward messages and statistic counting
        gamma = np.empty((2, T))
        gamma[:, (T - 1)] = alpha[:, (T - 1)].copy()

        # copy it to begin with for efficiency
        As_temp = As.copy()
        # only one pass of the previous update, which is now merged into this loop
        f = True
        for t in range(T - 2, -1, -1):
            resources_temp = allresources[sequence_start + t]
            k = 2 * (resources_temp - 1)
            A = As_temp[0: 2, k: k + 2]
            pair = A.copy() # don't want to modify original A
            pair[0] *= alpha[:, t]
            pair[1] *= alpha[:, t]
            dotted, gamma_t = dot(A, alpha[:, t]), gamma[:, (t + 1)]
            pair[:, 0] = (pair[:, 0] * gamma_t) / dotted
            pair[:, 1] = (pair[:, 1] * gamma_t) / dotted
            np.nan_to_num(pair, copy = False)
            trans_softcounts_temp[0: 2, k: k + 2] += pair
            gamma[:, t] = sum(pair, axis = 0)
            for n in range(num_subparts):
                data_temp = alldata[n][sequence_start+t]
                if data_temp:
                    emission_softcounts_temp[:, (2 * n + int(data_temp == 2))] += gamma[:, t]
                if f:
                    data_temp_p = alldata[n][sequence_start + (T-1)]
                    if data_temp_p:
                        emission_softcounts_temp[:, (2 * n + int(data_temp_p == 2))] += gamma[:, (T - 1)]
            f = False
        init_softcounts_temp += gamma[:, 0].reshape((2, 1))
        alphas.append((sequence_start, T, alpha))
    return [trans_softcounts_temp, emission_softcounts_temp, init_softcounts_temp, loglike, alphas]
