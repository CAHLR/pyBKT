#########################################
# synthetic_data_helper.py              #
# synthetic_data_helper                 #
#                                       #
# @author Anirudhan Badrinath           #
# Last edited: 26 February 2020         #
#########################################

import numpy as np
from numpy.random import uniform
from time import time
from pyBKT.generate import synthetic_data_helper

def synthetic_data(model, lengths, resources = None):
    start = time()
    num_resources = len(model["learns"])
    bigT = sum(lengths)

    # assert model['learns'].shape == (1, ), "synthetic data generation does not support multilearn"
    # assert model['guesses'].shape == (1, ), "synthetic data generation does not support multiguess/slip"
    # assert model['slips'].shape == (1, ), "synthetic data generation does not support multiguess/slip"

    if resources is None:
        resources = np.random.randint(1, high = num_resources+1, size = bigT)
    if "As" not in model:
        model["As"] = np.array([[1-model["learns"], model["forgets"]], [model["learns"], 1-model["forgets"]]])

    if "pi_0" not in model:
        model["pi_0"] = np.array([[1-model["prior"]], [model["prior"]]])

    starts = np.cumsum(lengths)
    starts = (starts - lengths + 1)[: len(starts)]
    syn_data = synthetic_data_helper.create_synthetic_data(model, starts, lengths, resources)
    d = syn_data['data']
    syn_data["data"] = d + 1 

    # syn_data["data"][:, resources != 1] = 0 #no data emitted unless resource == 1

    datastruct = {}
    datastruct["stateseqs"] = syn_data["stateseqs"]
    datastruct["data"] = syn_data["data"]
    datastruct["starts"] = starts
    datastruct["lengths"] = lengths
    datastruct["resources"] = resources

    return datastruct

def create_synthetic_data(model, starts, lengths, resources):
    """ Randomly models synthetic through the preexisting MODEL, given the
        STARTS and RESOURCES. """
    learns, forgets, guesses, slips = \
            model['learns'], model['forgets'], model['guesses'], model['slips']
    inverted_guess = 1 - guesses
    num_res, num_subparts, num_seqs, num_guesses = \
            len(learns), len(slips), len(starts), len(guesses)
    use_ne = len(guesses) >= 1000
    initial_dist = np.array([1 - model['prior'], model['prior']])

    as_matrix = np.empty((2, 2 * num_res))
    interleave(as_matrix[0], 1 - learns, forgets)
    interleave(as_matrix[1], learns, 1 - forgets)

    req_length = lengths[:num_seqs]
    big_t = int(sum(req_length))

    all_stateseqs, all_data, result = \
            np.empty((1, big_t), dtype = np.int32), np.empty((num_subparts, big_t), dtype = np.int32), {}
    all_data[0][0] = 0
    for seq_index in range(num_seqs):
        seq_start, T = starts[seq_index] - 1, lengths[seq_index]
        nextstate_dist = initial_dist
        big_rand = uniform(size = num_guesses * T)
        other_rand = uniform(size = T)
        loop(big_rand, other_rand, T, num_guesses, nextstate_dist, seq_start, 
             slips, inverted_guess, all_data, all_stateseqs, as_matrix, resources)
    result['stateseqs'], result['data'] = all_stateseqs, all_data
    return result

def loop(big_rand, other_rand, T, num_guesses, nextstate_dist, seq_start, 
             slips, inverted_guess, all_data, all_stateseqs, as_matrix, resources):
  for t in range(T):
    k = seq_start + t
    r = resources[k]
    all_stateseqs[0][k] = a_sq = nextstate_dist[0] < other_rand[t]
    m = t * num_guesses
    rand_arr = big_rand[m: m + num_guesses]
    all_data[:, k] = rand_arr > (slips if a_sq else inverted_guess)
    get = r * 2 + a_sq - 2
    nextstate_dist = as_matrix[:, get]

def interleave(m, v1, v2):
    m[0::2], m[1::2] = v1, v2

