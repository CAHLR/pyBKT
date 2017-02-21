import numpy as np
from generate import synthetic_data
from fit import EM_fit
from copy import deepcopy

#parameters
num_subparts = 1
num_resources = 2
num_fit_initializations = 10
observation_sequence_lengths = np.full(500, 100, dtype=np.int) #this used to be (1x500).

#generate synthetic model and data.
#model is really easy.
truemodel = {}

truemodel["As"] =  np.zeros((2, 2, num_resources), dtype=np.float_)
for i in range(num_resources):
    truemodel["As"][i, :, :] = np.transpose([[0.7, 0.3], [0.01, 0.99]]) #the third dimension goes first here, different from matlab.
truemodel["learns"] = truemodel["As"][:, 1, 0] # from state 1 to state 2
truemodel["forgets"] = truemodel["As"][:, 0, 1]

truemodel["pi_0"] = np.array([[0.9], [0.1]])
truemodel["prior"] = truemodel["pi_0"][1][0]

truemodel["guesses"] = np.full(num_subparts, 0.1, dtype=np.float_)
truemodel["slips"] = np.full(num_subparts, 0.03, dtype=np.float_) #np.float is fine? used to be 1xnum_subparts

#data!
print("generating data...")
data = synthetic_data.synthetic_data(truemodel, observation_sequence_lengths)

#fit models, starting with random initializations
print('fitting! each dot is a new EM initialization')

best_likelihood = float("-inf")

fitmodel = deepcopy(truemodel) # NOTE: include this line to initialize at the truth
(fitmodel, log_likelihoods) = EM_fit.EM_fit(fitmodel, data)
if(log_likelihoods[-1] > best_likelihood):
    best_likelihood = log_likelihoods[-1]
    best_model = fitmodel

# compare the fit model to the true model

print('')
print('\ttruth\tlearned')
for r in range(num_resources):
    print('learn%d\t%.4f\t%.4f' % (r+1, truemodel['As'][r, 1, 0].squeeze(), best_model['As'][r, 1, 0].squeeze()))
for r in range(num_resources):
    print('forget%d\t%.4f\t%.4f' % (r+1, truemodel['As'][r, 0, 1].squeeze(), best_model['As'][r, 0, 1].squeeze()))

for s in range(num_subparts):
    print('guess%d\t%.4f\t%.4f' % (s+1, truemodel['guesses'][s], best_model['guesses'][s]))
for s in range(num_subparts):
    print('slip%d\t%.4f\t%.4f' % (s+1, truemodel['slips'][s], best_model['slips'][s]))
