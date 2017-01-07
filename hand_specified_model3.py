import numpy as np
from generate import synthetic_data
from fit import EM_fit

#parameters
num_subparts = 1
num_resources = 2
num_fit_initializations = 10
observation_sequence_lengths = np.full(500, 100, dtype=np.int) #np.int is fine?

#generate synthetic model and data.
#model is really easy.
truemodel = {}

#np.float is fine?
truemodel["As"] =  np.zeros((2, 2, num_resources), dtype=np.float_)
for i in range(num_resources):
    #there was oringinally a commented call to the util.dirrnd to fill this array.
    truemodel["As"][i] = np.transpose([[0.7, 0.3], [0.01, 0.99]])
truemodel["learns"] = truemodel["As"][:, 1, 0]
truemodel["forgets"] = truemodel["As"][:, 0, 1]

truemodel["pi_0"] = np.array([0.9, 0.1]) #this used to be col array.
truemodel["prior"] = truemodel["pi_0"][1] #i think this is fine as scalar. but may need to be a 1x1 array.

truemodel["guesses"] = np.full(num_subparts, 0.1, dtype=np.float_) #np.float is fine? used to be 1xnum_subparts
truemodel["slips"] = np.full(num_subparts, 0.03, dtype=np.float_) #np.float is fine? used to be 1xnum_subparts

#data!
print("generating data...")

data = synthetic_data.synthetic_data(truemodel, observation_sequence_lengths) #maybe call this module generate?
#print(data)

#fit models, starting with random initializations
print('fitting! each dot is a new EM initialization')

best_likelihood = float("-inf")

fitmodel = truemodel # NOTE: include this line to initialize at the truth
(fitmodel, log_likelihoods) = EM_fit.EM_fit(fitmodel, data) #these functions should be in a module called generate
if(log_likelihoods[-1] > best_likelihood):
    best_likelihood = log_likelihoods[-1]
    best_model = fitmodel

# compare the fit model to the true model

print('')
print('\ttruth\tlearned')
for r in range(num_resources):
    print('learn%d\t%.4f\t%.4f' % (r+1, truemodel['As'][1, 0, r].squeeze(), best_model['As'][1, 0, r].squeeze()))
for r in range(num_resources):
    print('forget%d\t%.4f\t%.4f' % (r+1, truemodel['As'][0, 1, r].squeeze(), best_model['As'][0, 1, r].squeeze()))

for s in range(num_subparts):
    print('guess%d\t%.4f\t%.4f' % (s+1, truemodel['guesses'][s], best_model['guesses'][s]))
for s in range(num_subparts):
    print('slip%d\t%.4f\t%.4f' % (s+1, truemodel['slips'][s], best_model['slips'][s]))