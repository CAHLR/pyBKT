import sys, os
sys.path.append('..')
import numpy as np
from pyBKT.generate import synthetic_data, random_model_uni
from pyBKT.fit import EM_fit
from copy import deepcopy

#parameters classes
num_gs = 1 #number of guess/slip classes
num_learns = 1 #number of learning rates

num_fit_initializations = 20

#true params used for synthetic data generation
p_T = 0.30
p_F = 0.00
p_G = 0.10
p_S = 0.03
p_L0 = 0.10

#generate synthetic model and data.
truemodel = {}

truemodel["As"] =  np.zeros((num_learns,2,2), dtype=np.float_)
for i in range(num_learns):
    truemodel["As"][i] = np.transpose([[1-p_T, p_T], [p_F, 1-p_F]])

truemodel["learns"] = truemodel["As"][:,1, 0,]
truemodel["forgets"] = truemodel["As"][:,0, 1]

truemodel["pi_0"] = np.array([[1-p_L0], [p_L0]])
truemodel["prior"] = truemodel["pi_0"][1][0]

truemodel["guesses"] = np.full(num_gs, p_G, dtype=np.float_)
truemodel["slips"] = np.full(num_gs, p_S, dtype=np.float_)
#can optionally set learn class sequence - set randomly by synthetic_data if not included
#truemodel["resources"] = np.random.randint(1, high = num_resources, size = sum(observation_sequence_lengths))

#data!
print("generating data...")
observation_sequence_lengths = np.full(500, 100, dtype=np.int) #specifies 500 students with 100 observations for synthetic data
data = synthetic_data.synthetic_data(truemodel, observation_sequence_lengths)

#fit models, starting with random initializations
print('fitting! each dot is a new EM initialization')

num_fit_initializations = 5
best_likelihood = float("-inf")

for i in range(num_fit_initializations):
	fitmodel = random_model_uni.random_model_uni(num_learns, num_gs) # include this line to randomly set initial param values
	(fitmodel, log_likelihoods) = EM_fit.EM_fit(fitmodel, data);
	if(log_likelihoods[-1] > best_likelihood):
		best_likelihood = log_likelihoods[-1]
		best_model = fitmodel

# compare the fit model to the true model

print('')
print('\ttruth\tlearned')
print('prior\t%.4f\t%.4f' % (truemodel['prior'], best_model["pi_0"][1][0]))
for r in range(num_learns):
    print('learn%d\t%.4f\t%.4f' % (r+1, truemodel['As'][r, 1, 0].squeeze(), best_model['As'][r, 1, 0].squeeze()))
for r in range(num_learns):
    print('forget%d\t%.4f\t%.4f' % (r+1, truemodel['As'][r, 0, 1].squeeze(), best_model['As'][r, 0, 1].squeeze()))

for s in range(num_gs):
    print('guess%d\t%.4f\t%.4f' % (s+1, truemodel['guesses'][s], best_model['guesses'][s]))
for s in range(num_gs):
    print('slip%d\t%.4f\t%.4f' % (s+1, truemodel['slips'][s], best_model['slips'][s]))
