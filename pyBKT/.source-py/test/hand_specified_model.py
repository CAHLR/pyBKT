import numpy as np
from pyBKT.generate import synthetic_data
from pyBKT.generate import random_model, random_model_uni
from pyBKT.fit import EM_fit
from copy import deepcopy
from pyBKT.util import print_dot

#parameters
num_subparts = 4
num_resources = 2
num_fit_initializations = 25
observation_sequence_lengths = np.full(50, 100, dtype=np.int)

#generate synthetic model and data.
#model is really easy.
truemodel = {}

truemodel["As"] = np.zeros((num_resources, 2, 2), dtype=np.float_)
truemodel["As"][0, :, :] = np.transpose([[0.75, 0.25], [0.1, 0.9]])
truemodel["As"][1, :, :] = np.transpose([[0.9, 0.1], [0.1, 0.9]])
truemodel["learns"] = truemodel["As"][:, 1, 0]
truemodel["forgets"] = truemodel["As"][:, 0, 1]

truemodel["pi_0"] = np.array([[0.9], [0.1]]) #TODO: one prior per resource? does this array needs to be col?
truemodel["prior"] = 0.1

truemodel["guesses"] = np.full(num_subparts, 0.05, dtype=np.float_)
truemodel["slips"] = np.full(num_subparts, 0.25, dtype=np.float_)

truemodel["resources"] = np.random.randint(1, high = num_resources+1, size = sum(observation_sequence_lengths))

#data!
print("generating data...")
data = synthetic_data.synthetic_data(truemodel, observation_sequence_lengths)

#fit models, starting with random initializations
print('fitting! each dot is a new EM initialization')

best_likelihood = float("-inf")
for i in range(num_fit_initializations):
    print_dot.print_dot(i, num_fit_initializations)
    fitmodel = random_model.random_model(num_resources, num_subparts)
    (fitmodel, log_likelihoods) = EM_fit.EM_fit(fitmodel, data)
    if (log_likelihoods[-1] > best_likelihood):
        best_likelihood = log_likelihoods[-1]
        best_model = fitmodel

# compare the fit model to the true model
print('')

print('these two should look similar')
print(truemodel['As'])
print('')
print(best_model['As'])

print('')
print('these should look similar too')
print(1-truemodel['guesses'])
print('')
print(1-best_model['guesses'])

print('')
print('these should look similar too')
print(1-truemodel['slips'])
print('')
print(1-best_model['slips'])