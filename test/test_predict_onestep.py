import numpy as np
from pyBKT.generate import synthetic_data
from pyBKT.fit import predict_onestep

#parameters
num_subparts = 1
num_resources = 2
num_fit_initializations = 10
observation_sequence_lengths = np.full(500, 100, dtype=np.int)

#generate synthetic model and data.
#model is really easy.
truemodel = {}

truemodel["As"] =  np.zeros((2, 2, num_resources), dtype=np.float_)
for i in range(num_resources):
    truemodel["As"][i, :, :] = np.transpose([[0.7, 0.3], [0.01, 0.99]])
truemodel["learns"] = truemodel["As"][:, 1, 0]
truemodel["forgets"] = truemodel["As"][:, 0, 1]

truemodel["pi_0"] = np.array([[0.9], [0.1]])
truemodel["prior"] = truemodel["pi_0"][1][0]

truemodel["guesses"] = np.full(num_subparts, 0.1, dtype=np.float_)
truemodel["slips"] = np.full(num_subparts, 0.03, dtype=np.float_)

#data!
print("generating data...")
data = synthetic_data.synthetic_data(truemodel, observation_sequence_lengths)

(correct_predictions, state_predictions) = predict_onestep.run(truemodel, data)

print("finishing...")