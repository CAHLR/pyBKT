import numpy as np
from pyBKT.fit import E_step
from pyBKT.fit import predict_onestep_states

# correct_emission_predictions is a  num_subparts x T array, where element
# (i,t) is predicted probability that answer to subpart i at time t+1 is correct
def run(model, data):

    num_subparts = data["data"].shape[0]  # mmm the first dimension of data represents each subpart?? interesting.
    num_resources = len(model["learns"])

    trans_softcounts = np.zeros((num_resources, 2, 2))
    emission_softcounts = np.zeros((num_subparts, 2, 2))
    init_softcounts = np.zeros((2, 1))

    result = {}
    result['all_trans_softcounts'] = trans_softcounts
    result['all_emission_softcounts'] = emission_softcounts
    result['all_initial_softcounts'] = init_softcounts

    result = E_step.run(data, model, result['all_trans_softcounts'], result['all_emission_softcounts'], result['all_initial_softcounts'], 1)
    for j in range(num_resources):
        result['all_trans_softcounts'][j] = result['all_trans_softcounts'][j].transpose()
    for j in range(num_subparts):
        result['all_emission_softcounts'][j] = result['all_emission_softcounts'][j].transpose()

    state_predictions = predict_onestep_states.run(data, model, result['alpha'])

    correct_emission_predictions = model["guesses"]*state_predictions[0,:] + (1-model["slips"])*state_predictions[1,:]

    return (correct_emission_predictions, state_predictions)