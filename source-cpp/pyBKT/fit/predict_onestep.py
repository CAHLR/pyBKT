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
    p = state_predictions.shape
    state_predictions = state_predictions.flatten(order = 'C').reshape(p, order = 'F')
    # multiguess solution, should work
    correct_emission_predictions = np.expand_dims(model["guesses"], axis = 1) @ np.expand_dims(state_predictions[0,:], axis = 0) + np.expand_dims(1-model["slips"], axis = 1) @ np.expand_dims(state_predictions[1,:], axis = 0)
    #correct_emission_predictions = model['guesses'] * np.asarray([state_predictions[0,:]]).T + (1 - model['slips']) * np.asarray([state_predictions[1,:]]).T
    flattened_predictions = np.zeros((len(correct_emission_predictions[0]),))
    for i in range(len(correct_emission_predictions)):
        for j in range(len(correct_emission_predictions[0])):
            if data["data"][i][j] != 0:
                flattened_predictions[j] = correct_emission_predictions[i][j]
    return (flattened_predictions, state_predictions)

