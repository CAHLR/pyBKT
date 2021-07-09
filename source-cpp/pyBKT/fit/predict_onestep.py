import numpy as np
from pyBKT.fit import E_step
from pyBKT.fit import predict_onestep_states

# correct_emission_predictions is a  num_subparts x T array, where element
# (i,t) is predicted probability that answer to subpart i at time t+1 is correct
def run(model, data, parallel = True, fixed = {}):

    num_subparts = data["data"].shape[0]  # mmm the first dimension of data represents each subpart?? interesting.
    num_resources = len(model["learns"])

    result = E_step.run(data, model, 1, int(parallel), fixed)
    for j in range(num_resources):
        result['all_trans_softcounts'][j] = result['all_trans_softcounts'][j].transpose()
    for j in range(num_subparts):
        result['all_emission_softcounts'][j] = result['all_emission_softcounts'][j].transpose()

    state_predictions = predict_onestep_states.run(data, model, result['alpha'], int(parallel))
    p = state_predictions.shape
    state_predictions = state_predictions.flatten(order = 'C').reshape(p, order = 'F')
    # multiguess solution, should work
    correct_emission_predictions = np.expand_dims(model["guesses"], axis = 1) @ np.expand_dims(state_predictions[0,:], axis = 0) + np.expand_dims(1-model["slips"], axis = 1) @ np.expand_dims(state_predictions[1,:], axis = 0)
    #correct_emission_predictions = model['guesses'] * np.asarray([state_predictions[0,:]]).T + (1 - model['slips']) * np.asarray([state_predictions[1,:]]).T
    flattened_predictions = np.take_along_axis(correct_emission_predictions, (data['data'] != 0).argmax(axis = 0)[:, None].T, axis = 0)
    return (flattened_predictions.ravel(), state_predictions)

