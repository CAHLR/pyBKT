import numpy as np
from pyBKT.generate import synthetic_data, random_model_uni
from pyBKT.fit import EM_fit, predict_onestep
from copy import deepcopy

# returns data only for the indices given based on starts array
def fix_data(data, indices):
    training_data = {}
    prev_starts = data["starts"][indices]
    lengths = data["lengths"][indices]
    total_length = np.sum(lengths)
    d = np.zeros((len(data["data"]), total_length), dtype=np.int32)
    resources = np.ones(total_length, dtype=np.int64)
    
    if "resource_names" in data:
        training_data["resource_names"] = data["resource_names"]
    if "gs_names" in data:
        training_data["gs_names"] = data["gs_names"]
    
    starts = np.zeros(len(prev_starts), dtype=np.int64)
    current_index = 1
    for i in range(len(prev_starts)):
        starts[i] = current_index

        d[:,starts[i]-1:starts[i]+lengths[i]-1] = data["data"][:,prev_starts[i]-1:prev_starts[i]+lengths[i]-1]
        resources[starts[i]-1:starts[i]+lengths[i]-1] = data["resources"][prev_starts[i]-1:prev_starts[i]+lengths[i]-1]
        current_index += lengths[i]
    
    training_data["starts"] = starts
    training_data["lengths"] = lengths
    training_data["data"] = d
    training_data["resources"]=resources
    return (training_data)

def crossvalidate(model, data, skill, folds, metric, seed):

    num_learns = len(data["resource_names"]) if "resource_names" in data else 1
    num_gs = len(data["gs_names"]) if "gs_names" in data else num_gs
    split_size = len(data["starts"]) // folds

    # create random permutation to act as indices for folds for crossvalidation
    shuffle = np.random.RandomState(seed=seed).permutation(len(data["starts"]))
    all_true, all_pred = [], []
    metrics = []

    # crossvalidation on students which are identified by the starts array
    for iteration in range(folds):
        model.fit_model = {}
        # create training/test data based on random permutation from earlier
        train = np.concatenate((shuffle[0: iteration * split_size], shuffle[(iteration + 1) * split_size:
                                                                            len(data["starts"])]))
        training_data = fix_data(data, train)
        model.fit_model[skill] = model._fit(training_data, skill, model.forgets)

        test = shuffle[iteration*split_size:(iteration+1)*split_size]
        test_data = fix_data(data, test)
        metrics.append(model._evaluate({skill: test_data}, metric))

    return np.mean(metrics)
