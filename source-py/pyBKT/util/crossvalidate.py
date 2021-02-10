import numpy as np
from pyBKT.generate import synthetic_data, random_model_uni
from pyBKT.fit import EM_fit, predict_onestep
from copy import deepcopy

# returns data only for the indices given based on starts array
def fix_data(data, indices):
    training_data = {}
    resources = []
    d = [[] for _ in range(len(data["data"]))]
    start_temp = [data["starts"][i] for i in indices]
    length_temp = [data["lengths"][i] for i in indices]
    if "resource_names" in data:
        training_data["resource_names"] = data["resource_names"]
    if "gs_names" in data:
        training_data["gs_names"] = data["gs_names"]
    starts = []
    for i in range(len(start_temp)):
        starts.append(len(resources)+1)
        #print("A", start_temp[i], start_temp[i]+length_temp[i])
        resources.extend(data["resources"][start_temp[i]-1:start_temp[i]+length_temp[i]-1])
        for j in range(len(data["data"])):
            d[j].extend(data["data"][j][start_temp[i]-1:start_temp[i]+length_temp[i]-1])
    training_data["starts"] = np.asarray(starts)
    training_data["lengths"] = np.asarray(length_temp)
    training_data["data"] = np.asarray(d,dtype='int32')
    resource=np.asarray(resources)
    stateseqs=np.copy(resource)
    training_data["stateseqs"]=np.asarray([stateseqs],dtype='int32')
    training_data["resources"]=resource
    training_data=(training_data)
    return training_data

def crossvalidate(model, data, skill, folds, metric, seed):

    num_learns = len(data["resource_names"]) if "resource_names" in data else 1
    num_gs = len(data["gs_names"]) if "gs_names" in data else num_gs
    split_size = len(data["starts"]) // folds

    # create random permutation to act as indices for folds for crossvalidation
    shuffle = np.random.RandomState(seed=seed).permutation(len(data["starts"]))
    all_true, all_pred = [], []
    metrics = []
    model.fit_model = {}

    # crossvalidation on students which are identified by the starts array
    for iteration in range(folds):
        # create training/test data based on random permutation from earlier
        train = np.concatenate((shuffle[0: iteration * split_size], shuffle[(iteration + 1) * split_size:
                                                                            len(data["starts"])]))
        training_data = fix_data(data, train)
        model.fit_model[skill] = model._fit(training_data, skill, model.forgets)

        test = shuffle[iteration*split_size:(iteration+1)*split_size]
        test_data = fix_data(data, test)
        metrics.append(model._evaluate({skill: test_data}, metric))

    return metrics
