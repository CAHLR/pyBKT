#########################################
# data_helper.py                        #
# data_helper                           #
#                                       #
# @author Frederic Wang                 #
# Last edited: 27 March 2021            #
#########################################

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

def fix_data_specified(data, label, count):
    training_data = {}
    testing_data = {}
    
    train_starts = np.zeros(len(data["starts"]), dtype = 'int')
    train_lengths = np.zeros(len(data["lengths"]), dtype = 'int')
    train_resources = np.zeros(len(data["resources"]) - count, dtype = 'int')
    train_data = np.zeros((len(data["data"]), len(data["resources"]) - count), dtype = 'int')
    
    test_starts = np.zeros(len(data["starts"]), dtype = 'int')
    test_lengths = np.zeros(len(data["lengths"]), dtype = 'int')
    test_resources = np.zeros(count, dtype = 'int')
    test_data = np.zeros((len(data["data"]), count), dtype = 'int')
     
    train_idx = 1
    test_idx = 1
    for i in range(len(data["starts"])):
        train_starts[i] = train_idx
        test_starts[i] = test_idx
        for j in range(data["lengths"][i]):
            current_idx = data["starts"][i] + j - 1
            if data["folds"][current_idx] != label:
                save_idx = train_starts[i]+train_lengths[i]-1
                train_lengths[i] += 1
                train_data[:,save_idx] = data["data"][:,current_idx]
                train_resources[save_idx] = data["resources"][current_idx]
            else:
                save_idx = test_starts[i]+test_lengths[i]-1
                test_lengths[i] += 1
                test_data[:,save_idx] = data["data"][:,current_idx]
                test_resources[save_idx] = data["resources"][current_idx]
        train_idx += train_lengths[i]
        test_idx += test_lengths[i]
        
    real_train = np.nonzero(train_lengths)
    train_lengths = train_lengths[real_train]
    train_starts = train_starts[real_train]

    real_test = np.nonzero(test_lengths)
    test_lengths = test_lengths[real_test]
    test_starts = test_starts[real_test]
    
    training_data["starts"] = train_starts
    training_data["lengths"] = train_lengths
    training_data["data"] = train_data
    training_data["resources"] = train_resources
    
    if "resource_names" in data:
        training_data["resource_names"] = data["resource_names"]
        testing_data["resource_names"] = data["resource_names"]
    if "gs_names" in data:
        training_data["gs_names"] = data["gs_names"]
        testing_data["gs_names"] = data["gs_names"]
    
    testing_data["starts"] = test_starts
    testing_data["lengths"] = test_lengths
    testing_data["data"] = test_data
    testing_data["resources"] = test_resources
    
    
    
    return training_data, testing_data

def crossvalidate(model, data, skill, folds, metric, seed, use_folds=False):

    num_learns = len(data["resource_names"]) if "resource_names" in data else 1
    num_gs = len(data["gs_names"]) if "gs_names" in data else num_gs

    # create random permutation to act as indices for folds for crossvalidation
    shuffle = np.random.RandomState(seed=seed).permutation(len(data["starts"]))
    all_true, all_pred = [], []
    metrics = np.zeros((len(metric), ))

    
    if use_folds: # predetermined folds
        all_labels, all_counts = np.unique(data["folds"], return_counts=True)
        all_folds = dict(zip(all_labels, all_counts))
        folds = len(all_folds)
        
        for label, count in all_folds.items():
            training_data, test_data = fix_data_specified(data, label, count)
            model.fit_model[skill] = model._fit(training_data, skill, model.forgets)
            metrics += model._evaluate({skill: test_data}, metric)
            
    else: # crossvalidation on students which are identified by the starts array
        split_size = len(data["starts"]) // folds
        for iteration in range(folds):
            model.fit_model = {}
            # create training/test data based on random permutation from earlier
            train = np.concatenate((shuffle[0: iteration * split_size], shuffle[(iteration + 1) * split_size:
                                                                                len(data["starts"])]))
            training_data = fix_data(data, train)
            model.fit_model[skill] = model._fit(training_data, skill, model.forgets)

            test = shuffle[iteration*split_size:(iteration+1)*split_size]
            test_data = fix_data(data, test)
            metrics += model._evaluate({skill: test_data}, metric)

    return metrics / folds
