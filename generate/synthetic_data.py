import numpy as np
from generate import synthetic_data_helper

#TODO: check that parameters are not null, have data, match sizes, etc.
def synthetic_data(model, lengths, resources = None):
    num_resources = model["learns"].shape[0]
    bigT = sum(lengths)

    if resources is None: resources = np.random.randint(1, high = num_resources+1, size = bigT) #used to be 1xbigT
    if "As" not in model: model["As"] = np.array([[1-model["learns"][0], model["forgets"][0]], [model["learns"], 1-model["forgets"]]]) #TODO: is not considering more that one resource for this. fix it.

    if "pi_0" not in model: model["pi_0"] = np.array([1-model["prior"], model["prior"]])

    #col array from 1 to 49901 on steps of 100
    starts = np.cumsum(lengths)
    starts = np.array([starts[i] - lengths[i] + 1 for i in range(len(starts))])

    #print(model)
    #print(starts)
    #print(lengths)
    #print(resources)
    syn_data = synthetic_data_helper.create_synthetic_data(model, starts, lengths, resources)
    #print(syn_data["data"][0])
    syn_data["data"] = syn_data["data"] + 1

    syn_data["data"][:, resources != 1] = 0 #no data emitted unless resource == 1

    datastruct = {}
    datastruct["stateseqs"] = syn_data["stateseqs"]
    datastruct["data"] = syn_data["data"]
    datastruct["starts"] = starts
    datastruct["lengths"] = lengths
    datastruct["resources"] = resources

    return(datastruct)