import numpy as np
from pyBKT.generate import synthetic_data_helper

#TODO: check that parameters are not null, have data, match sizes, etc.
def synthetic_data(model, lengths, resources = None):

    num_resources = len(model["learns"])
    bigT = sum(lengths)

    if resources is None: resources = np.random.randint(1, high = num_resources+1, size = bigT)
    if "As" not in model: model["As"] = np.array([[1-model["learns"], model["forgets"]], [model["learns"], 1-model["forgets"]]])

    if "pi_0" not in model: model["pi_0"] = np.array([[1-model["prior"]], [model["prior"]]]) #TODO: maybe change this to hor array?.

    starts = np.cumsum(lengths)
    starts = np.array([starts[i] - lengths[i] + 1 for i in range(len(starts))])

    syn_data = synthetic_data_helper.create_synthetic_data(model, starts, lengths, resources)
    syn_data["data"] = syn_data["data"] + 1

    syn_data["data"][:, resources != 1] = 0 #no data emitted unless resource == 1

    #fixing data for testing purposes
    # for i in range(bigT):
    #    if i%2 == 0:
    #        resources[i] = 1
    #        syn_data["stateseqs"][0][i] = 0
    #    else:
    #        resources[i] = 2
    #        syn_data["stateseqs"][0][i] = 1
    #    if i%3 == 0:
    #        syn_data["data"][0][i] = 0
    #        syn_data["data"][1][i] = 1
    #        syn_data["data"][2][i] = 2
    #        syn_data["data"][3][i] = 0
    #    elif i%3 == 1:
    #        syn_data["data"][0][i] = 1
    #        syn_data["data"][1][i] = 2
    #        syn_data["data"][2][i] = 0
    #        syn_data["data"][3][i] = 1
    #    else:
    #        syn_data["data"][0][i] = 2
    #        syn_data["data"][1][i] = 0
    #        syn_data["data"][2][i] = 1
    #        syn_data["data"][3][i] = 2


    datastruct = {}
    datastruct["stateseqs"] = syn_data["stateseqs"]
    datastruct["data"] = syn_data["data"]
    datastruct["starts"] = starts
    datastruct["lengths"] = lengths
    datastruct["resources"] = resources

    return(datastruct)
