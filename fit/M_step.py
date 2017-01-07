import numpy as np

def run(model, trans_softcounts, emission_softcounts, init_softcounts):
    # print('trans_softcounts:')
    # print(trans_softcounts)
    # print('emission_softcounts:')
    # print(emission_softcounts)
    # print('init_softcounts:')
    # print(init_softcounts)

    #is this right??
    trans_softcounts[:, np.sum(trans_softcounts, axis=0) == 0] = 1
    #this should be on axis 2??
    emission_softcounts[:, np.sum(emission_softcounts, axis=1) == 0] = 1
    assert (trans_softcounts.shape[0] == 2)
    assert (trans_softcounts.shape[1] == 2)

    model['As'] = trans_softcounts / np.sum(trans_softcounts, axis=0)
    model['learns'] = model['As'][1, 0, :]
    model['forgets'] = model['As'][0, 1, :]

    model['emissions'] = emission_softcounts / np.sum(emission_softcounts, axis=1)
    #the expand dims is very weird.
    model['guesses'] = np.expand_dims(model['emissions'][0, 1, :].squeeze(), axis=0)
    model['slips'] = np.expand_dims(model['emissions'][1, 0, :].squeeze(), axis=0)

    #this is wrong, shouldn't be zero
    if(np.sum(init_softcounts[:]) != 0):
        model['pi_0'] = init_softcounts[:] / np.sum(init_softcounts[:])
    model['prior'] = model['pi_0'][1]

    return(model)