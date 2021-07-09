import numpy as np

def run(model, trans_softcounts, emission_softcounts, init_softcounts, fixed = {}):

    z = np.sum(trans_softcounts, axis=1) == 0
    for i in range(len(z)):
        for j in range(len(z[0])):
            if z[i,j]:
                trans_softcounts[i, 0, j] = 0
                trans_softcounts[i, 1, j] = 1
    # trans_softcounts[np.sum(trans_softcounts, axis=1) == 0,:] = 1
    emission_softcounts[np.sum(emission_softcounts, axis=2) == 0,:] = 1
    assert (trans_softcounts.shape[1] == 2)
    assert (trans_softcounts.shape[2] == 2)

    temp = np.sum(trans_softcounts, axis=1)

    #model['As'] = trans_softcounts / np.sum(trans_softcounts, axis=1)
    #model['As'] = np.divide(trans_softcounts, np.sum(trans_softcounts, axis=1))


    if 'learn' in fixed and 'forget' in fixed:
        model['As'] = np.empty((len(fixed['learn']), 2, 2))
        for i in range(len(fixed['learn'])):
            model['As'][i] = np.array([[1 - fixed['learn'][i], fixed['forget'][i]], [fixed['learn'][i], 1 - fixed['forget'][i]]])
        model['learns'] = fixed['learn']
        model['forgets'] = fixed['forget']
    else:
        model['As'][:model['As'].shape[0]] = (trans_softcounts / np.sum(trans_softcounts, axis=1)[:model['As'].shape[0], None])
        model['learns'] = model['As'][:, 1, 0]
        model['forgets'] = model['As'][:, 0, 1]

    temp = np.sum(emission_softcounts, axis=2)

    #model['emissions'] = emission_softcounts / np.sum(emission_softcounts, axis=1)
    model['emissions'] = emission_softcounts / temp[:, :, None]
    #the expand dims is very weird.
    #model['guesses'] = np.expand_dims(model['emissions'][:, 0, 1].squeeze(), axis=0)
    #model['slips'] = np.expand_dims(model['emissions'][:, 1, 0].squeeze(), axis=0)
    
    if 'guess' in fixed and 'slip' in fixed:
        model['guesses'] = fixed['guess']
        model['slips'] = fixed['slip']
    else:
        model['guesses'] = model['emissions'][:, 0, 1]
        model['slips'] = model['emissions'][:, 1, 0]

    if 'prior' in fixed:
        model['pi_0'] = np.array([[1 - fixed['prior']], [fixed['prior']]])
        model['prior'] = model['pi_0'][1][0]
    else:
        temp = np.sum(init_softcounts[:])
        model['pi_0'] = init_softcounts[:] / np.sum(init_softcounts[:])
        model['prior'] = model['pi_0'][1][0]

    return(model)
