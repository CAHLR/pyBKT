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

    model['As'][:model['As'].shape[0]] = (trans_softcounts / np.sum(trans_softcounts, axis=1)[:model['As'].shape[0], None])
    
    model['learns'] = model['As'][:, 1, 0]
    if 'learns' in fixed:
        model['learns'] = model['As'][:, 1, 0] * (fixed['learns'] < 0) + fixed['learns'] * (fixed['learns'] >= 0)
        for i in range(len(model['As'])):
            if fixed['learns'][i] >= 0:
                model['As'][i, 1, 0] =  fixed['learns'][i]
                model['As'][i, 0, 0] =  1 - fixed['learns'][i]
    
    model['forgets'] = model['As'][:, 0, 1]
    if 'forgets' in fixed:
        model['forgets'] = model['As'][:, 0, 1] * (fixed['forgets'] < 0) + fixed['forgets'] * (fixed['forgets'] >= 0)
        for i in range(len(model['As'])):
            if fixed['forgets'][i] >= 0:
                model['As'][i, 0, 1] =  fixed['forgets'][i]
                model['As'][i, 1, 1] =  1 - fixed['forgets'][i]

    temp = np.sum(emission_softcounts, axis=2)

    #model['emissions'] = emission_softcounts / np.sum(emission_softcounts, axis=1)
    model['emissions'] = emission_softcounts / temp[:, :, None]
    #the expand dims is very weird.
    #model['guesses'] = np.expand_dims(model['emissions'][:, 0, 1].squeeze(), axis=0)
    #model['slips'] = np.expand_dims(model['emissions'][:, 1, 0].squeeze(), axis=0)
    
    model['guesses'] = model['emissions'][:, 0, 1]
    if 'guesses' in fixed:
        model['guesses'] = model['guesses'] * (fixed['guesses'] < 0) + fixed['guesses'] * (fixed['guesses'] >= 0)
        
    model['slips'] = model['emissions'][:, 1, 0]
    if 'slips' in fixed:
        model['slips'] = model['slips'] * (fixed['slips'] < 0) + fixed['slips'] * (fixed['slips'] >= 0)
        

    if 'prior' in fixed:
        model['pi_0'] = np.array([[1 - fixed['prior']], [fixed['prior']]])
        model['prior'] = model['pi_0'][1][0]
    else:
        temp = np.sum(init_softcounts[:])
        model['pi_0'] = init_softcounts[:] / np.sum(init_softcounts[:])
        model['prior'] = model['pi_0'][1][0]

    return(model)
