import numpy as np

def dirrnd(alphavec):
    dimsum = 1
    if(alphavec.ndim == 3 or alphavec.ndim == 2): dimsum = alphavec.ndim - 2
    a = np.random.gamma(alphavec, 1)
    temp = np.sum(a, axis=dimsum, keepdims=True)
    a = a / np.sum(a, axis=dimsum, keepdims=True)
    return(a)