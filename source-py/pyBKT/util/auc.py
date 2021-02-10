import sys
sys.path.append('../')
import numpy as np
import sklearn.metrics as sk

def compute_auc(flat_true_values, pred_values):
    # represent correct as 1, incorrect as 0 for RMSE calculation
    flat_true_values = [x-1 for x in flat_true_values]
   # print(flat_true_values)
    #print(pred_values)
    #multiprior handling, remove phantom nondata
    i = 0
    while i < len(flat_true_values):
        if (flat_true_values[i] != 1 and flat_true_values[i] != 0) or (pred_values[i] < 0 or pred_values[i] > 1):
            flat_true_values.pop(i)
            pred_values = np.delete(pred_values, i)
            i -= 1
        i += 1
    

    auc = sk.roc_auc_score(flat_true_values, pred_values)
    return auc
