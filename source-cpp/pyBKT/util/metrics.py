import numpy as np
import sklearn.metrics as sk

SUPPORTED_METRICS = ['accuracy', 'auc', 'rmse']

def error_check(flat_true_values, pred_values):
    if len(flat_true_values) != len(pred_values):
        raise ValueError("preds and true values need to have same shape")

def accuracy(flat_true_values, pred_values):
    error_check(flat_true_values, pred_values)

    correct = 0
    for i in range(len(pred_values)):
        if pred_values[i] >= 0.5 and flat_true_values[i] == 1:
            correct += 1
        if pred_values[i] < 0.5 and flat_true_values[i] == 0:
            correct += 1
    return correct/len([x for x in flat_true_values if (x == 0 or x == 1)])

def auc(flat_true_values, pred_values):
    error_check(flat_true_values, pred_values)
    # multiprior handling, remove phantom nondata
    i = 0
    while i < len(flat_true_values):
        if (flat_true_values[i] != 1 and flat_true_values[i] != 0) or (pred_values[i] < 0 or pred_values[i] > 1):
            flat_true_values = np.delete(flat_true_values, i)
            pred_values = np.delete(pred_values, i)
            i -= 1
        i += 1
    if len(set(flat_true_values)) == 1:
        return np.nan

    auc = sk.roc_auc_score(flat_true_values, pred_values)
    return auc

def rmse(flat_true_values, pred_values):
    # represent correct as 1, incorrect as 0 for RMSE calculation
    if len(flat_true_values) == 0:
        return 0
    error_check(flat_true_values, pred_values)
    rmse, c = 0, 0
    for i in range(len(flat_true_values)):
        if flat_true_values[i] != -1:
            rmse += ((flat_true_values[i] - pred_values[i]) ** 2)
            c += 1
    rmse /= c
    rmse = rmse ** 0.5
    return rmse

