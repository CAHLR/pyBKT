import numpy as np
import re
import sklearn.metrics as sk

def error_check(flat_true_values, pred_values):
    if len(flat_true_values) != len(pred_values):
        raise ValueError("preds and true values need to have same shape")

def accuracy(flat_true_values, pred_values):
    error_check(flat_true_values, pred_values)
    if len(flat_true_values) == 0:
        return np.nan
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
    if len(flat_true_values) == 0:
        return np.nan
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
        return np.nan
    error_check(flat_true_values, pred_values)
    rmse, c = 0, 0
    for i in range(len(flat_true_values)):
        if flat_true_values[i] != -1:
            rmse += ((flat_true_values[i] - pred_values[i]) ** 2)
            c += 1
    rmse /= c
    rmse = rmse ** 0.5
    return rmse

def fetch_supported_metrics():
    supported_metrics = {}
    dummy_x, dummy_y = [0, 1] * 5, [1, 0] * 5
    for metric_locs in sk._regression, sk._classification:
        potential_metrics = {i: getattr(metric_locs, i) for i in dir(metric_locs) if re.search('_loss$|_score$|_error$', i)}
        for metric in potential_metrics:
            try:
                potential_metrics[metric](dummy_x, dummy_y)
                supported_metrics[metric] = potential_metrics[metric]
            except TypeError:
                pass
    return supported_metrics


SUPPORTED_METRICS = {'accuracy': accuracy, 'auc': auc, 'rmse': rmse}
SUPPORTED_METRICS.update(fetch_supported_metrics())
