import sys
sys.path.append('../')
import numpy as np

def compute_rmse(flat_true_values, pred_values):
    # represent correct as 1, incorrect as 0 for RMSE calculation
    flat_true_values = [x-1 for x in flat_true_values]
    rmse = 0

    for i in range(len(flat_true_values)):
        if flat_true_values[i] != -1:
            rmse += ((flat_true_values[i] - pred_values[i]) ** 2)
    rmse /= len([x for x in flat_true_values if (x == 0 or x == 1)])
    rmse = rmse ** 0.5
    return rmse
