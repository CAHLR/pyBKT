import sys
sys.path.append('../')
import numpy as np

def compute_acc(flat_true_values, pred_values):

    # represent correct as 1, incorrect as 0 for RMSE calculation
    flat_true_values = [x-1 for x in flat_true_values]

    correct = 0
    for i in range(len(pred_values)):
        if pred_values[i] >= 0.5 and flat_true_values[i] == 1:
            correct += 1
        if pred_values[i] < 0.5 and flat_true_values[i] == 0:
            correct += 1
    return correct/len([x for x in flat_true_values if (x == 0 or x == 1)])
    
