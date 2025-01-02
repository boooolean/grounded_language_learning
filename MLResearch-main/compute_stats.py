import numpy as np
import pandas as pd
import math

def compute_kappa(confusion_matrix):
    tp = confusion_matrix[0][0]
    tn = confusion_matrix[1][1]
    fp = confusion_matrix[1][0]
    fn = confusion_matrix[0][1]
    print(tp, tn, fp, fn)
    exp_accuracy = ((tp + fn) * ((tp + fp) / confusion_matrix.sum()) + (tn +fp) * ((tn + fn) / confusion_matrix.sum())) / confusion_matrix.sum()
    obs_accuracy = (tp + tn) / (confusion_matrix.sum())
    kappa_c = (obs_accuracy - exp_accuracy) / (1 - exp_accuracy)
    return kappa_c

def compute_acc(confusion_matrix):
    tp = confusion_matrix[0][0]
    tn = confusion_matrix[1][1]
    total_accuracy = (tp + tn) / (confusion_matrix.sum())
    return total_accuracy

def compute_precision(confusion_matrix):
    tp = confusion_matrix[0][0]
    fp = confusion_matrix[1][0]
    precision = tp / (tp + fp)
    return precision

def compute_recall(confusion_matrix):
    tp = confusion_matrix[0][0]
    fn = confusion_matrix[0][1]
    recall = tp / (tp + fn)
    return recall

def compute_F1(confusion_matrix):
    tp = confusion_matrix[0][0]
    fp = confusion_matrix[1][0]
    fn = confusion_matrix[0][1]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    return f1_score

