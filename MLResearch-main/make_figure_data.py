import pickle

from scipy.sparse.construct import rand
import numpy as np
import random

from construct_dataset import get_feature_vec
from compute_stats import compute_kappa
from oracle import Oracle

def make_multi_context_data(w, obj_test, context_set, n, num_contexts, multi_context_kappas, all_data, confusion_matrices, oracle, train_test_count):
    cm = np.zeros([2,2])
    for o in obj_test:
        for t in range(1,6): # For each trial
            # Class distribution initial belief
            class_dist = [0.0, 0.0]
            cm_index_total = 0
            index, = np.where(obj_test == o)
            random.seed((train_test_count + t) * (t + train_test_count + n + index[0]))
            rand_indices = random.sample(range(0, len(context_set)), n)
            for c in context_set:
                if c in context_set[rand_indices]:
                    # Get feature vector for trial t for object o for context c
                    x_c = get_feature_vec(all_data, c, o, t).reshape(1, -1)
                    if x_c.size != 0:
                        # Get distribution prediction from classifier for trial t
                        clf = pickle.load(open("/Users/zachosman1/Desktop/Summer 2021/jivkolab/classifiers/" 
                                        + str(w) + "_" + str(c) + "_" + str(t - 1) + ".pickle", "rb"))
                        dist_c = clf.predict_proba(x_c)
                        class_dist[0] += dist_c[0, 0]
                        class_dist[1] += dist_c[0, 1]
                        weight_c = compute_kappa(confusion_matrices[cm_index_total]) # For context and word
                        class_dist[0] += weight_c * dist_c[0, 0]
                        class_dist[1] += weight_c * dist_c[0, 1]
                cm_index_total += 1
            # Normalize - divide class_dist[0] and class_dist[1] by sum
            if sum(class_dist) != 0:
                normalized_class_dist = class_dist / sum(class_dist)
            # Determine predicted class label - which is larger between class_dist[0] and class_dist[1]
            y_hat = 1
            if normalized_class_dist[0] >= normalized_class_dist[1]:
                y_hat = 0
            # Compare to ground truth and update cm
            ground_truth = oracle.get_object_word(o, w)

            if y_hat == 1 and ground_truth == 1: # True Positive
                cm[0][0] += 1
            elif y_hat == 0 and ground_truth == 0: # True Negative
                cm[1][1] += 1
            elif y_hat == 0 and ground_truth == 1: # False Negative
                cm[0][1] += 1
            elif y_hat == 1 and ground_truth == 0: # False Positive
                cm[1][0] += 1
    multi_context_kappas[num_contexts.index(n)] += compute_kappa(cm)
