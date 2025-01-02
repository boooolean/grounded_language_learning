from typing import IO
from numpy.lib.nanfunctions import _nanquantile_dispatcher
from scipy.sparse.construct import rand
import csv
import pickle
import numpy as np
import random
import pandas as pd
import os
from os import listdir
from sklearn.model_selection import KFold
from sklearn import svm
import sys

from oracle import Oracle
from construct_dataset import construct_dataset
from construct_dataset import read_all_data
from construct_dataset import get_feature_vec
from compute_stats import compute_kappa
from compute_stats import compute_acc
from compute_stats import compute_F1
from compute_stats import compute_precision
from compute_stats import compute_recall
from upsample import upsample_data
from make_figure_data import make_multi_context_data

oracle = Oracle()

# Make set of words, objects, and behavior-modality combinations
word_set = np.array(oracle.ground_truth.columns)
object_set = np.array(oracle.ground_truth.index)
context_set = os.listdir('/Users/zachosman1/Desktop/Summer 2021/jivkolab/contexts')
context_set.remove('.DS_Store')
context_set = np.array([name.removesuffix('.txt') for name in context_set])
num_contexts = [1, 2, 5, int(.2 * len(context_set)), 
                    int(.4 * len(context_set)), int(.6 * len(context_set)), 
                    int(.8 * len(context_set))]
# Make data structure that has all data points for all contexts
all_data = read_all_data(context_set)
# TODO - maybe try filtering out words where too few objects apply - 4 or 5

for w in word_set:
    print(w)
    # For each word-context combination, generate a confusion matrix and store it in here
    confusion_matrices = []
    for i in range(len(context_set)):
        empty_mat = np.zeros([2,2])
        confusion_matrices.append(empty_mat)
    confusion_matrices = np.asarray(confusion_matrices)

    cm_combined = np.zeros([2,2])
    multi_context_kappas = np.zeros(len(num_contexts))

    # Create folds for train-test splits
    kf = KFold(n_splits=5, shuffle=True, random_state = 257)
    train_test_count = 0
    for train_index, test_index in kf.split(object_set):
        obj_train, obj_test = object_set[train_index], object_set[test_index]
        cm_index_indiv = 0
        for c in context_set:
            print(c)
            # If classifier is saved, unpickle; if it doesn't exist, create and train then pickle for later
            try:
                clf = pickle.load(open("/Users/zachosman1/Desktop/Summer 2021/jivkolab/classifiers/" 
                                       + str(w) + "_" + str(c) + "_" + str(train_test_count) + ".pickle", "rb"))
            except (OSError, IOError) as e:
                # Use Oracle and data from file to generate x and y data for word-context combination
                [X_train, y_train] = construct_dataset(all_data, obj_train, w, c, oracle)

                # Upsample - make number of positives and negatives equal
                [X_train, y_train] = upsample_data(X_train, y_train)

                # Train classifier
                clf = svm.SVC(gamma=0.001, C=10, probability = True) # TODO - Also try decision tree, k-nearest neighbor
                clf.fit(X_train, y_train)
                pickle.dump(clf, open("/Users/zachosman1/Desktop/Summer 2021/jivkolab/classifiers/" 
                                      + str(w) + "_" + str(c) + "_" + str(train_test_count) + ".pickle", "wb"))

            # Construct test data set to test accuracy of our model
            [X_test, y_test] = construct_dataset(all_data, obj_test, w, c, oracle)

            # Increment confusion matrix values
            y_hat = clf.predict(X_test)
            for true, pred in zip(y_test, y_hat):
                if true == 1 and pred == 1: # True Positive
                    (confusion_matrices[cm_index_indiv])[0][0] += 1
                elif true == 0 and pred == 0: # True Negative
                    (confusion_matrices[cm_index_indiv])[1][1] += 1
                elif true == 1 and pred == 0: # False Negative
                    (confusion_matrices[cm_index_indiv])[0][1] += 1
                elif true == 0 and pred == 1: # False Positive
                    (confusion_matrices[cm_index_indiv])[1][0] += 1
            cm_index_indiv += 1 

        # print(confusion_matrices)
        try:
            multi_context_kappas = pickle.load(open("/Users/zachosman1/Desktop/Summer 2021/jivkolab/kappas/" 
                                      + str(w) + "_" + "kappas" + ".pickle", "rb"))
        except (OSError, IOError) as e:
            for i in range(0, len(num_contexts)):
                make_multi_context_data(w, obj_test, context_set, num_contexts[i], num_contexts, multi_context_kappas, all_data, confusion_matrices, oracle, train_test_count)
                print("Kappas: ", multi_context_kappas)
            if train_test_count == 4:
                multi_context_kappas = np.divide(multi_context_kappas, 5)
                print("Multi Context Kappas Final: ", multi_context_kappas)
        try:
            cm_combined = pickle.load(open("/Users/zachosman1/Desktop/Summer 2021/jivkolab/confusion_matrices/" 
                                       + str(w) + "_" + "cm_all" + ".pickle", "rb"))
        except (OSError, IOError) as e:
            # Test recognition accuracy with all classifiers combined
            for o in obj_test:
                # for t in range(1,6): # For each trial
                #     # Class distribution initial belief
                #     class_dist = [0.0, 0.0]
                #     cm_index_total = 0
                #     for c in context_set:
                #         # Get feature vector for trial t for object o for context c
                #         x_c = get_feature_vec(all_data, c, o, t).reshape(1, -1)
                #         if x_c.size != 0:
                #             # Get distribution prediction from classifier for trial t
                #             clf = pickle.load(open("/Users/zachosman1/Desktop/Summer 2021/jivkolab/classifiers/" 
                #                             + str(w) + "_" + str(c) + "_" + str(t - 1) + ".pickle", "rb"))
                #             dist_c = clf.predict_proba(x_c)
                #             class_dist[0] += dist_c[0, 0]
                #             class_dist[1] += dist_c[0, 1]
                #             weight_c = compute_kappa(confusion_matrices[cm_index_total]) # For context and word
                #             class_dist[0] += weight_c * dist_c[0, 0]
                #             class_dist[1] += weight_c * dist_c[0, 1]
                #         cm_index_total += 1
                #     # Normalize - divide class_dist[0] and class_dist[1] by sum
                #     normalized_class_dist = class_dist / sum(class_dist)
                #     # Determine predicted class label - which is larger between class_dist[0] and class_dist[1]
                #     y_hat = 1
                #     if normalized_class_dist[0] >= normalized_class_dist[1]:
                #         y_hat = 0
                #     # Compare to ground truth and update cm_combined
                #     ground_truth = oracle.get_object_word(o, w)

                #     if y_hat == 1 and ground_truth == 1: # True Positive
                #         cm_combined[0][0] += 1
                #     elif y_hat == 0 and ground_truth == 0: # True Negative
                #         cm_combined[1][1] += 1
                #     elif y_hat == 0 and ground_truth == 1: # False Negative
                #         cm_combined[0][1] += 1
                #     elif y_hat == 1 and ground_truth == 0: # False Positive
                #         cm_combined[1][0] += 1
                # Class distribution initial belief
                class_dist = [0.0, 0.0]
                cm_index_total = 0
                for c in context_set:
                    # Get feature vector for trial t for object o for context c
                    x_c = get_feature_vec(all_data, c, o, t).reshape(1, -1)
                    if x_c.size != 0:
                        # Get distribution prediction from classifier for trial t
                        clf = pickle.load(open("/Users/zachosman1/Desktop/Summer 2021/jivkolab/classifiers/" 
                                        + str(w) + "_" + str(c) + "_" + str(train_test_count) + ".pickle", "rb"))
                        dist_c = clf.predict_proba(x_c)
                        class_dist[0] += dist_c[0, 0]
                        class_dist[1] += dist_c[0, 1]
                        weight_c = compute_kappa(confusion_matrices[cm_index_total]) # For context and word
                        class_dist[0] += weight_c * dist_c[0, 0]
                        class_dist[1] += weight_c * dist_c[0, 1]
                    cm_index_total += 1
                # Normalize - divide class_dist[0] and class_dist[1] by sum
                normalized_class_dist = class_dist / sum(class_dist)
                # Determine predicted class label - which is larger between class_dist[0] and class_dist[1]
                y_hat = 1
                if normalized_class_dist[0] >= normalized_class_dist[1]:
                    y_hat = 0
                # Compare to ground truth and update cm_combined
                ground_truth = oracle.get_object_word(o, w)

                if y_hat == 1 and ground_truth == 1: # True Positive
                    cm_combined[0][0] += 1
                elif y_hat == 0 and ground_truth == 0: # True Negative
                    cm_combined[1][1] += 1
                elif y_hat == 0 and ground_truth == 1: # False Negative
                    cm_combined[0][1] += 1
                elif y_hat == 1 and ground_truth == 0: # False Positive
                    cm_combined[1][0] += 1
        print(cm_combined)
        print(multi_context_kappas)

        # Increment for naming of pickled classifiers
        train_test_count += 1

    # TODO - Generating data for # contexts used vs Kappa
    # for o in obj_test
    # for i in range(1, 6):
    #     np.rand.seed = i
    #     rand_contexts = np.rand.shuffle(context_set)
    #     for num in num_contexts):
    #         for j in range(0, num):




    # TODO - Generating data for # behaviors used vs Kappa

    pickle.dump(cm_combined, open("/Users/zachosman1/Desktop/Summer 2021/jivkolab/confusion_matrices/" 
                                      + str(w) + "_" + "cm_all" + ".pickle", "wb")) 

    pickle.dump(multi_context_kappas, open("/Users/zachosman1/Desktop/Summer 2021/jivkolab/kappas/" 
                                      + str(w) + "_" + "kappas" + ".pickle", "wb"))   
    # Set index to 0 for looping through individual confusion matrices to write to CSV
    cm_index_csv = 0

    # Open file to write stats to
    filename = w + "_stats.csv"
    stats_file = open("/Users/zachosman1/Desktop/Summer 2021/jivkolab/stats/" + filename, 'w')
    header = ['Context', 'Kappa', 'Accuracy', 'F1', 'Recall', 'Precision']
    writer = csv.DictWriter(stats_file, fieldnames=header)
    writer.writeheader()

    # Write stats for using all classifiers
    kappa_all = compute_kappa(cm_combined)
    accuracy_all = compute_acc(cm_combined)
    f1_score_all = compute_F1(cm_combined)
    recall_all = compute_recall(cm_combined)
    precision_all = compute_precision(cm_combined)
    writer.writerows([{'Context' : "All", 
                        'Kappa' : kappa_all, 
                        'Accuracy' : accuracy_all, 
                        'F1' : f1_score_all, 
                        'Recall' : recall_all, 
                        'Precision' : precision_all}])

    # Calculate stats and write to file
    for c in context_set:
        kappa_c = compute_kappa(confusion_matrices[cm_index_csv])
        accuracy_c = compute_acc(confusion_matrices[cm_index_csv])
        f1_score = compute_F1(confusion_matrices[cm_index_csv])
        recall = compute_recall(confusion_matrices[cm_index_csv])
        precision = compute_precision(confusion_matrices[cm_index_csv])
        print(c , kappa_c, accuracy_c, f1_score, recall, precision)
        writer.writerows([{'Context' : c, 
                          'Kappa' : kappa_c, 
                          'Accuracy' : accuracy_c, 
                          'F1' : f1_score, 
                          'Recall' : recall, 
                          'Precision' : precision}])
        cm_index_csv += 1
    stats_file.close()