import numpy as np
import pandas as pd
from pandas import Series
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import math

# read in objects and words
data = pd.read_csv("crush_audio.txt", header=None)

# remove trial numbers from object names
for i in range(data.shape[0]):
    data[0][i] = (data[0][i])[:-3]

# seperate into inputs and outputs
y = data.iloc[:, 0].values
x = data.iloc[:, 1:].values

# split into training and testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

# train SVM classifier - find best model with regularization
best_C = 0
best_kappa = 0
C_grid = np.logspace(-9, 6, 31)
for c in C_grid:
    clf = SVC(C=c, decision_function_shape = 'ovo', gamma='scale')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    curr_kappa = cohen_kappa_score(y_test, y_pred)
    if curr_kappa > best_kappa:
        best_kappa = curr_kappa
        best_C = c
    # print(c)
    # print(cohen_kappa_score(y_test, y_pred))
    # print()

print(best_C)
print(best_kappa)
# clf = SVC(C=c, decision_function_shape = 'ovo', gamma='scale')
# clf.fit(x_train, y_train)
# y_pred = clf.predict(x_test)

# print(y_pred)
# print()
# print(y_test)



