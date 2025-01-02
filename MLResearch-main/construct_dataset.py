import numpy as np
import pandas as pd
from oracle import Oracle
from sklearn import svm

def construct_dataset(obj_list, w, c, oracle):
    # Use context provided to select proper x-data
    raw_data = pd.read_csv('/Users/zachosman1/Desktop/Summer 2021/jivkolab/contexts/' + c + '.txt', header=None)
    raw_data = raw_data.dropna()

    x_data = list()
    y_data = list()
    for ind in raw_data.index:
        if (raw_data.at[ind,0])[:-3] in obj_list:
            x_data.append(list(raw_data.loc[ind, 1:]))
            y_data.append(oracle.get_object_word((raw_data.at[ind,0])[:-3], w))

    return np.array(x_data), np.array(y_data)
