import numpy as np
import pandas as pd
import math

def upsample_data(X_data, y_data):
    positive_count = np.count_nonzero(y_data == 1)
    negative_count = len(y_data) - positive_count
    if positive_count < negative_count:
        dup_amount = math.ceil(negative_count / positive_count)
        original_len = len(y_data)
        for i in range(0, dup_amount):
            for index in range(0, original_len):
                if y_data[index] == 1:
                    y_data = np.append(y_data, 1)
                    X_data = np.vstack([X_data, X_data[index].reshape(1, -1)])
                    if np.count_nonzero(y_data == 1) == np.count_nonzero(y_data == 0):
                        return X_data, y_data

    return X_data, y_data

