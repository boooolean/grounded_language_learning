import numpy as np
import pandas as pd


def make_data_matrix(behavior_modality, class_label_vec):
    file_name = behavior_modality + '.txt'

    data = pd.read_csv(file_name, header=None)

    # Construct y-data
    y_data = np.zeros_like(data[0])
    y_data.fill(-1)
    for row in range(len(data.index)):
        obj_name = (data.at[row,0])[:-3]
        if class_label_vec[obj_name] == 1:
            y_data[row] = 1

    data[0] = y_data
    
    return data



