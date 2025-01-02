import numpy as np
import pandas as pd
import math


def make_modality_matrix():
    data = pd.read_csv("cy101_labels.csv")
    word_list = list()

    words_data = data['words'].tolist()
    objects_data = data['object'].tolist()


    for word_set in words_data:
        # If word is not for the no_object instance
        if not isinstance(word_set, float):
            individ_words = word_set.split(",") # Remove commas to split into individual words

            for word in individ_words:
                word_stripped = word.strip() # Remove spaces

                if word_stripped not in word_list and word_stripped != '':
                    word_list.append(word_stripped) # Add to list if not already         

    # Make Pandas DataFrame
    modality_matrix = pd.DataFrame(0, index=objects_data, columns=word_list)

    # Fill matrix
    row_num = 0
    for word_set in words_data:
        if not isinstance(word_set, float):
            individ_words = word_set.split(",")

            for ind_word in individ_words:
                word_stripped = ind_word.strip()
                if word_stripped != '':
                    modality_matrix.loc[modality_matrix.index[row_num], word_stripped] = 1
        row_num = row_num + 1

    return modality_matrix


