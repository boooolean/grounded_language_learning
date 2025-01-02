import pandas as pd
from make_modality_matrix import make_modality_matrix

class Oracle:
    def __init__(self):
        self.ground_truth = make_modality_matrix()

    def get_object_word(self, obj_id, predicate):
        return self.ground_truth.loc[obj_id, predicate]
