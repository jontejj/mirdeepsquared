from sklearn.naive_bayes import CategoricalNB
from mirdeepsquared.common import Y_values
from mirdeepsquared.model import MirdeepSquaredModel
import pickle

import numpy as np


# Accuracy: 0.628428927680798
class MotifModel(MirdeepSquaredModel):
    def __init__(self):
        self.model = CategoricalNB(force_alpha=True)

    def features_used(self):
        return ['motifs']

    def train(self, train, val):
        self.model.fit(self.X(train), Y_values(train))

    def save(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, X):
        # Only return the false positive probability
        return np.asarray([x[1] for x in self.model.predict_proba(X)])
