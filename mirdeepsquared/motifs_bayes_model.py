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
        f = open(model_path, 'wb')
        pickle.dump(self.model, f)
        f.close()

    def load(self, model_path):
        f = open(model_path, 'rb')
        self.model = pickle.load(f)
        f.close()

    def predict(self, X):
        # Only return the false positive probability
        return np.asarray([x[1] for x in self.model.predict_proba(X)])

    # Binary crossentropy (on val): Structure: 2.5, Numericals: 3.8, density map: 0.55834, motifs: 5.98, Ensemble: 0.5587
    def weight(self):
        return 2
