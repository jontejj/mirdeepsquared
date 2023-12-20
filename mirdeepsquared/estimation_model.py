from sklearn.linear_model import LinearRegression

from mirdeepsquared.common import Y_values
from mirdeepsquared.model import MirdeepSquaredModel
import pickle

import numpy as np


class EstimationModel(MirdeepSquaredModel):
    def __init__(self):
        self.model = LinearRegression()

    def features_used(self):
        return ['estimated_probability', 'estimated_probability_uncertainty']

    def train(self, train, val):
        x = self.transpose(self.X(train))
        self.model.fit(x, Y_values(train))

    def transpose(self, X):
        return np.vstack(X).T

    def save(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, X):
        x_t = self.transpose(X)
        return np.asarray(self.model.predict(x_t))
