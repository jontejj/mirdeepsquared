from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf


class MirdeepSquaredModel(ABC):

    @abstractmethod
    def features_used(self):
        pass

    def X(self, df):
        if len(self.features_used()) == 1:
            # This avoids having to unpack the tuple later
            return np.asarray(df[self.features_used()[0]].values.tolist())
        else:
            df_features = tuple(np.asarray(df[feature].values.tolist()) for feature in self.features_used())
            return df_features

    @abstractmethod
    def train(self, train, val):
        pass

    @abstractmethod
    def save(self, model_path):
        pass

    @abstractmethod
    def load(self, model_path):
        pass

    @abstractmethod
    def predict(self, X):
        """Predicts the probability that the input samples are false positives"""
        pass

    @abstractmethod
    def weight(self):
        pass


class KerasModel(MirdeepSquaredModel):
    def __init__(self, model=None):
        self.model = model
        pass

    def load(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def save(self, model_path):
        self.model.save(model_path)

    def predict(self, X):
        return self.model.predict(X, verbose=0).reshape(1, -1)[0]
