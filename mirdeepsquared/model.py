from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf


class MirdeepSquaredModel(ABC):

    @abstractmethod
    def features_used(self):
        """Returns the column names of the features that this model uses"""
        pass

    def X(self, df):
        """Returns the data for the features listed in features_used for the given dataframe"""
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
        """Saves self.model into a file at model_path"""
        pass

    @abstractmethod
    def load(self, model_path):
        """Loads a model into self.model from the file at model_path"""
        pass

    @abstractmethod
    def predict(self, X):
        """Predicts the probability that the input samples are false positives"""
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
