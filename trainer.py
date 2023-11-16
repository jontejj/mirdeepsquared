from tensorflow import keras
#Make training reproducable
keras.utils.set_random_seed(42)
import tensorflow as tf
tf.config.experimental.enable_op_determinism()

from mirdeepsquared.train_simple_structure import train_simple_structure
from mirdeepsquared.train_simple_density_map import train_density_map
from mirdeepsquared.train_simple_numerical_feature import train_simple_numerical_features
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import HeNormal, GlorotNormal, RandomNormal
from keras.layers import Input, Embedding, Flatten, Dense, TextVectorization, GlobalAveragePooling1D, Conv1D, Conv2D, GlobalMaxPooling1D, BatchNormalization, Concatenate, Normalization
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from mirdeepsquared.train import list_of_pickle_files_in, read_dataframes, prepare_data, split_data, train_main
from tensorflow import keras
from keras.saving import load_model

import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

if __name__ == '__main__':

    human = list_of_pickle_files_in("resources/dataset") #with dataset-backup (.pkl files from commit 4b9cf56) the accuracy is better because the entries were not filtered with the mirgene db file for some reason (Issue #1)
    other_species = list_of_pickle_files_in("resources/dataset/other_species")
    df = read_dataframes(human) # human + other_species) # 

    model, history = train_main("resources/dataset", "best-not-seen-test-model.keras", "mirdeepsquared/default-hyperparameters.yaml", "trainer-train-results.csv")
    #model, history = train_density_map(df)
    #model, history = train_simple_numerical_features(df)
    #model, history = train_simple_structure(df)
    print(history.history)
    model.save("mirdeepsquared/train-simple-model.keras")

    print("Max accuracy on val: " + str(max(history.history['val_accuracy'])))
    #TODO: do cross validation


