from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import HeNormal, GlorotNormal, RandomNormal
from keras.layers import Input, Embedding, Flatten, Dense, TextVectorization, GlobalAveragePooling1D, Conv1D, Conv2D, GlobalMaxPooling1D, BatchNormalization, Concatenate, Normalization
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from .train import list_of_pickle_files_in, read_dataframes, prepare_data, split_data
from keras.saving import load_model

import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def train_simple_numerical_features(df):
    X_train, Y_train, X_val, Y_val, X_test, Y_test, locations_train, locations_val, locations_test = split_data(prepare_data(df))
    numeric_features=X_train[2]

    single_numeric_data = numeric_features[ :,4].reshape(-1, 1) #Estimated probability

    input = Input(shape=(1,), dtype='int32')
    normalizer_layer = Normalization()
    normalizer_layer.adapt(single_numeric_data)
    numeric_features_dense = Dense(8, activation='relu')(normalizer_layer(input))

    dense_layer = Dense(10000, activation='relu', kernel_initializer=HeNormal(seed=42), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=2.5, seed=42))(numeric_features_dense)
    output_layer = Dense(1, activation='sigmoid', kernel_initializer=GlorotNormal(seed=42), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42))(dense_layer)

    model = Model(inputs=[input], outputs=output_layer)

    model.compile(optimizer=Adam(learning_rate=0.003), loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, start_from_epoch=4, restore_best_weights=True, verbose=1)
    history = model.fit(single_numeric_data, Y_train, epochs=100, batch_size=16, validation_data=(X_val[2][ :,4], Y_val), callbacks=[early_stopping]) #verbose=0
    return (model, history)


