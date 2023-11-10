from tensorflow import keras
#Make training reproducable
keras.utils.set_random_seed(42)
import tensorflow as tf
tf.config.experimental.enable_op_determinism()
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import HeNormal, GlorotNormal, RandomNormal
from keras.layers import Input, Embedding, Flatten, Dense, TextVectorization, GlobalAveragePooling1D, Conv1D, Conv2D, GlobalMaxPooling1D, BatchNormalization, Concatenate, Normalization
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from train import build_structure_1d, read_dataframes, prepare_data, split_data
from tensorflow import keras
from keras.saving import load_model

import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Bidirectional, Dropout

if __name__ == '__main__':

    df = read_dataframes(["resources/dataset/true_positives_TCGA_LUSC.pkl", "resources/dataset/false_positives_SRR2496781-84_bigger.pkl"])
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(prepare_data(df))

    model = Sequential()
    model.add(Embedding(input_dim=8, output_dim=128, input_length=112))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(0.8, input_shape=(256,)))
    model.add(Dense(1, activation='sigmoid'))
    #TODO: try tf.keras.optimizers.RMSprop(learning_rate=0.1)
    model.compile(optimizer=Adam(learning_rate=0.003), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, start_from_epoch=4, restore_best_weights=True, verbose=1)
    history = model.fit(X_train[3], Y_train, epochs=1000, validation_data=(X_val[3], Y_val), callbacks=[early_stopping])

    print(history.history)
    model.save("train-simple-model.keras")

"""
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(df)
    consensus_sequences=X_train[0]
    density_maps=X_train[1]
    numeric_features=X_train[2]

    input = Input(shape=(8, 14, 1), dtype='float32', name='structure_as_matrix')
    conv2d_layer = Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(16, 8, 14, 1), padding='same')(input)
    matrix_dense = Dense(8, activation='relu')(conv2d_layer)
    flatten_layer_structure = Flatten()(matrix_dense)

    dense_layer = Dense(10000, activation='relu')(flatten_layer_structure) #kernel_initializer=HeNormal(seed=42), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42)
    output_layer = Dense(1, activation='sigmoid')(dense_layer) #, kernel_initializer=GlorotNormal(seed=42), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42)

    model = Model(inputs=[input], outputs=output_layer)

    model.compile(optimizer=Adam(learning_rate=0.0003), loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, start_from_epoch=4, restore_best_weights=True, verbose=1)
    history = model.fit(X_train[3], Y_train, epochs=100, batch_size=16, validation_data=(X_val[3], Y_val), callbacks=[early_stopping]) #verbose=0
 
    print(history)

    model.save("train-simple-model.keras")
"""

