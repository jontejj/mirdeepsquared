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

#val accuracy 0.9343
#test accuracy 0.913
#percentage_change test accuracy was 1.0
def train_density_map(df, epochs=200):
    X_train, Y_train, X_val, Y_val, X_test, Y_test, locations_train, locations_val, locations_test = split_data(prepare_data(df))
    density_maps=X_train[1]

    #TODO: also use "exp" feature so that the model can understand if the slope is for the mature or the star sequence
    input = Input(shape=(111,), dtype='int32', name='density_map')
    density_map_normalizer_layer = Normalization(mean=np.mean(density_maps, axis=0), variance=np.var(density_maps, axis=0))(input)
    dense_layer = Dense(10000, activation='relu', kernel_initializer=HeNormal(seed=42), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=2.5, seed=42))(density_map_normalizer_layer)
    output_layer = Dense(1, activation='sigmoid', kernel_initializer=GlorotNormal(seed=42), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42))(dense_layer)

    model = Model(inputs=[input], outputs=output_layer)

    model.compile(optimizer=Adam(learning_rate=0.003), loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=20, start_from_epoch=4, restore_best_weights=True, verbose=1)
    history = model.fit(X_train[1], Y_train, epochs=epochs, batch_size=16, validation_data=(X_val[1], Y_val), callbacks=[early_stopping]) #verbose=0

    return (model, history)


