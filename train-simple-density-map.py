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
from train import read_dataframes, prepare_data, split_data
from tensorflow import keras
from keras.saving import load_model

import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

if __name__ == '__main__':

    df = read_dataframes(["resources/dataset/true_positives_TCGA_LUSC.pkl", "resources/dataset/false_positives_SRR2496781-84_bigger.pkl"])

    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(prepare_data(df))
    consensus_sequences=X_train[0]
    density_maps=X_train[1]
    numeric_features=X_train[2]

    input = Input(shape=(112,), dtype='int32', name='density_map')
    density_map_normalizer_layer = Normalization(mean=np.mean(density_maps, axis=0), variance=np.var(density_maps, axis=0))(input)
    dense_layer = Dense(10000, activation='relu', kernel_initializer=HeNormal(seed=42), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=2.5, seed=42))(density_map_normalizer_layer)
    output_layer = Dense(1, activation='sigmoid', kernel_initializer=GlorotNormal(seed=42), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42))(dense_layer)

    model = Model(inputs=[input], outputs=output_layer)

    model.compile(optimizer=Adam(learning_rate=0.003), loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, start_from_epoch=4, restore_best_weights=True, verbose=1)
    history = model.fit(X_train[1], Y_train, epochs=100, batch_size=16, validation_data=(X_val[1], Y_val), callbacks=[early_stopping]) #verbose=0
 
    print(history)

    model.save("train-simple-model.keras")


