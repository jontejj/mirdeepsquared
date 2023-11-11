from keras.constraints import MaxNorm
from keras.src.optimizers.schedules.learning_rate_schedule import ExponentialDecay
from tensorflow import keras
#Make training reproducable
keras.utils.set_random_seed(42)
import tensorflow as tf
tf.config.experimental.enable_op_determinism()
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import HeNormal, GlorotNormal, RandomNormal
from keras.layers import Input, Embedding, Flatten, Dense, TextVectorization, GlobalAveragePooling1D, Conv1D, Conv2D, GlobalMaxPooling1D, BatchNormalization, Concatenate, Normalization, Reshape, AveragePooling1D, MaxPooling1D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l1_l2
from keras.regularizers import l2
from train import read_dataframes, prepare_data, split_data
from tensorflow import keras
from keras.saving import load_model

from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Bidirectional, Dropout

if __name__ == '__main__':

    df = read_dataframes(["resources/dataset/true_positives_TCGA_LUSC.pkl", "resources/dataset/false_positives_SRR2496781-84_bigger.pkl"])
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(prepare_data(df))

    #Max accuracy on val: 0.8805
    l1_strength = 0.0001
    l2_strength = 0.01
    input = Input(shape=(112,), dtype='float32', name='structure_as_1D_array')
    embedding_layer = Embedding(input_dim=8, output_dim=128, input_length=112, mask_zero=True)(input)
    bidirectional_lstm = Bidirectional(LSTM(128))(embedding_layer)
    dense_after_lstm = Dense(320, activation='relu', kernel_initializer=HeNormal(seed=42), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42))(bidirectional_lstm)
    reshaped_lstm = Reshape((10, 32), input_shape=(320,))(dense_after_lstm)
    conv1d_k3 = Conv1D(filters=32, kernel_size=3, activation='relu')(embedding_layer) #TODO: specify input directly instead of the output of the lstm?
    avg_pooling_k3 = AveragePooling1D(pool_size=2)(conv1d_k3)
    #maxpooling_k3 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv1d_k3)
    conv1d_k5 = Conv1D(filters=32, kernel_size=5, activation='relu')(embedding_layer)
    avg_pooling_k5 = AveragePooling1D(pool_size=4)(conv1d_k5)
    #maxpooling_k5 = MaxPooling1D(pool_size=4, strides=1, padding='same')(conv1d_k5)
    #concatenated = Concatenate(axis=1)([maxpooling_k3, maxpooling_k5])
    concatenated = Concatenate(axis=1)([reshaped_lstm, avg_pooling_k3, avg_pooling_k5])

    flatten = Flatten()(concatenated)

    dense = Dense(128, activation='relu', kernel_initializer=HeNormal(seed=42), kernel_regularizer=l1_l2(l1=l1_strength, l2=l2_strength), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42), bias_regularizer=l2(l2_strength))(flatten)
    output_layer = Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(l1=l1_strength, l2=l2_strength), bias_regularizer=l2(l2_strength), kernel_initializer=GlorotNormal(seed=42), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42))(dense)

    model = Model(inputs=[input], outputs=output_layer)

    #model = Sequential()
    #Old
    #model.add(Input(shape=(112,), dtype='float32', name='structure_as_1D_array'))
    #model.add(Reshape((8, 14, 1), input_shape=(112,)))
    #model.add(Conv2D(3, kernel_size=(3, 3), activation='relu', input_shape=(16, 8, 14, 1), padding='same'))
    #model.add(Flatten())
    #New
    #model.add(Embedding(input_dim=8, output_dim=128, input_length=112, mask_zero=True))
    #model.add(Bidirectional(LSTM(128)))
    #model.add(Reshape((16, 16), input_shape=(256,)))
    #model.add(Conv1D(filters=3, kernel_size=3, activation='relu')) #With only 3 filters -> 0.8597
    #model.add(MaxPooling1D(pool_size=2, strides=1, padding='valid'))
    #model.add(Conv1D(filters=3, kernel_size=5, activation='relu')) #With additional Conv1d -> 0.8776
    #model.add(MaxPooling1D(pool_size=2, strides=1, padding='valid'))
    #model.add(Flatten())
    #With 128 instead of 64 -> 0.8687
    #model.add(Dense(128, activation='relu', kernel_initializer=HeNormal(seed=42), kernel_regularizer='l1_l2', use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42), bias_regularizer='l2'))
    #####Exclude #### model.add(Dropout(0.6, input_shape=(256,))) #Without: 0.8597, With 0.6 -> 0.8328, Other tests: 0.5 -> 0.8328, 0.6 -> 0.8448, 0.8 -> 0.7194 (without reg on Dense)
    #model.add(Dense(1, activation='sigmoid', kernel_regularizer='l1_l2', bias_regularizer='l2', kernel_initializer=GlorotNormal(seed=42), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42)))
    #lr_schedule = ExponentialDecay(0.003, decay_steps=1000, decay_rate=0.96), lr_schedule
    model.compile(optimizer=Adam(learning_rate=0.003), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, start_from_epoch=4, restore_best_weights=True, verbose=1)
    
    #Resume and improve accuracy
    #model = load_model("train-simple-model.keras")
    history = model.fit(X_train[3], Y_train, epochs=100, batch_size=16, validation_data=(X_val[3], Y_val), callbacks=[early_stopping])

    print(history.history)
    model.save("train-simple-model.keras")

    print("Max accuracy on val: " + str(max(history.history['val_accuracy'])))
