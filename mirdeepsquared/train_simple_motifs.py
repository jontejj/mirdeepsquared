
from keras.initializers import HeNormal, GlorotNormal, RandomNormal
from keras.layers import Input, Dense, GlobalMaxPooling1D, Conv1D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from mirdeepsquared.common import prepare_data, split_data_once, to_xy_with_location

import random


def all_true_or_random(motifs):
    if motifs.all():
        return True
    elif motifs[2] == 1:  # samples that has_cnnc_motif is much more likely to be a true positive
        return bool(random.randint(0, 4))
    else:
        return bool(random.getrandbits(1))


def train_simple_motifs(df):
    train, val = split_data_once(prepare_data(df))
    X_train, Y_train, _ = to_xy_with_location(train)
    X_val, Y_val, _ = to_xy_with_location(val)

    input = Input(shape=(3, 2), dtype='float32', name='motifs_one_hot_encoded')
    conv1d_k3 = Conv1D(filters=10, kernel_size=3, activation='relu', padding='valid')(input)
    maxpooling_layer = GlobalMaxPooling1D()(conv1d_k3)
    dense = Dense(10, activation='relu', kernel_initializer=HeNormal(seed=42), kernel_regularizer='l1_l2', use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42))(maxpooling_layer)
    output_layer = Dense(1, activation='sigmoid', kernel_initializer=GlorotNormal(seed=42), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42))(dense)

    model = Model(inputs=[input], outputs=output_layer)

    model.compile(optimizer=Adam(learning_rate=0.003), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, start_from_epoch=4, restore_best_weights=True, verbose=1)

    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # Resume and improve accuracy
    # model = load_model("train-simple-model.keras")
    history = model.fit(X_train[6], Y_train, epochs=200, batch_size=16, validation_data=(X_val[6], Y_val), callbacks=[early_stopping])
    print(history.history)
    return model
