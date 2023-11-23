
from keras.initializers import HeNormal, GlorotNormal, RandomNormal
from keras.layers import Input, Embedding, Dense, GlobalAveragePooling1D, LSTM, Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l1_l2
from keras.regularizers import l2
from mirdeepsquared.common import prepare_data, split_data_once, to_xy_with_location


def train_simple_structure(df):
    train, val = split_data_once(prepare_data(df))
    X_train, Y_train, _ = to_xy_with_location(train)
    X_val, Y_val, _ = to_xy_with_location(val)

    # Max accuracy on val: 0.8805, (l1=0.00001, l2_strength=0.001) -> 0.8925
    l1_strength = 0.0001
    l2_strength = 0.001  # 0.8716 with 0.001, On test set 0.001 -> 0.8388 whilst 0.01 -> 0.8238
    input = Input(shape=(111,), dtype='float32', name='structure_as_1D_array')
    embedding_layer = Embedding(input_dim=17, output_dim=128, input_length=111, mask_zero=True)(input)
    bidirectional_lstm = Bidirectional(LSTM(128))(embedding_layer)
    # dense_after_lstm = Dense(64, activation='relu', kernel_initializer=HeNormal(seed=42), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42))(bidirectional_lstm)
    dense = Dense(10000, activation='relu', kernel_initializer=HeNormal(seed=42), kernel_regularizer=l1_l2(l1=l1_strength, l2=l2_strength), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42), bias_regularizer=l2(l2_strength))(bidirectional_lstm)
    global_average = GlobalAveragePooling1D()(dense)
    output_layer = Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(l1=l1_strength, l2=l2_strength), bias_regularizer=l2(l2_strength), kernel_initializer=GlorotNormal(seed=42), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42))(global_average)

    model = Model(inputs=[input], outputs=output_layer)

    model.compile(optimizer=Adam(learning_rate=0.0003), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, start_from_epoch=4, restore_best_weights=True, verbose=1)

    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # Resume and improve accuracy
    # model = load_model("train-simple-model.keras")
    history = model.fit(X_train[3], Y_train, epochs=200, batch_size=16, validation_data=(X_val[3], Y_val), callbacks=[early_stopping])
    print(history.history)
    return model
