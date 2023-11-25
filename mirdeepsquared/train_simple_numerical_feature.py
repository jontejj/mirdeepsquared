from keras.initializers import HeNormal, GlorotNormal, RandomNormal
from keras.layers import Input, Dense, Normalization
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from mirdeepsquared.common import prepare_data, split_data_once, to_xy_with_location


def train_simple_numerical_features(df):
    train, val = split_data_once(prepare_data(df))
    X_train, Y_train, _ = to_xy_with_location(train)
    X_val, Y_val, _ = to_xy_with_location(val)
    numeric_features = X_train[4]

    # single_numeric_data = numeric_features[:, 4].reshape(-1, 1)  # Estimated probability (too good...)

    input = Input(shape=(7,), dtype='int32')
    normalizer_layer = Normalization()
    normalizer_layer.adapt(numeric_features)
    numeric_features_dense = Dense(8, activation='relu')(normalizer_layer(input))

    dense_layer = Dense(10000, activation='relu', kernel_initializer=HeNormal(seed=42), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=2.5, seed=42))(numeric_features_dense)
    output_layer = Dense(1, activation='sigmoid', kernel_initializer=GlorotNormal(seed=42), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42))(dense_layer)

    model = Model(inputs=[input], outputs=output_layer)

    model.compile(optimizer=Adam(learning_rate=0.003), loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, start_from_epoch=4, restore_best_weights=True, verbose=1)
    history = model.fit(numeric_features, Y_train, epochs=100, batch_size=16, validation_data=(X_val[4], Y_val), callbacks=[early_stopping])  # verbose=0
    print(history.history)
    return model
