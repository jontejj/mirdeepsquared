from keras.initializers import HeNormal, GlorotNormal, RandomNormal
from keras.layers import Input, Flatten, Dense, Concatenate, Normalization, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.metrics import F1Score
from .common import prepare_data, split_data, to_xy_with_location

import numpy as np

from sklearn.utils.class_weight import compute_class_weight
from keras.regularizers import l1_l2
from keras.regularizers import l2


# val accuracy 0.9343
# test accuracy 0.913
# percentage_change test accuracy was 1.0
def train_density_map(df, epochs=200):
    train, val, _ = split_data(prepare_data(df))

    X_train, Y_train, _ = to_xy_with_location(train)
    X_val, Y_val, _ = to_xy_with_location(val)
    density_maps = X_train[2]
    location_of_mature_star_and_hairpin = X_train[1]
    X_val_two_features = [X_val[1], X_val[2]]

    # Max accuracy on val: 0.8805, (l1=0.00001, l2_strength=0.001) -> 0.8925
    l1_strength = 0.01
    l2_strength = 0.01  # 0.8716 with 0.001, On test set 0.001 -> 0.8388 whilst 0.01 -> 0.8238

    input_location_of_mature_star_and_hairpin = Input(shape=(111, 4), dtype='float32', name='location_of_mature_star_and_hairpin')
    mean_values = np.mean(density_maps, axis=0)
    variance_values = np.var(density_maps, axis=0)
    input_density_maps = Input(shape=(111,), dtype='float32', name='density_map')
    density_map_normalizer_layer = Normalization(mean=mean_values, variance=variance_values)(input_density_maps)

    density_map_reshaped_as_rows = Reshape((111, 1), input_shape=(111,))(density_map_normalizer_layer)

    concatenated = Concatenate(axis=-1)([input_location_of_mature_star_and_hairpin, density_map_reshaped_as_rows])
    flatten_layer_structure = Flatten()(concatenated)

    dense_layer = Dense(10000, activation='relu', kernel_initializer=HeNormal(seed=42), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=2.5, seed=42), kernel_regularizer=l1_l2(l1=l1_strength, l2=l2_strength), bias_regularizer=l2(l2_strength))(flatten_layer_structure)
    output_layer = Dense(1, activation='sigmoid', kernel_initializer=GlorotNormal(seed=42), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42))(dense_layer)

    model = Model(inputs=[input_location_of_mature_star_and_hairpin, input_density_maps], outputs=output_layer)

    model.compile(optimizer=Adam(learning_rate=0.003), loss='binary_crossentropy', metrics=['accuracy', F1Score(average='weighted', threshold=0.5, name='f1_score')])
    model.summary()
    early_stopping = EarlyStopping(monitor='val_f1_score', mode='max', min_delta=0.00001, patience=20, start_from_epoch=4, restore_best_weights=True, verbose=1)
    class_weights = compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
    class_weights_dict = dict(enumerate(class_weights))

    history = model.fit([location_of_mature_star_and_hairpin, density_maps], Y_train, class_weight=class_weights_dict, epochs=epochs, batch_size=16, validation_data=(X_val_two_features, Y_val), callbacks=[early_stopping])  # verbose=0
    print(history.history)
    print("Max train accuracy: " + str(max(history.history['accuracy'])))
    print("Max validation accuracy: " + str(max(history.history['val_accuracy'])))
    print("Max validation F1-score: " + str(max(history.history['val_f1_score'])))
    return model
