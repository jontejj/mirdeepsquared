from keras.initializers import HeNormal, GlorotNormal, RandomNormal
from keras.layers import Input, Dense, Normalization
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from mirdeepsquared.common import Y_values
from mirdeepsquared.model import KerasModel
from keras.metrics import F1Score


class NumericalModel(KerasModel):
    def features_used(self):
        return ['combined_numerics']

    def train(self, train, val):
        X_train = self.X(train)
        Y_train = Y_values(train)
        X_val = self.X(val)
        Y_val = Y_values(val)

        # single_numeric_data = numeric_features[:, 4].reshape(-1, 1)  # Estimated probability (too good...)

        input = Input(shape=(4,), dtype='int32')
        normalizer_layer = Normalization()
        normalizer_layer.adapt(X_train)
        numeric_features_dense = Dense(8, activation='relu')(normalizer_layer(input))

        dense_layer = Dense(10000, activation='relu', kernel_initializer=HeNormal(seed=42), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=2.5, seed=42))(numeric_features_dense)
        output_layer = Dense(1, activation='sigmoid', kernel_initializer=GlorotNormal(seed=42), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42))(dense_layer)

        self.model = Model(inputs=[input], outputs=output_layer)

        self.model.compile(optimizer=Adam(learning_rate=0.003), loss='binary_crossentropy', metrics=['accuracy', F1Score(average='weighted', threshold=0.5, name='f1_score')])
        early_stopping = EarlyStopping(monitor='val_f1_score', mode='max', min_delta=0.00001, patience=10, start_from_epoch=4, restore_best_weights=True, verbose=1)
        history = self.model.fit(X_train, Y_train, epochs=100, batch_size=16, validation_data=(X_val, Y_val), callbacks=[early_stopping])  # verbose=0
        print(history.history)

    def weight(self):
        return 3
