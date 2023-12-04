
from keras.initializers import HeNormal, GlorotNormal, RandomNormal
from keras.layers import Input, Embedding, Dense, LSTM, Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l1_l2
from keras.regularizers import l2
from mirdeepsquared.common import Y_values
from mirdeepsquared.model import KerasModel
from keras.metrics import F1Score


class StructureModel(KerasModel):
    def features_used(self):
        return ['structure_as_1D_array']

    def train(self, train, val):
        X_train = self.X(train)
        Y_train = Y_values(train)
        X_val = self.X(val)
        Y_val = Y_values(val)

        # Max accuracy on val: 0.8805, (l1=0.00001, l2_strength=0.001) -> 0.8925
        l1_strength = 0.0001
        l2_strength = 0.001  # 0.8716 with 0.001, On test set 0.001 -> 0.8388 whilst 0.01 -> 0.8238
        input = Input(shape=(111,), dtype='float32', name='structure_as_1D_array')
        embedding_layer = Embedding(input_dim=17, output_dim=128, input_length=111, mask_zero=True)(input)
        bidirectional_lstm = Bidirectional(LSTM(128))(embedding_layer)
        dense = Dense(10000, activation='relu', kernel_initializer=HeNormal(seed=42), kernel_regularizer=l1_l2(l1=l1_strength, l2=l2_strength), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42), bias_regularizer=l2(l2_strength))(bidirectional_lstm)
        output_layer = Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(l1=l1_strength, l2=l2_strength), bias_regularizer=l2(l2_strength), kernel_initializer=GlorotNormal(seed=42), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42))(dense)

        self.model = Model(inputs=[input], outputs=output_layer)

        self.model.compile(optimizer=Adam(learning_rate=0.0003), loss='binary_crossentropy', metrics=['accuracy', F1Score(average='weighted', threshold=0.5, name='f1_score')])
        self.model.summary()
        early_stopping = EarlyStopping(monitor='val_f1_score', mode='max', min_delta=0.00001, patience=20, start_from_epoch=4, restore_best_weights=True, verbose=1)

        history = self.model.fit(X_train, Y_train, epochs=200, batch_size=16, validation_data=(X_val, Y_val), callbacks=[early_stopping])
        print(history.history)
