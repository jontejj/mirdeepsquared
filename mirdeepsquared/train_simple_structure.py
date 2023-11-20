
from keras.initializers import HeNormal, GlorotNormal, RandomNormal
from keras.layers import Input, Embedding, Dense, GlobalAveragePooling1D, Conv1D, Concatenate, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l1_l2
from keras.regularizers import l2
from mirdeepsquared.common import prepare_data, split_data, to_xy_with_location

from keras.layers import LSTM, Dense, Embedding, Bidirectional

def train_simple_structure(df):
    train, val, _ = split_data(prepare_data(df))
    X_train, Y_train, _ = to_xy_with_location(train)
    X_val, Y_val, _ = to_xy_with_location(val)
    
    input_precursor = Input(shape=(111,5), dtype='float32', name='precursor')
    #Max accuracy on val: 0.8805, (l1=0.00001, l2_strength=0.001) -> 0.8925
    l1_strength = 0.0001
    l2_strength = 0.001 #0.8716 with 0.001, On test set 0.001 -> 0.8388 whilst 0.01 -> 0.8238
    input = Input(shape=(111,), dtype='float32', name='structure_as_1D_array')
    embedding_layer = Embedding(input_dim=17, output_dim=128, input_length=111, mask_zero=True)(input)
    bidirectional_lstm = Bidirectional(LSTM(128))(embedding_layer)
    #dense_after_lstm = Dense(64, activation='relu', kernel_initializer=HeNormal(seed=42), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42))(bidirectional_lstm)
    reshaped_lstm = Reshape((4, 64), input_shape=(256,))(bidirectional_lstm)
    precursor_conv1d_k3 = Conv1D(filters=64, kernel_size=3, activation='relu')(input_precursor) #64 filters -> 0.872, 128 -> 0.848
    #avg_pooling_k3 = AveragePooling1D(pool_size=2)(conv1d_k3)
    #conv1d_k5 = Conv1D(filters=64, kernel_size=5, activation='relu')(embedding_layer)
    #avg_pooling_k5 = AveragePooling1D(pool_size=4)(conv1d_k5)
    #concatenated = Concatenate(axis=1)([reshaped_lstm, avg_pooling_k3, avg_pooling_k5])
    #reshaped = Reshape((83, 64, 1), input_shape=(83,64))(concatenated)

    #conv2d = Conv2D(filters=256, kernel_size=3, activation='relu')(reshaped)
    #global_average = GlobalAveragePooling2D()(conv2d)

    concatenated = Concatenate(axis=1)([reshaped_lstm, precursor_conv1d_k3])
    dense = Dense(10000, activation='relu', kernel_initializer=HeNormal(seed=42), kernel_regularizer=l1_l2(l1=l1_strength, l2=l2_strength), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42), bias_regularizer=l2(l2_strength))(concatenated)
    global_average = GlobalAveragePooling1D()(dense)
    output_layer = Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(l1=l1_strength, l2=l2_strength), bias_regularizer=l2(l2_strength), kernel_initializer=GlorotNormal(seed=42), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42))(global_average)

    model = Model(inputs=[input, input_precursor], outputs=output_layer)

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
    model.compile(optimizer=Adam(learning_rate=0.0003), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, start_from_epoch=4, restore_best_weights=True, verbose=1)
    
    #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    #Resume and improve accuracy
    #model = load_model("train-simple-model.keras")
    history = model.fit([X_train[3], X_train[5]], Y_train, epochs=200, batch_size=16, validation_data=([X_val[3], X_val[5]], Y_val), callbacks=[early_stopping])
    return (model, history)
