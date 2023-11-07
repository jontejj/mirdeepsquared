import screed # a library for reading in FASTA/FASTQ

import numpy as np
import pandas as pd

from tensorflow import keras, convert_to_tensor
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import HeNormal, GlorotNormal, RandomNormal
from keras.layers import Input, Embedding, Flatten, Dense, TextVectorization, GlobalAveragePooling1D, Conv1D, GlobalMaxPooling1D, BatchNormalization, Concatenate, Normalization
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

KMER_SIZE = 6
NUCLEOTIDE_NR = 5 #U C A G D (D for Dummy)

vectorize_layer = TextVectorization(output_mode="int", input_shape=(1,))

def build_kmers(sequence, ksize):
    kmers = []
    n_kmers = len(sequence) - ksize + 1

    for i in range(n_kmers):
        kmer = sequence[i:i + ksize]
        kmers.append(kmer)

    return kmers

def read_kmers_from_file(filename, ksize):
    all_kmers = []
    for record in screed.open(filename):
        sequence = record.sequence
        kmers = build_kmers(sequence, ksize)
        all_kmers += kmers

    return all_kmers

def kmers_from_list(list, ksize):
    all_kmers = []
    for sequence in list:
        kmers = build_kmers(sequence, ksize)
        all_kmers += kmers

    return all_kmers

def get_model_chatgpt(consensus_sequences, density_maps, numeric_features):
    max_features = pow(NUCLEOTIDE_NR, KMER_SIZE)
    seq_length = len(max(consensus_sequences, key=len))

    input_layer_consensus_sequence = Input(shape=(1,), dtype='string', name='consensus_sequence')
    # Embedding layer for one-hot encoding
    vectorized_layer = vectorize_layer(input_layer_consensus_sequence)
    vectorize_layer.adapt(consensus_sequences)
    embedding_layer = Embedding(input_dim=max_features, output_dim=64, input_length=seq_length)(vectorized_layer)
    conv1d_layer = Conv1D(filters=64, kernel_size=3, activation='relu')(embedding_layer)
    maxpooling_layer = GlobalMaxPooling1D()(conv1d_layer)
    # Global average pooling over the sequence dimension
    #pooled_layer = GlobalAveragePooling1D()(embedding_layer)
    #flatten_layer = Flatten()(embedding_layer)

    # Dense layers for classification
    #batch_norm_layer = BatchNormalization(trainable=True)(maxpooling_layer) #TODO: remember to set trainable=False when inferring
    
    # Define the input layer for read density maps
    input_layer_density_map = Input(shape=(112,), dtype='int32', name='density_map')
    density_map_normalizer_layer = Normalization(mean=np.mean(density_maps, axis=0), variance=np.var(density_maps, axis=0))(input_layer_density_map)
    #TODO: use 1d CNN for the density maps
    density_map_dense = Dense(64, activation='relu')(density_map_normalizer_layer)

    input_layer_numeric_features = Input(shape=(2,), dtype='int64', name='numeric_features')
    normalizer_layer = Normalization()
    normalizer_layer.adapt(numeric_features)
    numeric_features_dense = Dense(64, activation='relu')(normalizer_layer(input_layer_numeric_features))

    concatenated = Concatenate()([maxpooling_layer, density_map_dense, numeric_features_dense])

    dense_layer = Dense(128, activation='relu', kernel_initializer=HeNormal(seed=42), kernel_regularizer='l1_l2', use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42), bias_regularizer='l2')(concatenated)
    output_layer = Dense(1, activation='sigmoid', kernel_initializer=GlorotNormal(seed=42), kernel_regularizer='l1_l2', bias_regularizer='l2', use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42))(dense_layer)

    model = Model(inputs=[input_layer_consensus_sequence, input_layer_density_map, input_layer_numeric_features], outputs=output_layer)

    initial_learning_rate = 0.0003
    lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

if __name__ == '__main__':

    df1 = pd.read_pickle("not_false_positives_small.pkl")
    df2 = pd.read_pickle("false_positives_small.pkl")

    df = pd.concat([df1, df2], axis=0)

    #From https://github.com/dhanush77777/DNA-sequencing-using-NLP/blob/master/DNA%20sequencing.ipynb
    df['consensus_sequence_kmers'] = df.apply(lambda x: build_kmers(x['consensus_sequence'], KMER_SIZE), axis=1)
    df['consensus_sequence_as_sentence'] = df.apply(lambda x: ' '.join(x['consensus_sequence_kmers']), axis=1)

    consensus_texts = df['consensus_sequence_as_sentence'].values.tolist()
    density_maps = df['read_density_map'].values.tolist()
    #TODO: add mature_read_count/star_read_count ratio as feature? Needed?
    numeric_feature_names = ['mature_read_count', 'star_read_count']
    numeric_features = df[numeric_feature_names]

    print(df['mm_struct'])
    y_data = df['false_positive'].values.astype(np.float32)

    X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, y_train, y_test = train_test_split(consensus_texts, density_maps, numeric_features, y_data, test_size=0.2, random_state=42)

    model = get_model_chatgpt(consensus_sequences=X1_train, density_maps=X2_train, numeric_features=X3_train)
    

    #TODO: pad consensus_texts with D
    #max_sequence_length = seq_length  # Adjust based on your data
    #X1_train = tf.keras.preprocessing.sequence.pad_sequences(X1_train, maxlen=max_sequence_length, padding='post', truncating='post', dtype=object, value='D')
    #X1_test = tf.keras.preprocessing.sequence.pad_sequences(X1_test, maxlen=max_sequence_length, padding='post', truncating='post', dtype=object, value='D')

    history = model.fit([np.asarray(X1_train), np.asarray(X2_train), np.asarray(X3_train)], np.asarray(y_train), epochs=15, batch_size=2, validation_data=([np.asarray(X1_test), np.asarray(X2_test), np.asarray(X3_test)], np.asarray(y_test)))
    print(history.history)

    pred = model.predict([np.asarray(X1_test), np.asarray(X2_test), np.asarray(X3_test)])
    pred = (pred>=0.50) #If probability is equal or higher than 0.50, It's most likely a false positive (True)
    print("Confusion matrix:")
    print(confusion_matrix(y_test,pred))
    print("Accuracy: " + str(accuracy_score(y_test,pred)))
    print("F1-score: " + str(f1_score(y_test,pred)))

    pass
