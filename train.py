import os
#import tensorflow
import screed # a library for reading in FASTA/FASTQ

import numpy as np
import pandas as pd

from tensorflow import keras
import tensorflow as tf
#from tensorflow.keras import layers
from keras.layers import Input, Embedding, Flatten, Dense, TextVectorization, GlobalAveragePooling1D, Conv1D, GlobalMaxPooling1D, BatchNormalization
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

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
    
def sklearn_dhanush(df, consensus_texts):
    cv = CountVectorizer(ngram_range=(4,4))
    X_miRNA = cv.fit_transform(consensus_texts)

    y_data = df['false_positive'].values

    X_train, X_test, y_train, y_test = train_test_split(X_miRNA, 
                                                    y_data, 
                                                    test_size = 0.20, 
                                                    random_state=42)

    classifier = MultinomialNB(alpha=0.1)
    classifier.fit(X_train, y_train)

    pred=classifier.predict(X_test)

    print(confusion_matrix(y_test,pred))
    print(accuracy_score(y_test,pred))

def get_basic_model(numeric_features, df):
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(numeric_features)

    print(normalizer(numeric_features.iloc[:3]))
    input1 = Input(shape=(2,), dtype=tf.int64, name='numerics')
    consensus_texts = list(df['consensus_sequence_as_sentence'])
    vectorize_layer = TextVectorization(output_mode="int")
    vectorize_layer.adapt(consensus_texts)

    input2 = Input(shape=(1,), dtype=tf.string, name='text')
    x = vectorize_layer(input2)
    max_features = pow(NUCLEOTIDE_NR, KMER_SIZE)
    x = Embedding(max_features + 1, 64)(x)

    #TODO: how to concatenate x and normalizer?
    merged = keras.layers.Concatenate(axis=1)([input1, x])
    model = tf.keras.Sequential([
        merged,
    Dense(10, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model

def get_model_chatgpt(df, seq_length):
    max_features = pow(NUCLEOTIDE_NR, KMER_SIZE)

    input_layer = Input(shape=(1,), dtype='string')
    # Embedding layer for one-hot encoding

    vectorized_layer = vectorize_layer(input_layer)

    embedding_layer = Embedding(input_dim=max_features, output_dim=64, input_length=seq_length)(vectorized_layer)
    
    conv1d_layer = Conv1D(filters=64, kernel_size=3, activation='relu')(embedding_layer)
    maxpooling_layer = GlobalMaxPooling1D()(conv1d_layer)
    # Global average pooling over the sequence dimension
    #pooled_layer = GlobalAveragePooling1D()(embedding_layer)
    #flatten_layer = Flatten()(embedding_layer)

    # Dense layers for classification
    #batch_norm_layer = BatchNormalization(trainable=True)(maxpooling_layer) #TODO: remember to set trainable=False when inferring
    dense_layer = Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(seed=42), kernel_regularizer='l1_l2', use_bias=True, bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=42), bias_regularizer='l2')(maxpooling_layer)
    output_layer = Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42), kernel_regularizer='l1_l2', bias_regularizer='l2', use_bias=True, bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=42))(dense_layer)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    initial_learning_rate = 0.0003
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)
    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['accuracy'])

    # Print a summary of the model architecture
    model.summary()
    return model

if __name__ == '__main__':


    #inputs = tf.random.uniform(shape=(1,3)) 
    #inputs = tf.constant([[[1, 1, 1], [2, 2, 2]]])

    #flatten = Flatten()
    # What should we expect as the output?
    #outputs = flatten(inputs)

    #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=42)
    #tf.keras.initializers.Constant(value=0.5)
    #dense_layer = Dense(units=4, activation="relu", use_bias=True, bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=42), kernel_initializer=tf.keras.initializers.HeNormal(seed=42), kernel_regularizer='l1_l2', bias_regularizer='l2')
    #print(dense_layer(outputs))
    #print(dense_layer.weights)
    #exit()

    df1 = pd.read_pickle("not_false_positives_small.pkl")
    df2 = pd.read_pickle("false_positives_small.pkl")

    df = pd.concat([df1, df2], axis=0)
    #print(df['mm_struct'])
    print(df['consensus_sequence'].tolist())
    #From https://github.com/dhanush77777/DNA-sequencing-using-NLP/blob/master/DNA%20sequencing.ipynb
    df['consensus_sequence_kmers'] = df.apply(lambda x: build_kmers(x['consensus_sequence'], KMER_SIZE), axis=1)
    df['consensus_sequence_as_sentence'] = df.apply(lambda x: ' '.join(x['consensus_sequence_kmers']), axis=1)
    print(df.head())

    print(df.dtypes)

    print(df['read_density_map'])

    consensus_texts = list(df['consensus_sequence_as_sentence'])
    
    y_data = df['false_positive'].values.astype(np.float32)

    #sklearn_dhanush(df, consensus_texts)

    #Keras
    numeric_feature_names = ['mature_read_count', 'star_read_count']
    numeric_features = df[numeric_feature_names]
    tf.convert_to_tensor(numeric_features)

    seq_length = len(max(consensus_texts, key=len))
    #input_layer = Input(shape=(seq_length,))

    #model = get_basic_model(numeric_features, df)
    model = get_model_chatgpt(df, seq_length)

    vectorize_layer.adapt(consensus_texts)
    #x_data = vectorize_layer(np.array(consensus_texts))

    #TODO: pad consensus_texts with D
    X_train, X_test, y_train, y_test = train_test_split(consensus_texts, 
                                                    y_data, 
                                                    test_size = 0.20, 
                                                    random_state=42)

    
    #max_sequence_length = seq_length
    #X_train = vectorize_layer(np.array(X_train)) #, maxlen=max_sequence_length, value="D")
    #X_test = vectorize_layer(np.array(X_test)) #, maxlen=max_sequence_length, value="D")

    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(2)

    pred = model.predict(np.array(X_train))

    print(pred.shape)
    history = model.fit(np.array(X_train), np.array(y_train), epochs=15, batch_size=2, validation_data=val_dataset)
    print(history.history)

    #integer_data = vectorize_layer(consensus_texts)
    #print(integer_data)

    #kmers = kmers_from_list(df['consensus_sequence'].tolist(), 6)
    #truth_labels = df['false_positive']
    #kmers = read_kmers_from_file("resources/known-mature-sequences-h_sapiens.fas", 20)
    #print(kmers[:5])

    #model = keras.models.Sequential()

    #model.add(Conv2D(1, kernel_size=(3,3), input_shape = (128, 128, 3)))

    #model.summary()
    pass
