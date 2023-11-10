from tensorflow import keras

#Make training reproducable
keras.utils.set_random_seed(42)
import tensorflow as tf
tf.config.experimental.enable_op_determinism()

import os
import screed # a library for reading in FASTA/FASTQ

import numpy as np
import pandas as pd

from keras.preprocessing.sequence import pad_sequences
from keras.initializers import HeNormal, GlorotNormal, RandomNormal
from keras.layers import Input, Embedding, Flatten, Dense, TextVectorization, GlobalAveragePooling1D, Conv1D, Conv2D, GlobalMaxPooling1D, BatchNormalization, Concatenate, Normalization, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
from keras.metrics import F1Score
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import csv

KMER_SIZE = 6
NUCLEOTIDE_NR = 5 #U C A G D (D for Dummy)

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

def build_structure_1D(pri_struct, mm_struct, mm_offset):
    pri_struct_padded = pri_struct.ljust(112, '-')

    pri_struct_truncated = pri_struct_padded[:112]

    # Define a mapping for characters
    char_mapping = {'-': 0, '.': 1, '(': 2, ')': 3}

    #TODO: can mm_struct be expected to match pri_struct?
    mm_mapping = {0: 4, 1: 5, 2: 6, 3:7}

    # Convert the string to a 1D array
    array_1d = [char_mapping[char] for char in pri_struct_truncated]
    array_1d[mm_offset:mm_offset+len(mm_struct)] = [mm_mapping[digit] for digit in array_1d[mm_offset:mm_offset+len(mm_struct)]]

    return array_1d

def get_model(consensus_sequences, density_maps, numeric_features, model_size = 64, initial_learning_rate = 0.0003, batch_size = 6, regularize = True):
    max_features = pow(NUCLEOTIDE_NR, KMER_SIZE)
    seq_length = len(max(consensus_sequences, key=len))

    #Input 1 - consensus_sequence
    input_layer_consensus_sequence = Input(shape=(1,), dtype='string', name='consensus_sequence')
    vectorize_layer = TextVectorization(output_mode="int", input_shape=(1,))
    vectorized_layer = vectorize_layer(input_layer_consensus_sequence)
    vectorize_layer.adapt(consensus_sequences)
    embedding_layer = Embedding(input_dim=max_features, output_dim=model_size, input_length=seq_length)(vectorized_layer)
    conv1d_layer = Conv1D(filters=model_size, kernel_size=3, activation='relu')(embedding_layer)
    maxpooling_layer = GlobalMaxPooling1D()(conv1d_layer)

    #batch_norm_layer = BatchNormalization(trainable=True)(maxpooling_layer) #TODO: remember to set trainable=False when inferring
    
    #Input 2 - density maps
    input_layer_density_map = Input(shape=(112,), dtype='int32', name='density_map')
    density_map_normalizer_layer = Normalization(mean=np.mean(density_maps, axis=0), variance=np.var(density_maps, axis=0))(input_layer_density_map)
    #TODO: use 1d CNN for the density maps
    density_map_dense = Dense(model_size, activation='relu')(density_map_normalizer_layer)

    #Input 3 - numerical features
    input_layer_numeric_features = Input(shape=(6,), dtype='float32', name='numeric_features')
    normalizer_layer = Normalization()
    normalizer_layer.adapt(numeric_features)
    numeric_features_dense = Dense(model_size, activation='relu')(normalizer_layer(input_layer_numeric_features))

    #Input 4 - structural information
    input_structure_as_matrix = Input(shape=(112,), dtype='float32', name='structure_as_1D_array')
    reshaped_as_matrix = Reshape((8, 14, 1), input_shape=(112,))(input_structure_as_matrix)
    #TODO: Normalize input_structure_as_matrix?
    conv2d_layer = Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(batch_size, 8, 14, 1), padding='same')(reshaped_as_matrix)
    matrix_dense = Dense(model_size, activation='relu')(conv2d_layer)
    flatten_layer_structure = Flatten()(matrix_dense)

    concatenated = Concatenate()([maxpooling_layer, density_map_dense, numeric_features_dense, flatten_layer_structure])

    if regularize:
        dense_layer = Dense(10000, activation='relu', kernel_initializer=HeNormal(seed=42), kernel_regularizer='l1_l2', use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42), bias_regularizer='l2')(concatenated)
        output_layer = Dense(1, activation='sigmoid', kernel_initializer=GlorotNormal(seed=42), kernel_regularizer='l1_l2', bias_regularizer='l2', use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42))(dense_layer)
    else:
        dense_layer = Dense(10000, activation='relu', kernel_initializer=HeNormal(seed=42), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42))(concatenated)
        output_layer = Dense(1, activation='sigmoid', kernel_initializer=GlorotNormal(seed=42), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42))(dense_layer)

    model = Model(inputs=[input_layer_consensus_sequence, input_layer_density_map, input_layer_numeric_features, input_structure_as_matrix], outputs=output_layer)

    lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['accuracy', F1Score(average='weighted', threshold=0.5, name='f1_score')])
    return model

def read_dataframes(paths):
    dfs = []
    for path in paths:
        dfs.append(pd.read_pickle(path))

    return pd.concat(dfs, axis=0)

def prepare_data(df):
    epsilon = 1e-7
    #From https://github.com/dhanush77777/DNA-sequencing-using-NLP/blob/master/DNA%20sequencing.ipynb
    df['consensus_sequence_kmers'] = df.apply(lambda x: build_kmers(x['consensus_sequence'], KMER_SIZE), axis=1)
    df['consensus_sequence_as_sentence'] = df.apply(lambda x: ' '.join(x['consensus_sequence_kmers']), axis=1)
    df['mature_vs_star_read_ratio'] = df.apply(lambda x: x['mature_read_count'] / (x['star_read_count'] + epsilon), axis=1)
    df['structure_as_1D_array'] = df.apply(lambda x: build_structure_1D(x['pri_struct'], x['mm_struct'], x['mm_offset']), axis=1)
    return df

def split_data(df):
    consensus_texts = df['consensus_sequence_as_sentence'].values.tolist()
    density_maps = df['read_density_map'].values.tolist()
    numeric_feature_names = ['mature_read_count', 'star_read_count', 'significant_randfold', 'mature_vs_star_read_ratio', 'estimated_probability', 'estimated_probability_uncertainty']
    numeric_features = df[numeric_feature_names]

    structure_as_1D_array = df['structure_as_1D_array'].values.tolist()
    y_data = df['false_positive'].values.astype(np.float32)

    # Split the data into training and temporary set (combined validation and test)
    X1_train, X1_tmp, X2_train, X2_tmp, X3_train, X3_tmp, X4_train, X4_tmp, Y_train, y_tmp = train_test_split(consensus_texts, density_maps, numeric_features, structure_as_1D_array, y_data, test_size=0.4, random_state=42)
    
    # Split the temporary set into validation and test sets
    X1_test, X1_val, X2_test, X2_val, X3_test, X3_val, X4_test, X4_val, Y_test, Y_val = train_test_split(X1_tmp, X2_tmp, X3_tmp, X4_tmp, y_tmp, test_size=0.5, random_state=42)

    X_train = np.asarray(X1_train), np.asarray(X2_train), np.asarray(X3_train), np.asarray(X4_train)
    X_val = [np.asarray(X1_val), np.asarray(X2_val), np.asarray(X3_val), np.asarray(X4_val)]
    X_test = [np.asarray(X1_test), np.asarray(X2_test), np.asarray(X3_test), np.asarray(X4_test)]
    return (X_train, np.asarray(Y_train), X_val, np.asarray(Y_val), X_test, np.asarray(Y_test))

#Best on test set (99.4%): batch_sizes = [16], nr_of_epochs = [8], model_sizes = [16], learning_rates = [0.0003], regularize = [False] (cheated though, because the hyperparameters were tuned against the test set)
#When max_val_f1_score was used the best parameters were: batch_sizes = [16], nr_of_epochs = [100], model_sizes = [64], learning_rates = [0.003], regularize = [True]
def generate_hyperparameter_combinations():
    batch_sizes = [16] #[1, 2, 4, 8, 16, 32, 64, 128, 256] # 
    nr_of_epochs = [100] #[1, 2, 4, 8, 16] # 
    model_sizes = [64] #[8, 16, 32, 64, 128, 256, 512, 1024, 2048] #
    learning_rates = [0.003] #[0.03, 0.003, 0.0003] # 
    regularize = [True] #[True, False] # 
    print(f'Will generate {len(batch_sizes) * len(nr_of_epochs) * len(model_sizes) * len(learning_rates) * len(regularize)} combinations of hyperparameters')
    parameters = list()
    for batch_size in batch_sizes:
        for epochs in nr_of_epochs:
            for model_size in model_sizes:
                for lr in learning_rates:
                    for reg in regularize:
                        parameters.append({'batch_size' : batch_size, 'epochs' : epochs, 'model_size'  : model_size, 'learning_rate' : lr, 'regularize' : reg})
    
    best_f1_score = 0
    lowest_val_loss = 9223372036854775807
    max_val_f1_score = 0
    #Resume grid search if there already are results
    if os.path.exists('train-results.csv'):
        already_run_parameters = list()
        
        with open('train-results.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            next(reader, None) #Skip header row
            for row in reader:
                already_run_parameters.append({'batch_size' : int(row[0]), 'epochs' : int(row[1]), 'model_size'  : int(row[2]), 'learning_rate' : float(row[3]), 'regularize' : row[4] == 'True'})
                f1_score = float(row[9])
                if  f1_score > best_f1_score:
                    best_f1_score = f1_score
                row_lowest_val_loss = float(row[10])
                if  row_lowest_val_loss < lowest_val_loss:
                    lowest_val_loss = row_lowest_val_loss
                row_max_val_f1_score = float(row[11])
                if  row_max_val_f1_score > max_val_f1_score:
                    max_val_f1_score = row_max_val_f1_score

        print(f'Removing {len(already_run_parameters)} parameter combinations already run')
        for parameter in already_run_parameters:
            if parameter in parameters:
                parameters.remove(parameter)
    else:
        print("Storing training results in train-results.csv")
        with open('train-results.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['batch_size', 'epochs', 'model_size', 'learning_rate', 'regularize', 'accuracy', 'loss', 'val_accuracy', 'val_loss', 'test_accuracy', 'test_F1-score', 'lowest_val_loss', 'max_val_f1_score'])

    return (parameters, best_f1_score, lowest_val_loss, max_val_f1_score)

def save_result_to_csv(parameters, metrics):
    with open('train-results.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([parameters['batch_size'], parameters['epochs'], parameters['model_size'], parameters['learning_rate'], parameters['regularize'] , metrics['history']['accuracy'][-1], metrics['history']['loss'][-1], metrics['history']['val_accuracy'][-1], metrics['history']['val_loss'][-1], metrics['test_accuracy'], metrics['test_F1-score'], metrics['lowest_val_loss'], metrics['max_val_f1_score']])

if __name__ == '__main__':
    df = read_dataframes(["resources/dataset/true_positives_TCGA_LUSC.pkl", "resources/dataset/false_positives_SRR2496781-84_bigger.pkl"])

    print("False positives:" + str(len(df[(df['false_positive']==True)])))
    print("True positives:" + str(len(df[(df['false_positive']==False)])))

    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(prepare_data(df))

    parameters, best_f1_score, stored_lowest_val_loss, stored_max_val_f1_score = generate_hyperparameter_combinations()

    best_model = None
    best_metrics = {'accuracy' : 0, 'test_F1-score' : best_f1_score, 'lowest_val_loss' : stored_lowest_val_loss, 'max_val_f1_score' : stored_max_val_f1_score}
    best_parameters = None
    for parameters in parameters:
        print("Parameters: " + str(parameters))
        
        model = get_model(consensus_sequences=X_train[0], density_maps=X_train[1], numeric_features=X_train[2], model_size=parameters['model_size'], initial_learning_rate=parameters['learning_rate'], batch_size = parameters['batch_size'], regularize=parameters['regularize'])
        
        early_stopping = EarlyStopping(monitor='val_f1_score', mode='max', patience=10, start_from_epoch=4, restore_best_weights=True, verbose=1)
        
        history = model.fit(X_train, Y_train, epochs=parameters['epochs'], batch_size=parameters['batch_size'], validation_data=(X_val, Y_val), callbacks=[early_stopping]) #verbose=0
        lowest_val_loss = min(history.history['val_loss'])
        max_val_f1_score = max(history.history['val_f1_score'])
        pred = model.predict(X_test)
        pred = (pred>=0.50) #If probability is equal or higher than 0.50, It's most likely a false positive (True)
        print(f'Test accuracy: {accuracy_score(Y_test,pred)}. Lowest val loss: {lowest_val_loss}. Max val F1-score: {max_val_f1_score}.')
        F1_score = f1_score(Y_test,pred)
        accuracy = accuracy_score(Y_test,pred)
        metrics = {'test_accuracy' : accuracy, 'test_F1-score' : F1_score, 'lowest_val_loss' : lowest_val_loss, 'max_val_f1_score' : max_val_f1_score, 'history' : history.history}
        save_result_to_csv(parameters, metrics)
        if max_val_f1_score > best_metrics['max_val_f1_score']:
            best_model = model
            best_parameters = parameters
            best_metrics = metrics
            best_model.save("best-model-not-seen-test.keras")

    print("Best parameters: " + str(best_parameters))
    print("Best metrics: " + str(best_metrics))
