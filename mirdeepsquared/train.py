import os
import argparse
import sys
from mirdeepsquared.common import KMER_SIZE, NUCLEOTIDE_NR, list_of_pickle_files_in, prepare_data, read_dataframes, to_xy_with_location

import numpy as np

from keras.initializers import HeNormal, GlorotNormal, RandomNormal
from keras.layers import Input, Embedding, Flatten, Dense, TextVectorization, Conv1D, GlobalMaxPooling1D, Concatenate, Normalization, Reshape, Dropout, LSTM, Bidirectional 
from keras.constraints import MaxNorm
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
from keras.metrics import F1Score
from keras.callbacks import EarlyStopping

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold

import csv
import yaml

def get_model(consensus_sequences, density_maps, numeric_features, model_size = 64, initial_learning_rate = 0.0003, batch_size = 6, regularize = True, dropout_rate=0.8, weight_constraint=3.0):
    max_features = pow(NUCLEOTIDE_NR, KMER_SIZE)
    seq_length = len(max(consensus_sequences, key=len))

    #Input 1 - consensus_sequence
    #TODO: use precursor sequence with LSTM instead?
    input_layer_consensus_sequence = Input(shape=(1,), dtype='string', name='consensus_sequence')
    vectorize_layer = TextVectorization(output_mode="int", input_shape=(1,))
    vectorized_layer = vectorize_layer(input_layer_consensus_sequence)
    vectorize_layer.adapt(consensus_sequences)
    embedding_layer = Embedding(input_dim=max_features, output_dim=model_size, input_length=seq_length)(vectorized_layer)
    conv1d_layer = Conv1D(filters=model_size, kernel_size=3, activation='relu')(embedding_layer)
    consensus_maxpooling_layer = GlobalMaxPooling1D()(conv1d_layer)

    #batch_norm_layer = BatchNormalization(trainable=True)(maxpooling_layer) #TODO: remember to set trainable=False when inferring
    
    #Input 2 - Location of mature, star and hairpin sequences
    input_location_of_mature_star_and_hairpin = Input(shape=(111,4), dtype='float32', name='location_of_mature_star_and_hairpin')

    #Input 3 - density maps
    input_layer_density_map = Input(shape=(111,), dtype='int32', name='density_map_rate_of_change')
    density_map_normalizer_layer = Normalization(mean=np.mean(density_maps, axis=0), variance=np.var(density_maps, axis=0))(input_layer_density_map)

    density_map_reshaped_as_rows = Reshape((111,1), input_shape=(111,))(density_map_normalizer_layer)

    concatenated_2_3 = Concatenate(axis=-1)([input_location_of_mature_star_and_hairpin, density_map_reshaped_as_rows])
    flatten_layer_2_3 = Flatten()(concatenated_2_3)

    density_map_dense = Dense(model_size * 32, activation='relu')(flatten_layer_2_3)

    #Input 4 - structural information
    input_structure_as_matrix = Input(shape=(111,), dtype='float32', name='structure_as_1D_array')
    structure_embedding = Embedding(input_dim=17, output_dim=(128), input_length=111, mask_zero=True)(input_structure_as_matrix)
    bidirectional_lstm = Bidirectional(LSTM(128))(structure_embedding)
    structure_dense = Dense(model_size * 32, activation='relu')(bidirectional_lstm)

    #Input 5 - numerical features
    input_layer_numeric_features = Input(shape=(4,), dtype='float32', name='numeric_features')
    normalizer_layer = Normalization()
    normalizer_layer.adapt(numeric_features)
    numeric_features_dense = Dense(model_size * 4, activation='relu')(normalizer_layer(input_layer_numeric_features))

    concatenated = Concatenate()([consensus_maxpooling_layer, density_map_dense, numeric_features_dense, structure_dense])

    if regularize:
        dense_layer = Dense(model_size, activation='relu', kernel_constraint=MaxNorm(weight_constraint), kernel_initializer=HeNormal(seed=42), kernel_regularizer='l1_l2', use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42), bias_regularizer='l2')(concatenated)
        dropout_layer = Dropout(dropout_rate, input_shape=(model_size,))(dense_layer)
        output_layer = Dense(1, activation='sigmoid', kernel_initializer=GlorotNormal(seed=42), kernel_regularizer='l1_l2', bias_regularizer='l2', use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42))(dropout_layer)
    else:
        dense_layer = Dense(model_size, activation='relu', kernel_constraint=MaxNorm(weight_constraint), kernel_initializer=HeNormal(seed=42), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42))(concatenated)
        dropout_layer = Dropout(dropout_rate, input_shape=(model_size,))(dense_layer)
        output_layer = Dense(1, activation='sigmoid', kernel_initializer=GlorotNormal(seed=42), use_bias=True, bias_initializer=RandomNormal(mean=0.0, stddev=0.5, seed=42))(dropout_layer)

    model = Model(inputs=[input_layer_consensus_sequence, input_location_of_mature_star_and_hairpin, input_layer_density_map, input_structure_as_matrix, input_layer_numeric_features], outputs=output_layer)

    lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['accuracy', F1Score(average='weighted', threshold=0.5, name='f1_score')])
    return model

#Best on test set (99.4%): batch_sizes = [16], nr_of_epochs = [8], model_sizes = [16], learning_rates = [0.0003], regularize = [False] (cheated though, because the hyperparameters were tuned against the test set)
#When max_val_f1_score was used the best parameters were: batch_sizes = [16], nr_of_epochs = [100], model_sizes = [64], learning_rates = [0.003], regularize = [True]
def generate_hyperparameter_combinations(hyperparameter_file, train_results_file):
    print("Reading hyperparameters from: " + hyperparameter_file)
    with open(hyperparameter_file, 'r') as file:
        hyperparameters = yaml.safe_load(file)
    batch_sizes = hyperparameters['batch_sizes']
    nr_of_epochs = hyperparameters['nr_of_epochs']
    model_sizes = hyperparameters['model_sizes']
    learning_rates = hyperparameters['learning_rates']
    regularize = hyperparameters['regularize']
    dropout_rates = hyperparameters['dropout_rates']
    weight_constraints = hyperparameters['weight_constraints']
    print(f'Will generate {len(batch_sizes) * len(nr_of_epochs) * len(model_sizes) * len(learning_rates) * len(regularize) * len(dropout_rates) * len(weight_constraints)} combinations of hyperparameters')
    parameters = list()
    for batch_size in batch_sizes:
        for epochs in nr_of_epochs:
            for model_size in model_sizes:
                for lr in learning_rates:
                    for reg in regularize:
                        for dropout in dropout_rates:
                            for weight_constraint in weight_constraints:
                                parameters.append({'batch_size' : batch_size, 'epochs' : epochs, 'model_size'  : model_size, 'learning_rate' : lr, 'regularize' : reg, 'dropout_rate' : dropout, 'weight_constraint' : weight_constraint})

    best_mean_max_val_f1_score = 0
    #Resume grid search if there already are results
    if os.path.exists(train_results_file):
        already_run_parameters = list()
        
        with open(train_results_file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            next(reader, None) #Skip header row
            for row in reader:
                already_run_parameters.append({'batch_size' : int(row[0]), 'epochs' : int(row[1]), 'model_size'  : int(row[2]), 'learning_rate' : float(row[3]), 'regularize' : row[4] == 'True', 'dropout_rate' : float(row[5]), 'weight_constraint' : float(row[6])})
                row_mean_max_val_f1_score = float(row[13])
                if  row_mean_max_val_f1_score > best_mean_max_val_f1_score:
                    best_mean_max_val_f1_score = row_mean_max_val_f1_score

        print(f'Removing {len(already_run_parameters)} parameter combinations already run')
        for parameter in already_run_parameters:
            if parameter in parameters:
                parameters.remove(parameter)
    else:
        print("Storing training results in " + train_results_file)
        with open(train_results_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['batch_size', 'epochs', 'model_size', 'learning_rate', 'regularize', 'dropout_rate', 'weight_constraint', 'accuracy', 'loss', 'val_accuracy', 'val_loss', 'max_val_f1_score', 'best_epoch', 'mean_max_val_f1_score'])

    return (parameters, best_mean_max_val_f1_score)

def save_result_to_csv(parameters, metrics, train_results_file):
    history = metrics['history'].history
    with open(train_results_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([parameters['batch_size'], parameters['epochs'], parameters['model_size'], parameters['learning_rate'], parameters['regularize'], parameters['dropout_rate'], parameters['weight_constraint'] , history['accuracy'][-1], history['loss'][-1], history['val_accuracy'][-1], history['val_loss'][-1], metrics['max_val_f1_score'], metrics['best_epoch'], metrics['mean_max_val_f1_score']])

def train_main(dataset_path, model_output_path, hyperparameter_file, train_results_file, cross_validation_folds=2):
    df = read_dataframes(list_of_pickle_files_in(dataset_path))

    print("False positives:" + str(len(df[(df['false_positive']==True)])))
    print("True positives:" + str(len(df[(df['false_positive']==False)])))

    kfold = StratifiedKFold(n_splits=cross_validation_folds, shuffle=True, random_state=42)

    X, Y, _ = to_xy_with_location(prepare_data(df))

    parameters, best_mean_max_val_f1_score = generate_hyperparameter_combinations(hyperparameter_file, train_results_file)

    best_parameters = None
    best_metrics = None
    for parameters in parameters:
        print("Parameters: " + str(parameters))
        cv_history = []
        fold_best_max_val_f1_score = 0
        fold_best_model = None
        fold_nr = 0
        for train, test in kfold.split(X[0], Y):
            X_train = tuple(x[train] for x in X)
            X_val = tuple(x[test] for x in X)
            Y_train, Y_val = Y[train], Y[test]

            class_weights = compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
            class_weights_dict = dict(enumerate(class_weights))

            model = get_model(consensus_sequences=X_train[0], density_maps=X_train[2], numeric_features=X_train[4], model_size=parameters['model_size'], initial_learning_rate=parameters['learning_rate'], batch_size = parameters['batch_size'], regularize=parameters['regularize'], dropout_rate=parameters['dropout_rate'], weight_constraint = parameters['weight_constraint'])
            early_stopping = EarlyStopping(monitor='val_f1_score', mode='max', patience=10, start_from_epoch=4, restore_best_weights=True, verbose=1)
            
            history = model.fit(X_train, Y_train, epochs=parameters['epochs'], batch_size=parameters['batch_size'], class_weight=class_weights_dict, validation_data=(X_val, Y_val), callbacks=[early_stopping]) #verbose=0
            max_val_f1_score = max(history.history['val_f1_score'])
            print(f'Max val F1-score: {max_val_f1_score} (fold nr {fold_nr})')
            if max_val_f1_score > fold_best_max_val_f1_score:
                fold_best_model = model
                fold_best_max_val_f1_score = max_val_f1_score
            cv_history.append(history)
            fold_nr += 1
        max_val_f1_score = [max(history.history['val_f1_score']) for history in cv_history]
        mean_max_val_f1_score = np.mean(max_val_f1_score)
        best_fold_index = np.argmax(max_val_f1_score)
        best_epoch = np.argmax(cv_history[best_fold_index].history['val_f1_score']) + 1
        metrics = {'max_val_f1_score' : max(max_val_f1_score), 'mean_max_val_f1_score' : mean_max_val_f1_score, 'best_epoch': best_epoch, 'history' : cv_history[best_fold_index]}
        if mean_max_val_f1_score > best_mean_max_val_f1_score:
            best_mean_max_val_f1_score = mean_max_val_f1_score
            best_parameters = parameters
            best_metrics = metrics
            fold_best_model.save(model_output_path)
        print(f'Mean-max val F1-score: {mean_max_val_f1_score}')
        save_result_to_csv(parameters, metrics, train_results_file)

    print("Best parameters: " + str(best_parameters))
    print("Best metrics: " + str(best_metrics))

def parse_args(args):
    parser = argparse.ArgumentParser(prog='MirDeepSquared-train', description='Trains a deep learning model based on dataframes in pickle files', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('dataset_path', help="Path to the pickle files") # positional argument
    parser.add_argument('-o', '--output', help="Path where the model file will be saved", default="best-model.keras")
    parser.add_argument('-hp', '--hyperparameters', help="Path to the hyperparameter config file", default=os.path.join(os.path.dirname(__file__), 'default-hyperparameters.yaml'))
    parser.add_argument('-tr', '--train_results', help="Path to a file training results in it. Used to resume training if it is stopped", default='train-results.csv')
    parser.add_argument('-cvf', '--cross_validation_folds', help="Number of folds to use for cross-validation", default=2)

    return parser.parse_args(args)

def main():
    args = parse_args(sys.argv[1:])
    train_main(args.dataset_path, args.output, args.hyperparameters, args.train_results, args.cross_validation_folds)
    
    
if __name__ == '__main__':
    main()