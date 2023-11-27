import os


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from mirdeepsquared.common import files_in, list_of_pickle_files_in, prepare_data, read_dataframes, split_data_once, Y_values
from mirdeepsquared.predict import map_filename_to_model, predict
import numpy as np
# from mirdeepsquared.train_simple_motifs import all_true_or_random


def balance_classes(df, target_column):
    # Find the minimum number of samples for any target label
    min_samples = df[target_column].value_counts().min()

    # Use groupby and apply to get a balanced DataFrame
    balanced_df = df.groupby(target_column).apply(lambda x: x.sample(min_samples)).reset_index(drop=True)

    return balanced_df


def binary_crossentropy(pred, Y):
    epsilon = 1e-15  # to avoid log(0) which is undefined
    pred = np.clip(pred, epsilon, 1 - epsilon)
    pred = np.clip(pred, 1e-7, 1 - 1e-7)
    return -np.mean(Y * np.log(pred) + (1 - Y) * np.log(1 - pred))


if __name__ == '__main__':
    path = "resources/dataset/split/holdout"
    # df = read_dataframes(list_of_pickle_files_in("resources/dataset/other_species/true_positives/zebrafish"))
    list_of_files = list_of_pickle_files_in(path)
    print("Predicting for samples in: " + str([os.path.basename(path) for path in list_of_files]))
    df = prepare_data(read_dataframes(list_of_files))  # easy/split/holdout
    print("False positives:" + str(len(df[(df['false_positive'] == True)])))
    print("True positives:" + str(len(df[(df['false_positive'] == False)])))

    models = [map_filename_to_model(model_file) for model_file in files_in("models/")]
    for model in models:
        Y_test = Y_values(df)
        print("Model: " + str(model))
        pred = model.predict(model.X(df))
        pred = np.round(pred)
        print("Test Confusion matrix:")
        print(confusion_matrix(Y_test, pred))
        print("Test Accuracy: " + str(accuracy_score(Y_test, pred)))
        print("Test F1-score: " + str(f1_score(Y_test, pred)))
        bce = binary_crossentropy(pred, Y_test)
        print("Binary Crossentropy:", bce)

    # Check validation scores
    path = "resources/dataset/split/train"
    # df = read_dataframes(list_of_pickle_files_in("resources/dataset/other_species/true_positives/zebrafish"))
    list_of_files = list_of_pickle_files_in(path)
    df = prepare_data(read_dataframes(list_of_files))
    train, val = split_data_once(prepare_data(df))
    Y_val = Y_values(val)
    for model in models:
        print("Model: " + str(model))
        pred = model.predict(model.X(val))
        pred = np.round(pred)
        print("Val Confusion matrix:")
        print(confusion_matrix(Y_val, pred))
        print("Val Accuracy: " + str(accuracy_score(Y_val, pred)))
        print("Val F1-score: " + str(f1_score(Y_val, pred)))
        bce = binary_crossentropy(pred, Y_val)
        print("Binary Crossentropy:", bce)

    print("Ensemble Model: ")
    ensemble_predictions = predict("models/", val)
    # Convert the averaged predictions to binary predictions (0 or 1)
    pred = np.round(ensemble_predictions)
    print("Val Confusion matrix:")
    print(confusion_matrix(Y_val, pred))
    print("Val Accuracy: " + str(accuracy_score(Y_val, pred)))
    print("Val F1-score: " + str(f1_score(Y_val, pred)))
    bce = binary_crossentropy(pred, Y_val)
    print("Binary Crossentropy:", bce)
"""
    #TODO: remove prepare_data if done above
    X_test, Y_test, locations_test = to_xy_with_location(prepare_data(df))
    # TODO: use estimated_probability_uncertainty to decide which model to use (ensemble)
    model = load_model("mirdeepsquared/train-simple-model.keras")
    # model = load_model("mirdeepsquared/train-simple-model-motifs.keras")
    # model = load_model("mirdeepsquared/train-simple-model-numerical-features.keras")
    # model = load_model("mirdeepsquared/train-simple-model-density-map.keras")
    # model = load_model("mirdeepsquared/train-simple-model-precursors.keras")
    pred = model.predict(X_test)
    # pred = [not all_true_or_random(x) for x in X_test[6]]  # X_test[1], X_test[2], X_test[5])  #
    pred = (pred >= 0.50)  # If probability is equal or higher than 0.50, It's most likely a false positive (True)
    print("Test Confusion matrix:")
    print(confusion_matrix(Y_test, pred))
    print("Test Accuracy: " + str(accuracy_score(Y_test, pred)))
    print("Test F1-score: " + str(f1_score(Y_test, pred)))
    for i in range(0, len(pred)):
        if pred[i] != Y_test[i]:
            source_pickle = df[(df['location'] == locations_test[i])]['source_pickle'].array[0]
            print(f'Predicted: {not pred[i]} positive for {locations_test[i]}, real is: {not bool(Y_test[i])} positive. Source pickle: {source_pickle}')
"""
