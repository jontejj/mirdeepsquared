import argparse
import os
from pathlib import Path
import shutil
import yaml

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from mirdeepsquared.common import files_in, float_range, list_of_pickle_files_in, locations_in, prepare_data, read_dataframes, Y_values
from mirdeepsquared.predict import cut_off, map_filename_to_model, model_weights_from_file, predict
import numpy as np


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


def parse_args(args):
    parser = argparse.ArgumentParser(prog='MirDeepSquared-test', description='Outputs accuracy scores for a model')

    parser.add_argument('dataset')  # positional argument
    parser.add_argument('model_path', help="The path to the trained .keras/.pkl model files to use for the predictions")  # positional argument
    parser.add_argument('--output_pdf', help="Where to put pdf:s for the wrong predictions", nargs='?', default=argparse.SUPPRESS)
    parser.add_argument('-t', '--threshold', type=float_range(0, 1), help="Threshold to use for determining if predictions are treated as true positives or not, between 0 and 1", default=0.5)
    return parser.parse_args(args)


def print_measurements_return_bce(Y_test, pred):
    print("Confusion matrix:")
    print(confusion_matrix(Y_test, pred))
    print("Accuracy: " + str(accuracy_score(Y_test, pred)))
    print("F1-score: " + str(f1_score(Y_test, pred)))
    bce = binary_crossentropy(pred, Y_test)
    print("Binary Crossentropy:", bce)
    return bce


def find_pdf(location, source_pickle):
    pdf_dirs = {'mouse.mature_only_mirgene_db.pkl': '/Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/mouse/pdfs_21_11_2023_t_09_54_51/',
                'mouse.mature.pkl': '/Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/mouse/pdfs_21_11_2023_t_09_54_51/',
                'zebrafish.mature.2nd.run_only_in_mirgene_db.pkl': '/Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/zebrafish/pdfs_20_11_2023_t_14_11_15/',
                'false_positives_SRR2496781-84_bigger.pkl': '/Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/SRR2496781-84/pdfs_08_11_2023_t_19_35_00/',
                'true_positives_TCGA_BRCA_only_precursors_in_mirgene_db.pkl': '/Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/TCGA-BRCA/pdfs_30_12_2022_t_12_51_40/',
                'true_positives_TCGA_LUSC_only_precursors_in_mirgene_db.pkl': '/Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/TCGA-LUSC/pdfs_19_01_2023_t_23_35_49/',
                'true_positives_TCGA_LIHC_only_precursors_in_mirgene_db.pkl': '/Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/TCGA-LIHC/pdfs_11_04_2023_t_12_39_19/',
                'tricky_true_positives_zebrafish.pkl': '/Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/zebrafish/pdfs_20_11_2023_t_14_11_15/',
                'tricky_true_positives_TCGA_BRCA.pkl': '/Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/TCGA-BRCA/pdfs_30_12_2022_t_12_51_40/',
                'tricky_true_positives_TCGA_LUSC.pkl': '/Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/TCGA-LUSC/pdfs_19_01_2023_t_23_35_49/',
                'tricky_true_positives_mouse.pkl': '/Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/mouse/pdfs_21_11_2023_t_09_54_51/',
                'false_positives_mouse_b_less_than_zero.pkl': '/Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/mouse/b-5/pdfs_22_11_2023_t_14_28_31/',
                'false_positives_zebrafish_b_less_than_zero.pkl': '/Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/zebrafish/b-5/pdfs_22_11_2023_t_14_45_02/'
                }
    try:
        return pdf_dirs[source_pickle] + location + ".pdf"
    except KeyError:
        print(f'No pdf found for {location} ({source_pickle})')
        return None


def output_pdf(location, source_pickle, pdf_path, actual_label):
    pdf = find_pdf(location, source_pickle)
    if pdf is not None:
        shutil.copyfile(pdf, pdf_path + "/" + str(int(actual_label)) + "_" + source_pickle + "_" + location + ".pdf")


if __name__ == '__main__':
    args = parse_args(['resources/dataset/split/holdout/', 'mirdeepsquared/models16/'])
    # args = parse_args(sys.argv[1:])
    path = args.dataset
    list_of_files = list_of_pickle_files_in(path)
    print("Predicting for samples in: " + str([os.path.basename(path) for path in list_of_files]))
    df = prepare_data(read_dataframes(list_of_files))
    print("False positives:" + str(len(df[(df['false_positive'] == True)])))
    print("True positives:" + str(len(df[(df['false_positive'] == False)])))

    Y_test = Y_values(df)
    locations_test = locations_in(df)

    models = [map_filename_to_model(model_file) for model_file in files_in(args.model_path)]

    problematic_samples = {location: 0 for location in locations_test}

    total_bce = 0
    future_model_weights = {}
    for model in models:
        print("Model: " + str(model))
        pred = model.predict(model.X(df))
        pred = cut_off(pred, args.threshold)
        bce = print_measurements_return_bce(Y_test, pred)
        future_model_weights[model.__class__.__name__] = bce
        total_bce += bce
        too_confident_count = 0
        for i in range(0, len(pred)):
            if pred[i] != Y_test[i]:
                if (pred[i] > 0.90 and Y_test[i] == 0) or (pred[i] < 0.10 and Y_test[i] == 1):
                    too_confident_count += 1
                problematic_samples[locations_test[i]] += 1
        print(f'Too confident for {too_confident_count} samples')

    # Save future model weights
    relative_future_model_weights = {model: float(total_bce / bce) for model, bce in future_model_weights.items()}
    with open('mirdeepsquared/future_model_weights.yaml', 'w') as outfile:
        yaml.dump(relative_future_model_weights, outfile)

    if hasattr(args, 'output_pdf'):
        Path(args.output_pdf + "/ensemble").mkdir(parents=True, exist_ok=True)
        Path(args.output_pdf + "/most_problematic").mkdir(parents=True, exist_ok=True)

    print("Most problematic samples: ")
    most_problematic_samples = dict(sorted(problematic_samples.items(), key=lambda item: item[1], reverse=True))
    count = 0
    for key, value in most_problematic_samples.items():
        if value > 0:
            source_pickle = df[(df['location'] == key)]['source_pickle'].array[0]
            print(f'{key}: {value}. Source pickle: {source_pickle}')
            if hasattr(args, 'output_pdf'):
                output_pdf(key, source_pickle, args.output_pdf + "/most_problematic", df[(df['location'] == key)]['false_positive'].array[0])
            count += 1
            if count == 20:
                break
    print(f'Printed {count} of the most problematic samples')

    print("Ensemble Model: ")
    model_weights = model_weights_from_file("mirdeepsquared/model_weights.yaml")
    ensemble_predictions = predict(args.model_path, df, model_weights)
    # Convert the averaged predictions to binary predictions (0 or 1)
    pred = cut_off(ensemble_predictions, args.threshold)
    print_measurements_return_bce(Y_test, pred)

    print("Wrong predictions for ensemble model:")
    for i in range(0, len(pred)):
        if pred[i] != Y_test[i]:
            source_pickle = df[(df['location'] == locations_test[i])]['source_pickle'].array[0]
            print(f'Predicted: {not pred[i]} positive for {locations_test[i]}, real is: {not bool(Y_test[i])} positive. Source pickle: {source_pickle}')
            if hasattr(args, 'output_pdf'):
                output_pdf(locations_test[i], source_pickle, args.output_pdf + "/ensemble", Y_test[i])
