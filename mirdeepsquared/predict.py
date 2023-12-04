#!/usr/bin/env python3

import os
from mirdeepsquared.extract_features import extract_features
from mirdeepsquared.common import files_in, prepare_data, locations_in
from mirdeepsquared.motifs_bayes_model import MotifModel
from mirdeepsquared.train import BigModel
from mirdeepsquared.density_map_model import DensityMapModel
from mirdeepsquared.numerical_model import NumericalModel
from mirdeepsquared.structure_model import StructureModel
import numpy as np
import yaml

def cut_off(pred, threshold):
    # y_predicted = np.round(pred) (default)
    y_predicted = (pred > threshold).astype(int)
    return y_predicted


def predict_main(args):
    mrd_filepath = args.output_mrd
    result_filepath = args.result_csv
    df = extract_features(mrd_filepath, result_filepath)
    df = prepare_data(df)
    novel_slice = df.loc[df['predicted_as_novel'] == True]
    if len(novel_slice) == 0:
        raise ValueError("No novel predictions in input files. Nothing to filter")

    model_weights = model_weights_from_file(args.weights)

    # X = np.asarray(novel_slice['read_density_map_percentage_change'].values.tolist())

    return true_positives(args.models, novel_slice, args.threshold, model_weights)
    """
    mature_slice = df.loc[df['predicted_as_novel'] == False]
    if len(mature_slice) > 0:
        X = np.asarray(mature_slice['read_density_map_percentage_change'].values.tolist())
        pred = model.predict(X, verbose=0)
        pred = (pred>=0.50) #If probability is equal or higher than 0.50, It's most likely a false positive (True)
        [print(location) for location, pred in zip(mature_slice['location'], pred) if pred == True]
    """

def model_weights_from_file(model_weight_file):
    with open(model_weight_file, 'r') as file:
        model_weights = yaml.safe_load(file)
    return model_weights

# List of supported model class names
supported_classes = [MotifModel, BigModel, DensityMapModel, StructureModel, NumericalModel]

def map_filename_to_model(model_path):
    parts = os.path.basename(model_path).split('_')

    if len(parts) >= 2:
        class_name = parts[0]

        for model_class in supported_classes:
            if model_class.__name__ == class_name:
                model = model_class()
                model.load(model_path)
                return model

    raise ValueError(f'Unknown model type based on path: {model_path}, make sure you only have models in the model path provided')


def true_positives(model_path, df, threshold, model_weights):
    ensemble_predictions = predict(model_path, df, model_weights)

    # Convert the averaged predictions to binary predictions (0 or 1)
    pred = cut_off(ensemble_predictions, threshold)
    # pred = (ensemble_predictions >= 0.50)  # If probability is equal or higher than 0.50, It's most likely a false positive (True)
    locations = locations_in(df)

    return [location for location, pred in zip(locations, pred) if pred == False]


def predict(model_path, df, model_weights):
    models = [map_filename_to_model(model_file) for model_file in files_in(model_path)]
    pred_sums = np.zeros(len(df.values), dtype=np.float32)
    total_weights = 0
    for model in models:
        model_weight = model_weights[model.__class__.__name__]
        pred_sums += model_weight * model.predict(model.X(df))
        total_weights += model_weight

    # Ensemble by weighing predictions
    ensemble_predictions = pred_sums / total_weights
    return ensemble_predictions
