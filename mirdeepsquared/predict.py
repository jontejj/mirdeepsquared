#!/usr/bin/env python3

from .extract_features import extract_features
from .train import prepare_data
import numpy as np
from keras.saving import load_model

def predict_main(args):
    mrd_filepath = args.output_mrd
    result_filepath = args.result_csv
    df = extract_features(mrd_filepath, result_filepath)
    df = prepare_data(df)
    novel_slice = df.loc[df['predicted_as_novel'] == True]
    if len(novel_slice) == 0:
        raise ValueError("No novel predictions in input files. Nothing to filter")
    X = np.asarray(novel_slice['read_density_map_percentage_change'].values.tolist())

    model = load_model(args.model) #load_model("best-model-not-seen-test.keras")
    pred = model.predict(X, verbose=0)
    pred = (pred>=0.50) #If probability is equal or higher than 0.50, It's most likely a false positive (True)
    return [location for location, pred in zip(novel_slice['location'], pred) if pred == False]
    """
    mature_slice = df.loc[df['predicted_as_novel'] == False]
    if len(mature_slice) > 0:
        X = np.asarray(mature_slice['read_density_map_percentage_change'].values.tolist())
        pred = model.predict(X, verbose=0)
        pred = (pred>=0.50) #If probability is equal or higher than 0.50, It's most likely a false positive (True)
        [print(location) for location, pred in zip(mature_slice['location'], pred) if pred == True]
    """
