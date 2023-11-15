#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys
import argparse

from .extract_features import extract_features
from .train import prepare_data
import numpy as np
from keras.saving import load_model

def parse_args(args):
    parser = argparse.ArgumentParser(prog='MirDeepSquared-predict', description='Classifies novel miRNA sequences either as false positive or not based on the result.csv and output.mrd files from MiRDeep2. Each row of the standard output represents the location name of the true positives', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('result_csv') # positional argument
    parser.add_argument('output_mrd') # positional argument
    parser.add_argument('-m', '--model', help="The trained .keras model file to use for the predictions", 
        default=os.path.join(os.path.dirname(__file__), 'train-simple-model.keras'))
    #TODO: add batch-size as argument or automatically calculate it?
    return parser.parse_args(args)

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

def main():
    args = parse_args(sys.argv[1:])
    false_positives = predict_main(args)
    for false_positive in false_positives:
        print(false_positive)

if __name__ == '__main__':
    main()
