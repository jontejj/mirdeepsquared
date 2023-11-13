import argparse
from extract_features import extract_features
from train import prepare_data
import numpy as np
from keras.saving import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='MirDeepSquared-predict', description='Classifies novel miRNA sequences either as false positive or not based on the result.csv and output.mrd files from MiRDeep2. Each row of the standard output represents the location name of the true positives')

    parser.add_argument('result_csv') # positional argument
    parser.add_argument('output_mrd') # positional argument
    #TODO: add batch-size as argument or automatically calculate it?
    args = parser.parse_args()
    mrd_filepath = args.output_mrd
    result_filepath = args.result_csv
    df = extract_features(mrd_filepath, result_filepath)
    df = prepare_data(df)
    novel_slice = df.loc[df['predicted_as_novel'] == True]
    if len(novel_slice) == 0:
        raise ValueError("No novel predictions in input files. Nothing to filter")
    X = np.asarray(novel_slice['read_density_map_percentage_change'].values.tolist())

    #TODO: use estimated_probability_uncertainty to decide which model to use (ensemble)?
    model = load_model("train-simple-model.keras") #load_model("best-model-not-seen-test.keras")
    pred = model.predict(X, verbose=0)
    pred = (pred>=0.50) #If probability is equal or higher than 0.50, It's most likely a false positive (True)
    [print(location) for location, pred in zip(novel_slice['location'], pred) if pred == False]
