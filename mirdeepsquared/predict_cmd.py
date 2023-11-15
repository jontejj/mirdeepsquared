#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys
import argparse

def parse_args(args):
    parser = argparse.ArgumentParser(prog='MirDeepSquared-predict', description='Classifies novel miRNA sequences either as false positive or not based on the result.csv and output.mrd files from MiRDeep2. Each row of the standard output represents the location name of the true positives', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('result_csv') # positional argument
    parser.add_argument('output_mrd') # positional argument
    parser.add_argument('-m', '--model', help="The trained .keras model file to use for the predictions", 
        default=os.path.join(os.path.dirname(__file__), 'train-simple-model.keras'))
    #TODO: add batch-size as argument or automatically calculate it?
    return parser.parse_args(args)

def main():
    args = parse_args(sys.argv[1:])
    #Avoid booting tensorflow until the correct params have been given
    from mirdeepsquared.predict import predict_main
    false_positives = predict_main(args)
    for false_positive in false_positives:
        print(false_positive)


if __name__ == '__main__':
    main()
