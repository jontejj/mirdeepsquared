import argparse
import sys

from mirdeepsquared.common import split_into_different_files
def parse_args(args):
    parser = argparse.ArgumentParser(prog='MirDeepSquared-split', description='Splits data in several pickle files into one train/val pickle file and one test (hold-out) file in different folders under pickle_output_path')

    parser.add_argument('path_to_pickle_files') # positional argument
    parser.add_argument('pickle_output_path') # positional argument
    return parser.parse_args(args)

def main():
    args = parse_args(sys.argv[1:])
    split_into_different_files(args.path_to_pickle_files, args.pickle_output_path)

if __name__ == '__main__':
    main()