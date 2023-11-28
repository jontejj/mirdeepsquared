import argparse
import sys

from mirdeepsquared.common import float_range, split_into_different_files


def parse_args(args):
    parser = argparse.ArgumentParser(prog='MirDeepSquared-split', description='Splits data in several pickle files into one train/val pickle file and one test (hold-out) file in different folders under pickle_output_path', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('path_to_pickle_files')  # positional argument
    parser.add_argument('pickle_output_path')  # positional argument
    parser.add_argument('-f', '--fraction', type=float_range(0, 1), help="Fraction of items to use for training, between 0 and 1", default=0.8)
    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])
    # args = parse_args(["resources/dataset/", "resources/dataset/split/"])
    split_into_different_files(args.path_to_pickle_files, args.pickle_output_path, args.fraction)


if __name__ == '__main__':
    main()
