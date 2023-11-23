import pandas as pd
import argparse
import sys
from mirdeepsquared.common import save_dataframe_to_pickle


def parse_args(args):
    parser = argparse.ArgumentParser(prog='MirDeepSquared-subset', description="Outputs the subset of entries in 'big_pickle_file' that are not in 'subset_to_remove_pickle_file' into the new file 'pickle_output_file'")

    parser.add_argument('big_pickle_file')  # positional argument
    parser.add_argument('subset_to_remove_pickle_file')  # positional argument
    parser.add_argument('pickle_output_file')  # positional argument

    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    #args = parse_args(["resources/dataset/other_species/mouse.novel_b_5.pkl", "resources/dataset/other_species/possibly_false_positives/mouse.novel.pkl", "resources/dataset/false_positives_mouse_b_less_than_zero.pkl"])
    big_dataset = pd.read_pickle(args.big_pickle_file)
    subset_to_remove = pd.read_pickle(args.subset_to_remove_pickle_file)
    subset = set(subset_to_remove['location'].values)
    smaller_dataset = big_dataset[~big_dataset.location.isin(subset)]
    print(f'Saving {len(smaller_dataset)} samples to {args.pickle_output_file}')
    save_dataframe_to_pickle(smaller_dataset, args.pickle_output_file)
