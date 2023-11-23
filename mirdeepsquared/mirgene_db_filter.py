import argparse
import sys
import pandas as pd
import screed
from mirdeepsquared.common import save_dataframe_to_pickle


def read_in_mirgene_db_sequences(mirgene_db_filepath):
    mirgene_sequences = set()
    with screed.open(mirgene_db_filepath) as seqfile:
        for record in seqfile:
            mirgene_sequences.add(record.sequence.lower())
    return mirgene_sequences


def has_mirgene_db_sequence_in_it(sequence, mirgene_sequences):
    for mirgene_sequence in mirgene_sequences:
        if mirgene_sequence in sequence:
            return True
    return False


def filter_out_sequences_not_in_mirgene_db(df, mirgene_db_file):
    mirgene_db_sequences = read_in_mirgene_db_sequences(mirgene_db_file)
    df['in_mirgene_db'] = df.apply(lambda x: has_mirgene_db_sequence_in_it(x['pri_seq'].lower(), mirgene_db_sequences), axis=1)
    print_mirgene_db_stats(df)
    df = df.loc[(df['in_mirgene_db'] == True)]
    df = df.drop('in_mirgene_db', axis=1)
    return df


def print_mirgene_db_stats(df):
    print("Novel sequences not in mirgene db: " + str(len(df[(df['predicted_as_novel'] == True) & (df['in_mirgene_db'] == False)])))
    print("Mature sequences not in mirgene db: " + str(len(df[(df['predicted_as_novel'] == False) & (df['in_mirgene_db'] == False)])))


def parse_args(args):
    parser = argparse.ArgumentParser(prog='MirDeepSquared-mirgenedb', description='Creates a copy of a dataframe where only entries that have a pri_seq, containing a known miRNA sequence that is in a mirgene db file, are kept')

    parser.add_argument('pickle_file')  # positional argument
    parser.add_argument('mirgene_db_file')  # positional argument
    parser.add_argument('pickle_output_file')  # positional argument

    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])

    df = pd.read_pickle(args.pickle_file)
    df = filter_out_sequences_not_in_mirgene_db(df, args.mirgene_db_file)
    save_dataframe_to_pickle(df, args.pickle_output_file)
