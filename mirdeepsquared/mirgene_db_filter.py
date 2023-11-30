import argparse
import sys
import pandas as pd
import screed
from mirdeepsquared.common import save_dataframe_to_pickle, extract_precursor_from_exp_and_pri_seq


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


def has_precursor_in_mirgene_db(exp, pri_seq, mirgene_db_sequences):
    precursor = extract_precursor_from_exp_and_pri_seq(exp, pri_seq)
    return precursor in mirgene_db_sequences


def filter_out_sequences_not_in_mirgene_db(df, mirgene_db_file, stringent):
    mirgene_db_sequences = read_in_mirgene_db_sequences(mirgene_db_file)
    if stringent:
        df['in_mirgene_db'] = df.apply(lambda x: has_precursor_in_mirgene_db(x['exp'], x['pri_seq'], mirgene_db_sequences), axis=1)
    else:
        df['in_mirgene_db'] = df.apply(lambda x: has_mirgene_db_sequence_in_it(x['pri_seq'].lower(), mirgene_db_sequences), axis=1)
    print_mirgene_db_stats(df)
    df = df.loc[(df['in_mirgene_db'] == True)]
    df = df.drop('in_mirgene_db', axis=1)
    return df


def print_mirgene_db_stats(df):
    print("'Known' sequences: " + str(len(df[(df['predicted_as_novel'] == False)])))
    print("'Known' sequences not in mirgene db: " + str(len(df[(df['predicted_as_novel'] == False) & (df['in_mirgene_db'] == False)])))


def parse_args(args):
    parser = argparse.ArgumentParser(prog='MirDeepSquared-mirgenedb', description='Creates a copy of a dataframe where only entries that have a pri_seq, containing a known miRNA sequence that is in a mirgene db file, are kept')

    parser.add_argument('pickle_file')  # positional argument
    parser.add_argument('mirgene_db_file')  # positional argument
    parser.add_argument('pickle_output_file')  # positional argument
    parser.add_argument('-s', '--stringent', action='store_true', help='With this flag, the mirgene file is expected to contain the whole precursor sequence instead of just the mature sequence')

    return parser.parse_args(args)


def main_mirgene_filter(pickle_file, mirgene_db_file, pickle_output_file, stringent):
    df = pd.read_pickle(pickle_file)
    df = filter_out_sequences_not_in_mirgene_db(df, mirgene_db_file, stringent)
    save_dataframe_to_pickle(df, pickle_output_file)


if __name__ == '__main__':
    # args = parse_args(["resources/dataset/true_positives/true_positives_TCGA_LUSC_all.pkl", "resources/ALL-precursors_in_mirgene_db.fas", "resources/dataset/true_positives_TCGA_LUSC_only_precursors_in_mirgene_db.pkl", "--stringent"])
    args = parse_args(sys.argv[1:])
    main_mirgene_filter(args.pickle_file, args.mirgene_db_file, args.pickle_output_file, args.stringent)
