#Functions that don't require tensorflow can be placed here. This avoids long boot times for programs that don't use tensorflow
import os
import pandas as pd
import screed # a library for reading in FASTA/FASTQ
import glob
import numpy as np

KMER_SIZE = 6
NUCLEOTIDE_NR = 5 #U C A G D (D for Dummy)
EPSILON = 1e-7

def build_kmers(sequence, ksize):
    kmers = []
    n_kmers = len(sequence) - ksize + 1

    for i in range(n_kmers):
        kmer = sequence[i:i + ksize]
        kmers.append(kmer)

    return kmers

def read_kmers_from_file(filename, ksize):
    all_kmers = []
    with screed.open(filename) as seqfile:
        for record in seqfile:
            sequence = record.sequence
            kmers = build_kmers(sequence, ksize)
            all_kmers += kmers
    return all_kmers

def kmers_from_list(list, ksize):
    all_kmers = []
    for sequence in list:
        kmers = build_kmers(sequence, ksize)
        all_kmers += kmers

    return all_kmers

def build_structure_1D(pri_struct, mm_struct, mm_offset, exp):
    pri_struct_padded = pri_struct.ljust(111, '-')
    pri_struct_truncated = pri_struct_padded[:111]

    exp_padded = exp.ljust(111, 'f')
    exp_truncated = exp_padded[:111]

    # Defines a vocabalary index for structural information, S = Star, l = hairpin, M = Mature
    char_mappings = {}
    char_mappings['f'] = {'-': 0, '.': 1, '(': 2, ')': 3}
    char_mappings['S'] = {'-': 4, '.': 5, '(': 6, ')': 7}
    char_mappings['l'] = {'-': 8, '.': 9, '(': 10, ')': 11}
    char_mappings['M'] = {'-': 12, '.': 13, '(': 14, ')': 15}

    merged_structure_information = [char_mappings[x][pri_struct_truncated[ind]] for ind, x in enumerate(exp_truncated)]
    return merged_structure_information

def list_of_pickle_files_in(path):
    return glob.glob(path + "/*.pkl")

def read_dataframes(paths):
    dfs = []
    for path in paths:
        df = pd.read_pickle(path)
        df['source_pickle'] = os.path.basename(path)
        dfs.append(df)

    return pd.concat(dfs, axis=0)

def calc_percentage_change(numbers):
    #np.diff(numbers) = rate of change
    data_no_zeros = np.where(numbers == 0, EPSILON, numbers)
    percentage_change = np.diff(numbers) / data_no_zeros[:-1] * 100
    return percentage_change

"""
Converts 'fffffffffffffffffffffffffffffffSSSSSSSSSSSSSSSSSSSSSSSllllllllllllllllMMMMMMMMMMMMMMMMMMMMMMffffffffffffffffff' to an array like:
          00000000000000000000000000000001111111111111111111111122222222222222223333333333333333333333000000000000000000
"""
def encode_exp(exp):
    exp_padded = exp.ljust(111, 'f')
    exp_truncated = exp_padded[:111]

    char_mapping = {'f': 0, 'S': 1, 'l': 2, 'M': 3}
    indices = [char_mapping[char] for char in exp_truncated]
    one_hot_encoded = np.eye(len(char_mapping))[indices]
    return one_hot_encoded

def encode_precursor(precursor):
    precursor_padded = precursor.ljust(111, 'D')
    precursor_truncated = precursor_padded[:111]

    char_mapping = {'D' : 0, 'u': 1, 'g': 2, 'c': 3, 'a': 4}
    indices = [char_mapping[char] for char in precursor_truncated]
    one_hot_encoded = np.eye(len(char_mapping))[indices]
    return one_hot_encoded

def prepare_data(df):
    
    #From https://github.com/dhanush77777/DNA-sequencing-using-NLP/blob/master/DNA%20sequencing.ipynb
    df['consensus_sequence_kmers'] = df.apply(lambda x: build_kmers(x['consensus_sequence'], KMER_SIZE), axis=1)
    df['consensus_sequence_as_sentence'] = df.apply(lambda x: ' '.join(x['consensus_sequence_kmers']), axis=1)
    #TODO: create other features for mature vs star, such as:
    #feature_difference = feature1 - feature2
    #feature_interaction = feature1 * feature2
    #feature_log = np.log(feature1) or np.log(feature1) / np.log(feature2)
    df['mature_vs_star_read_ratio'] = df.apply(lambda x: x['mature_read_count'] / (x['star_read_count'] + EPSILON), axis=1)
    df['structure_as_1D_array'] = df.apply(lambda x: build_structure_1D(x['pri_struct'], x['mm_struct'], x['mm_offset'], x['exp']), axis=1)
    df['read_density_map_percentage_change'] = df.apply(lambda x: calc_percentage_change(x['read_density_map']), axis=1)
    df['location_of_mature_star_and_hairpin'] = df.apply(lambda x: encode_exp(x['exp']), axis=1)
    df['precursor_encoded'] = df.apply(lambda x: encode_precursor(x['pri_seq']), axis=1)
    return df

def split_data_holdout(df):
    train=df.sample(frac=0.8,random_state=42)
    holdout=df.drop(train.index)
    return (train, holdout)

def split_into_different_files(path_to_pickle_files, pickle_output_path):
    df = read_dataframes(list_of_pickle_files_in(path_to_pickle_files))
    train, holdout = split_data_holdout(df)
    os.makedirs(pickle_output_path)
    os.mkdir(pickle_output_path + "/train")
    os.mkdir(pickle_output_path + "/holdout")
    train.to_pickle(pickle_output_path + "/train/train.pkl")
    holdout.to_pickle(pickle_output_path + "/holdout/holdout.pkl")

def split_data(df):
    train=df.sample(frac=0.6,random_state=42)
    tmp=df.drop(train.index)
    val=tmp.sample(frac=0.5,random_state=42)
    test=tmp.drop(val.index)
    return (train, val, test)

def to_x_with_location(df):
    locations = df['location'].values.tolist()
    consensus_texts = np.asarray(df['consensus_sequence_as_sentence'].values.tolist())
    density_maps = np.asarray(df['read_density_map_percentage_change'].values.tolist())
    numeric_feature_names = ['mature_read_count', 'star_read_count', 'significant_randfold', 'mature_vs_star_read_ratio'] #, 'estimated_probability', 'estimated_probability_uncertainty'
    numeric_features = np.asarray(df[numeric_feature_names])

    structure_as_1D_array = np.asarray(df['structure_as_1D_array'].values.tolist())
    location_of_mature_star_and_hairpin = np.asarray(df['location_of_mature_star_and_hairpin'].values.tolist())
    precursors = np.asarray(df['precursor_encoded'].values.tolist())
    #TODO: add , precursors
    return ((consensus_texts, location_of_mature_star_and_hairpin, density_maps, structure_as_1D_array, numeric_features), locations)

def to_xy_with_location(df):
    X, locations = to_x_with_location(df)
    y_data = np.asarray(df['false_positive'].values.astype(np.float32))
    return (X, y_data, locations)
