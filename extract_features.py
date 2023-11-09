import os
import pprint
from numpy.core.multiarray import empty
import screed # a library for reading in FASTA/FASTQ
import pandas as pd
import re
import numpy as np
import argparse

def process_mrd_file(mrd_file):
    input_features = {}
    mrd = open(mrd_filepath, "r")
    pri_seq = ""
    pri_struct = ""
    mature_read_count = 0
    star_read_count = 0
    exp = ""
    location_name = ""
    read_density_map = np.zeros(112, dtype=np.int32)
    for x in mrd:
        #TODO: can also be cel-miR-38 etc...
        match_for_read = re.search(r"[A-Za-z]{3}_(\d*)_x(\d*)\s+([\.ucagUCAGN]*)\t\d*\n", x)
        if x.startswith(">"):
            location_name = x[1:-1] #Usually chromosome location
        if x.startswith("exp"):
            exp = re.sub(r"exp\s*", "", x)[0:-1]
        elif x.startswith("pri_seq"):
            pri_seq = re.sub(r"pri_seq\s*", "", x)[0:-1]
        elif x.startswith("pri_struct"):
            pri_struct = re.sub(r"pri_struct\s*", "", x)[0:-5]
        elif x.startswith("mature read count"):
            mature_read_count = int(x.replace("mature read count", ""))
        elif x.startswith("star read count"):
            star_read_count = int(x.replace("star read count", ""))
        elif match_for_read is not None:
            repeated_count = int(match_for_read.group(2))
            read_sequence = match_for_read.group(3)
            i = 0
            for c in read_sequence[:112]: #Truncate at 112
                if c != '.':
                    read_density_map[i] += repeated_count
                i+=1
        elif x == "\n" and location_name not in input_features:
            input_features[location_name] = { "pri_seq" : pri_seq, 
                                                    "pri_struct" : pri_struct, 
                                                    "exp" : exp,
                                                    "mature_read_count": mature_read_count, 
                                                    "star_read_count":star_read_count,
                                                    "read_density_map":read_density_map}
            read_density_map = np.zeros(112, dtype=np.int32)
    mrd.close()
    return input_features

def add_info_from_result_file(result_filepath, data_from_mrd):
    result_file = open(result_filepath, "r")

    is_novel = False
    started = False
    for x in result_file:
        if x.startswith("novel miRNAs predicted by miRDeep2"):
            started = True
            is_novel = True
        elif x.startswith("mature miRBase miRNAs detected by miRDeep2"):
            is_novel = False
        elif x.startswith("#miRBase miRNAs not detected by miRDeep2"):
            break
        elif not x.startswith("provisional") and not x.startswith("tag") and  not x.startswith("\n") and started:
            data_for_location = x.split('\t')
            location_name = data_for_location[0]
            estimated_probability = data_for_location[2].split(' ')
            data_from_mrd[location_name]["mirdeep_score"] = float(data_for_location[1])
            data_from_mrd[location_name]["estimated_probability"] = float(estimated_probability[0])
            data_from_mrd[location_name]["estimated_probability_uncertainty"] = float(estimated_probability[2][0:-1])
            data_from_mrd[location_name]["significant_randfold"] = 1 if (data_for_location[8] == 'yes') else 0
            data_from_mrd[location_name]["consensus_sequence"] = data_for_location[13]
            data_from_mrd[location_name]["predicted_as_novel"] = is_novel
            mm_offset = data_from_mrd[location_name]["pri_seq"].index(data_from_mrd[location_name]["consensus_sequence"])
            mm_struct = data_from_mrd[location_name]["pri_struct"][mm_offset: mm_offset + len(data_from_mrd[location_name]["consensus_sequence"])]
            data_from_mrd[location_name]["mm_struct"] = mm_struct
            data_from_mrd[location_name]["mm_offset"] = mm_offset
    #pp = pprint.PrettyPrinter(width=100, compact=True)
    #pp.pprint(input_features)
    #print("Unfiltered size: " + str(len(input_features)))
    result_file.close()

def classify_as_in_mirgene_db_or_not(mirgene_db_filepath, input_features):
    mirgene_sequences = set()
    for record in screed.open(mirgene_db_filepath):
        mirgene_sequences.add(record.sequence.lower())
        
    for key in input_features:
        part_of_sequence_is_in_mirgene_db = False
        sample_sequence = input_features[key]["pri_seq"].lower()
        for mirgene_sequence in mirgene_sequences:
            if mirgene_sequence in sample_sequence:
                part_of_sequence_is_in_mirgene_db = True
                break
        input_features[key]["in_mirgene_db"] = part_of_sequence_is_in_mirgene_db

def print_stats(input_features):

    novel_filtered_dict = {k: v for k, v in input_features.items() if "predicted_as_novel" in v and v["predicted_as_novel"]}
    print("Novel size: " + str(len(novel_filtered_dict)))

    filtered_dict = {k: v for k, v in input_features.items() if "predicted_as_novel" in v and v["predicted_as_novel"] == False}
    print("Mature size: " + str(len(filtered_dict)))

    filtered_dict = {k: v for k, v in input_features.items() if "predicted_as_novel" not in v}
    print("Neither novel nor mature size: " + str(len(filtered_dict)))

    filtered_dict = {k: v for k, v in input_features.items() if "predicted_as_novel" in v and v["predicted_as_novel"] == False and v["in_mirgene_db"] == False}
    print("Mature miRNA:s not in mirgene db: " + str(len(filtered_dict)))

    filtered_dict = {k: v for k, v in input_features.items() if "predicted_as_novel" in v and v["predicted_as_novel"] == True and v["in_mirgene_db"] == False}
    print("Novel miRNA:s not in mirgene db: " + str(len(filtered_dict)))

def convert_to_dataframe(input_features):
    input_features_as_lists_in_dict = {"location" : [], "pri_seq" : [],"pri_struct" : [], "exp" : [], "mature_read_count" : [], "star_read_count" : [], "estimated_probability" : [], "estimated_probability_uncertainty" : [], "significant_randfold" : [], "consensus_sequence" : [], "predicted_as_novel" : [], "mm_struct" : [], "mm_offset" : [], "in_mirgene_db" : [], "read_density_map" : []}
    for location, values in input_features.items():
        if 'predicted_as_novel' in values: #Ignore entries not in result.csv
            input_features_as_lists_in_dict['location'].append(location)
            input_features_as_lists_in_dict['pri_seq'].append(values['pri_seq'])
            input_features_as_lists_in_dict['pri_struct'].append(values['pri_struct'])
            input_features_as_lists_in_dict['exp'].append(values['exp'])
            input_features_as_lists_in_dict['mature_read_count'].append(values['mature_read_count'])
            input_features_as_lists_in_dict['star_read_count'].append(values['star_read_count'])
            input_features_as_lists_in_dict['estimated_probability'].append(values['estimated_probability'])
            input_features_as_lists_in_dict['estimated_probability_uncertainty'].append(values['estimated_probability_uncertainty'])
            input_features_as_lists_in_dict['significant_randfold'].append(values['significant_randfold'])
            input_features_as_lists_in_dict['consensus_sequence'].append(values['consensus_sequence'])
            input_features_as_lists_in_dict['predicted_as_novel'].append(values['predicted_as_novel'])
            input_features_as_lists_in_dict['mm_struct'].append(values['mm_struct'])
            input_features_as_lists_in_dict['mm_offset'].append(values['mm_offset'])
            input_features_as_lists_in_dict['in_mirgene_db'].append(values['in_mirgene_db'])
            input_features_as_lists_in_dict['read_density_map'].append(values['read_density_map'])

    return pd.DataFrame.from_dict(input_features_as_lists_in_dict)

def extract_features(mrd_filepath, result_filepath, false_positives, labels, true_positives, mirgene_db_file, pickle_output_file):
    input_features = process_mrd_file(mrd_filepath)
    add_info_from_result_file(result_filepath, input_features)
    classify_as_in_mirgene_db_or_not(mirgene_db_file, input_features)
    print_stats(input_features)
    df = convert_to_dataframe(input_features)

    if false_positives:
        false_positive_list = set(df['location'].values)
    elif true_positives:
        false_positive_list = set()
    elif labels != None:
        with open(labels) as file:
            false_positive_list = set(line.rstrip() for line in file)
    else:
        raise ValueError("Expected either of -fp, -tp or -l to be supplied. Unable to label data as false positives")

    df['false_positive'] = df.apply(lambda x: x['location'] in false_positive_list, axis=1)

    #print(df[df['location'].str.contains('chrII:11534525-11540624_19')])
    only_relevant_data = None
    if true_positives:
        only_relevant_data = df.loc[(df['predicted_as_novel'] == False) & (df['in_mirgene_db'] == True)]
    else:
        only_relevant_data = df.loc[df['predicted_as_novel'] == True]

    only_relevant_data.to_pickle(pickle_output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='MirDeepSquared-preprocessor', description='Extracts features from result.csv and output.mrd from MiRDeep2 and puts them in dataframes in pickle files')

    parser.add_argument('result_csv') # positional argument
    parser.add_argument('output_mrd') # positional argument
    parser.add_argument('pickle_output_file') # positional argument
    parser.add_argument('-fp', '--false_positives', action='store_true', help="Treat all novel classifications as false positives")
    parser.add_argument('-tp', '--true_positives', action='store_true', help="Treat all mature classifications as true positives")
    parser.add_argument('-l', '--labels', help="Manual file with curated false positives in it")
    parser.add_argument('-m', '--mirgene_db_file', help="File to filter results with")

    args = parser.parse_args()
    mrd_filepath = args.output_mrd
    result_filepath = args.result_csv
    false_positives = args.false_positives
    labels = args.labels
    true_positives = args.true_positives
    mirgene_db_file = args.mirgene_db_file
    pickle_output_file = args.pickle_output_file

    extract_features(mrd_filepath, result_filepath, false_positives, labels, true_positives, mirgene_db_file, pickle_output_file)
    