import sys
import pandas as pd
import re
import numpy as np
import argparse
from mirdeepsquared.common import save_dataframe_to_pickle


def extract_features(mrd_filepath, result_filepath):
    input_features = process_mrd_file(mrd_filepath)
    add_info_from_result_file(result_filepath, input_features)
    return convert_to_dataframe(input_features)


def process_mrd_file(mrd_filepath):
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
        # TODO: can also be cel-miR-38 etc... (but how many reads are there for those?)
        match_for_read = re.search(r"[A-Za-z0-9]{3}_(\d*)_x(\d*)\s+([\.ucagUCAGN]*)\t\d*\n", x)
        if x.startswith(">"):
            location_name = x[1:-1]  # Usually chromosome location
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
            for c in read_sequence[:112]:  # Truncate at 112
                # TODO: how should capital letters be handled? lowercase letters might be used to denote low-confidence?
                # Should high confidence count as twice the amount of reads?
                if c != '.':
                    read_density_map[i] += repeated_count
                i += 1
        elif x == "\n" and location_name not in input_features:
            input_features[location_name] = {"pri_seq": pri_seq,
                                             "pri_struct": pri_struct,
                                             "exp": exp,
                                             "mature_read_count": mature_read_count,
                                             "star_read_count": star_read_count,
                                             "read_density_map": read_density_map}
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
        elif not x.startswith("provisional") and not x.startswith("tag") and not x.startswith("\n") and started:
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
    result_file.close()


def convert_to_dataframe(input_features):
    input_features_as_lists_in_dict = {"location": [], "pri_seq": [], "pri_struct": [], "exp": [], "mature_read_count": [], "star_read_count": [], "estimated_probability": [], "estimated_probability_uncertainty": [], "significant_randfold": [], "consensus_sequence": [], "predicted_as_novel": [], "mm_struct": [], "mm_offset": [], "read_density_map": []}
    ignored_entries = 0
    for location, values in input_features.items():
        if 'predicted_as_novel' in values:  # Ignore entries not in result.csv
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
            input_features_as_lists_in_dict['read_density_map'].append(values['read_density_map'])
        else:
            ignored_entries += 1
    # TODO: enable this printout?
    # print(f'{ignored_entries} sequences were not in the result.csv file, ignoring them')
    return pd.DataFrame.from_dict(input_features_as_lists_in_dict)


def print_basic_stats(df):
    print("Novel sequences: " + str(len(df[(df['predicted_as_novel'] == True)])))
    print("Mature sequences: " + str(len(df[(df['predicted_as_novel'] == False)])))


def label_false_positive(df, false_positives, true_positives, section):
    df['false_positive'] = false_positives

    # print(df[df['location'].str.contains('chrII:11534525-11540624_19')])
    only_relevant_data = df
    if section == 'known':
        only_relevant_data = df.loc[(df['predicted_as_novel'] == False)]
    elif section == 'novel':
        only_relevant_data = df.loc[df['predicted_as_novel'] == True]
    return only_relevant_data


def parse_args(args):
    parser = argparse.ArgumentParser(prog='MirDeepSquared-preprocessor', description='Extracts features from result.csv and output.mrd from MiRDeep2 and puts them in dataframes in pickle files', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('result_csv')  # positional argument
    parser.add_argument('output_mrd')  # positional argument
    parser.add_argument('pickle_output_file')  # positional argument
    parser.add_argument('-s', '--section', default='both', help="Which section to extract features from, can be known, novel or both")
    parser.add_argument('-fp', '--false_positives', action='store_true', help="Treat miRNA:s as false positives")
    parser.add_argument('-tp', '--true_positives', action='store_true', help="Treat miRNA:s as true positives")
    return parser.parse_args(args)


def extract_features_main(args):
    mrd_filepath = args.output_mrd
    result_filepath = args.result_csv
    false_positives = args.false_positives
    true_positives = args.true_positives
    section = args.section

    if not false_positives and not true_positives:
        raise Exception("Either -fp or -tp must be specified")

    df = extract_features(mrd_filepath, result_filepath)
    print_basic_stats(df)
    df = label_false_positive(df, false_positives, true_positives, section)
    return df


def main():
    args = parse_args(sys.argv[1:])
    df = extract_features_main(args)
    save_dataframe_to_pickle(df, args.pickle_output_file)


if __name__ == '__main__':
    main()
