import os
import pprint
import screed # a library for reading in FASTA/FASTQ
import pandas as pd
import re

def process_mrd_file(mrd_file):
    input_features = {}
    mrd = open(mrd_filepath, "r")
    pri_seq = ""
    pri_struct = ""
    mature_read_count = 0
    star_read_count = 0
    mm_offset = 0
    mm_count = 0
    exp = ""
    for x in mrd:
        if x.startswith(">"):
            location_name = x[1:-1] #Usually chromosome location
        if x.startswith("exp"):
            exp = re.sub(r"exp\s*", "", x)[0:-1]
        elif x.startswith("pri_seq"):
            pri_seq = re.sub(r"pri_seq\s*", "", x)[0:-1]
        elif x.startswith("pri_struct"):
            pri_struct = re.sub(r"pri_struct\s*", "", x)[0:-5]
            input_features[location_name] = { "pri_seq" : pri_seq, 
                                                    "pri_struct" : pri_struct, 
                                                    "exp" : exp,
                                                    "mature_read_count": mature_read_count, 
                                                    "star_read_count":star_read_count}

        elif x.startswith("mature read count"):
            mature_read_count = int(x.replace("mature read count", ""))
        elif x.startswith("star read count"):
            star_read_count = int(x.replace("star read count", ""))
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
              location_name = x.split('\t')[0]
              data_from_mrd[location_name]["consensus_sequence"] = x.split('\t')[13]
              data_from_mrd[location_name]["predicted_as_novel"] = is_novel
              mm_offset = data_from_mrd[location_name]["pri_seq"].index(data_from_mrd[location_name]["consensus_sequence"])
              mm_struct = data_from_mrd[location_name]["pri_struct"][mm_offset: mm_offset + len(data_from_mrd[location_name]["consensus_sequence"])]
              data_from_mrd[location_name]["mm_struct"] = mm_struct
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

def convert_to_dataframe(input_features, false_positive):
    input_features_as_lists_in_dict = {"location" : [], "pri_seq" : [],"pri_struct" : [], "exp" : [], "mature_read_count" : [], "star_read_count" : [], "consensus_sequence" : [], "predicted_as_novel" : [], "mm_struct" : [], "in_mirgene_db" : [], "false_positive" : []}
    for location, values in input_features.items():
        if 'predicted_as_novel' in values: #Ignore entries not in result.csv
            input_features_as_lists_in_dict['location'].append(location)
            input_features_as_lists_in_dict['pri_seq'].append(values['pri_seq'])
            input_features_as_lists_in_dict['pri_struct'].append(values['pri_struct'])
            input_features_as_lists_in_dict['exp'].append(values['exp'])
            input_features_as_lists_in_dict['mature_read_count'].append(values['mature_read_count'])
            input_features_as_lists_in_dict['star_read_count'].append(values['star_read_count'])
            input_features_as_lists_in_dict['consensus_sequence'].append(values['consensus_sequence'])
            input_features_as_lists_in_dict['predicted_as_novel'].append(values['predicted_as_novel'])
            input_features_as_lists_in_dict['mm_struct'].append(values['mm_struct'])
            input_features_as_lists_in_dict['in_mirgene_db'].append(values['in_mirgene_db'])
            input_features_as_lists_in_dict['false_positive'].append(false_positive)

    return pd.DataFrame.from_dict(input_features_as_lists_in_dict)

if __name__ == '__main__':

    mirgene_db_known_human_mature_filepath = "resources/known-mature-sequences-h_sapiens.fas"
    #mrd_filepath = "resources/output.mrd"
    #result_filepath = "resources/result_30_10_2023_t_15_05_15.csv"
    false_positive = False
    mrd_filepath = "/Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/TCGA-LUSC/output.mrd" #false_positive = False
    result_filepath = "/Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/TCGA-LUSC/result_19_01_2023_t_23_35_49.csv"

    #TODO: read in healthy dataset and set false_positive = True

    input_features = process_mrd_file(mrd_filepath)
    add_info_from_result_file(result_filepath, input_features)
    classify_as_in_mirgene_db_or_not(mirgene_db_known_human_mature_filepath, input_features)
    print_stats(input_features)
    df = convert_to_dataframe(input_features, false_positive)

    #print(df[df['location'].str.contains('chrII:11534525-11540624_19')])
    if not false_positive:
        only_relevant_data = df.loc[(df['predicted_as_novel'] == False) & (df['in_mirgene_db'] == True)]
        only_relevant_data.to_pickle("not_false_positives_small.pkl")
    else:
        only_relevant_data = df.loc[df['predicted_as_novel'] == True]
        only_relevant_data.to_pickle("false_positives_small.pkl")
    