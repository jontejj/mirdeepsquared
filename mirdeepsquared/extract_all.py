import os

if __name__ == '__main__':
    #True positives
    os.system("python extract_features.py /Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/TCGA-LUSC/result_19_01_2023_t_23_35_49.csv /Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/TCGA-LUSC/output.mrd resources/dataset/true_positives_TCGA_LUSC.pkl -m resources/known-mature-sequences-h_sapiens.fas -tp -l resources/true_positives/invalid_mirna_marked_as_known.txt")
    #False positives
    os.system("python extract_features.py resources/false_positives/result_08_11_2023_t_19_35_00.csv resources/false_positives/08_11_2023_t_19_35_00_output.mrd resources/dataset/false_positives_SRR2496781-84_bigger.pkl -fp")
    #Zebrafish
    os.system("python mirdeepsquared/extract_features.py resources/zebrafish/zebrafish_result_13_11_2023_t_18_47_00.csv resources/zebrafish/zebrafish_13_11_2023_t_18_47_00_output.mrd resources/dataset/other_species/zebrafish_mature.pkl -tp -m resources/zebrafish/dre.fas")
