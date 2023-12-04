import pandas as pd
# Note: remember to run split_dataset.py again after running this


def correct_label(file, location, correct_label):
    faulty_df = pd.read_pickle(file)
    faulty_df.loc[(faulty_df['location'] == location), 'false_positive'] = correct_label
    faulty_df.to_pickle(file)


if __name__ == '__main__':
    # chr1_1_1315 was reported as a true positive by a good model and it turned out to actually be a true positive?
    correct_label("resources/dataset/false_positives_SRR2496781-84_bigger.pkl", 'chr1_1_1315', False)
    # TODO: Is chr3_3_3199 (https://drive.google.com/file/d/1NHVvLKS6NTbdNszBcjoMbW473oC_s_JJ/view?usp=drive_link) a true positive?
    # chr12_12_57335 exists in mirgene db (hsa-mir-6502) but is not a true positive in the TCGA_LUSC datafile.
    # chr12_12_57336 is an example of an actual true positive
    correct_label("resources/dataset/true_positives_TCGA_LUSC_only_precursors_in_mirgene_db.pkl", 'chr12_12_57335', True)

    # Note: this can also be used for adversial training: by purposefully setting the wrong label on a sample, overfitting can be avoided / detected.
