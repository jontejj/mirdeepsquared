Dataset files in this folder contains samples that have the wrong label. This can be used to avoid overfitting.

To create such a file the following command was used:
python extract_features.py /Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/TCGA-LUSC/result_19_01_2023_t_23_35_49.csv /Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/TCGA-LUSC/output.mrd resources/dataset/adversarial_training/invalid_labels_TCGA_LUSC.pkl -m resources/known-mature-sequences-h_sapiens.fas -tp -f resources/true_positives/invalid_mirna_marked_as_known.txt