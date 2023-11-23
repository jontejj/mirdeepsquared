import os

if __name__ == '__main__':
    # True positives
    os.system("python mirdeepsquared/extract_features.py /Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/TCGA-LUSC/result_19_01_2023_t_23_35_49.csv /Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/TCGA-LUSC/output.mrd resources/dataset/true_positives/true_positives_TCGA_LUSC_all.pkl -tp --section known")
    os.system("python mirdeepsquared/mirgene_db_filter.py resources/dataset/true_positives/true_positives_TCGA_LUSC_all.pkl resources/known-mature-sequences-h_sapiens.fas resources/dataset/true_positives_TCGA_LUSC_only_in_mirgene_db.pkl")
    # False positives
    os.system("python mirdeepsquared/extract_features.py resources/false_positives/result_08_11_2023_t_19_35_00.csv resources/false_positives/08_11_2023_t_19_35_00_output.mrd resources/dataset/false_positives_SRR2496781-84_bigger.pkl -fp --section novel")

    # Mouse
    os.system("python mirdeepsquared/extract_features.py /Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/mouse/result_20_11_2023_t_14_26_34.csv /Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/mouse/output.mrd resources/dataset/other_species/mouse.mature.pkl -tp --section known")
    os.system("python mirdeepsquared/mirgene_db_filter.py resources/dataset/other_species/mouse.mature.pkl resources/not_version_controlled/mouse/mmu.fas resources/dataset/other_species/mouse.mature_only_mirgene_db.pkl")

    # Zebrafish
    os.system("python mirdeepsquared/extract_features.py /Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/zebrafish/result_20_11_2023_t_14_11_15.csv /Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/zebrafish/output.mrd resources/dataset/other_species/zebrafish.mature.2nd.run.pkl -tp --section known")
    os.system("python mirdeepsquared/mirgene_db_filter.py resources/dataset/other_species/zebrafish.mature.2nd.run.pkl resources/zebrafish/dre.fas resources/dataset/other_species/zebrafish.mature.2nd.run_only_in_mirgene_db.pkl")

    # Zebrafish false positives
    os.system("python mirdeepsquared/extract_features.py /Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/zebrafish/result_20_11_2023_t_14_11_15.csv /Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/zebrafish/output.mrd resources/dataset/other_species/possibly_false_positives/zebrafish.novel_b_default.pkl -fp --section novel")
    os.system("python mirdeepsquared/extract_features.py /Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/zebrafish/b-5/result_22_11_2023_t_14_45_02.csv /Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/zebrafish/b-5/output.mrd resources/dataset/other_species/zebrafish.novel_b_5.pkl -fp --section novel")
    os.system("python subset.py resources/dataset/other_species/zebrafish.novel_b_5.pkl resources/dataset/other_species/possibly_false_positives/zebrafish.novel_b_default.pkl resources/dataset/false_positives_zebrafish_b_less_than_zero.pkl")

    # Mouse false positives
    os.system("python mirdeepsquared/extract_features.py /Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/mouse/result_20_11_2023_t_14_26_34.csv /Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/mouse/output.mrd resources/dataset/other_species/mouse.novel_b_default.pkl -fp --section novel")
    os.system("python mirdeepsquared/extract_features.py /Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/mouse/b-5/result_22_11_2023_t_14_28_31.csv /Volumes/Mac/Users/jonatanjoensson/school/molecular-biology/mirdeep2-data/mouse/b-5/output.mrd resources/dataset/other_species/mouse.novel_b_5.pkl -fp --section novel")
    os.system("python subset.py resources/dataset/other_species/mouse.novel_b_5.pkl resources/dataset/other_species/mouse.novel_b_default.pkl resources/dataset/false_positives_mouse_b_less_than_zero.pkl")

    os.system("python mirdeepsquared/correct_invalid_labels.py")
