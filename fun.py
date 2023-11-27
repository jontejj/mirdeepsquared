import pandas as pd
from mirdeepsquared.common import find_motifs, list_of_pickle_files_in, read_dataframes

if __name__ == '__main__':
    # star_first_motifs = find_motifs('ffffffffffffffffffffffffffffffSSSSSSSSSSSSSSSSSSSSSSlllllllllllllllllMMMMMMMMMMMMMMMMMMMMMMMffffffffffffffffffffffffffffff', 'CCCAAGGUGGGCGGGCUGGGCGGGGGCCCUCGUCUUACCCAGCAGUGUUUGGGUGCGGUUGGGAGUCUCUAAUACUGCCGGGUAAUGAUGGAGGCCCCUGUCCCUGUGUCAGCAACAUCCAU'.lower())
    # print(star_first_motifs)
    # mature_first_motifs = find_motifs('ffffffffffffffffffffffffffffffMMMMMMMMMMMMMMMMMMMMMMMMllllllllllllllllSSSSSSSSSSSSSSSSSSSSSSffffffffffffffffffffffffffffff', 'CCAUCCUAGAGAGCACUGAGCGACAGAUACUGUAAACAUCCUACACUCUCAGCUGUGGAAAGUAAGAAAGCUGGGAGAAGGCUGUUUACUCUUUCUGCCUUGGAAGUCAACUAAAGAGAAAU'.lower())
    # print(mature_first_motifs)
    df = read_dataframes(list_of_pickle_files_in("resources/dataset/split/train"))
    fp = df.loc[(df['false_positive'] == True)].copy()
    tp = df.loc[(df['false_positive'] == False)].copy()
    # fp = pd.read_pickle("resources/dataset/false_positives_SRR2496781-84_bigger.pkl")
    # tp = pd.read_pickle("resources/dataset/true_positives_TCGA_LUSC_only_in_mirgene_db.pkl")

    fp[['has_ug_motif', 'has_ugu_motif', 'has_cnnc_motif']] = fp.apply(lambda x: pd.Series(find_motifs(x['exp'], x['pri_seq'])), axis=1)
    print("False positives with all motifs: " + str(len(fp[(fp['has_ug_motif'] == 1) & (fp['has_ugu_motif'] == 1) & (fp['has_cnnc_motif'] == 1)])))
    print("False positives with no motifs: " + str(len(fp[(fp['has_ug_motif'] == 0) & (fp['has_ugu_motif'] == 0) & (fp['has_cnnc_motif'] == 0)])))
    print(fp.describe())

    tp[['has_ug_motif', 'has_ugu_motif', 'has_cnnc_motif']] = tp.apply(lambda x: pd.Series(find_motifs(x['exp'], x['pri_seq'])), axis=1)
    print("True positives with all motifs: " + str(len(tp[(tp['has_ug_motif'] == 1) & (tp['has_ugu_motif'] == 1) & (tp['has_cnnc_motif'] == 1)])))
    print("True positives with no motifs: " + str(len(tp[(tp['has_ug_motif'] == 0) & (tp['has_ugu_motif'] == 0) & (tp['has_cnnc_motif'] == 0)])))
    print(tp.describe())
