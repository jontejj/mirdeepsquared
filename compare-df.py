import pandas as pd
import numpy as np


def np_array_to_tuple(df):
    return df.applymap(lambda x: tuple(x) if isinstance(x, np.ndarray) else x)


if __name__ == '__main__':
    new_true = np_array_to_tuple(pd.read_pickle("resources/dataset/true_positives_TCGA_LUSC_only_in_mirgene_db.pkl"))
    new_false = np_array_to_tuple(pd.read_pickle("resources/dataset/generated/false_positives_with_empty_read_density_maps.pkl"))

    old_true = np_array_to_tuple(pd.read_pickle("/Users/jonatanjonsson/code/mirdeepsquared/resources/dataset/true_positives_TCGA_LUSC.pkl"))
    old_false = np_array_to_tuple(pd.read_pickle("/Users/jonatanjonsson/code/mirdeepsquared/resources/dataset/false_positives_SRR2496781-84_bigger.pkl"))
    print("Old true")
    print(old_true.describe())
    print("New true")
    print(new_true.describe())
    print("Old false")
    print(old_false.describe())
    print("New false")
    print(new_false.describe())

    merged_true = pd.merge(new_true, old_true, how='outer', indicator=True)
    # Select rows where the indicator column is not both
    differences = merged_true[merged_true['_merge'] != 'both']
    print("True differences")
    print(differences)

    print("Items in new false")
    print(len(new_false))
    print("Items in old false")
    print(len(old_false))

    merged_false = pd.merge(new_false, old_false, how='outer', indicator=True)
    # Select rows where the indicator column is not both
    differences = merged_false[merged_false['_merge'] != 'both']
    print("False differences")
    print(differences)
