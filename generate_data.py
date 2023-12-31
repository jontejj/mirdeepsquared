from pathlib import Path
import pandas as pd
import numpy as np


def generate_false_read_density_maps(file, output):
    data = pd.read_pickle(file)
    data['read_density_map'] = data.apply(lambda x: np.zeros(112, dtype=np.int32), axis=1)
    data = data.assign(false_positive=True)
    data['location'] = data['location'].astype(str) + "_generated"
    data.to_pickle(output)


if __name__ == '__main__':
    Path("resources/dataset/generated/").mkdir(parents=True, exist_ok=True)
    # if there is no read density information, it's a false positive
    generate_false_read_density_maps("resources/dataset/false_positives_SRR2496781-84_bigger.pkl", "resources/dataset/generated/false_positives_with_empty_read_density_maps.pkl")
    # TODO: random distribution of density is also a false positive
    # TODO: for true positives, add one more read on each of the MMMMM columns of read_density_map
