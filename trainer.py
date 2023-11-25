from tensorflow import keras
# Make training reproducable
keras.utils.set_random_seed(42)
import tensorflow as tf
tf.config.experimental.enable_op_determinism()
from mirdeepsquared.common import list_of_pickle_files_in, read_dataframes
# from mirdeepsquared.train_simple_density_map import train_density_map
from mirdeepsquared.train import train_main
# from mirdeepsquared.train_simple_structure import train_simple_structure
# from mirdeepsquared.train_simple_numerical_feature import train_simple_numerical_features
# from mirdeepsquared.train_simple_motifs import train_simple_motifs
from tensorflow import keras

if __name__ == '__main__':
    train_files = list_of_pickle_files_in("resources/dataset/split/train")  # with .pkl files from commit 4b9cf56 the accuracy was better because the entries were not filtered with the mirgene db file for some reason (Issue #1)
    df = read_dataframes(train_files)
    print("False positives:" + str(len(df[(df['false_positive'] == True)])))
    print("True positives:" + str(len(df[(df['false_positive'] == False)])))
    train_main("resources/dataset/split/train/", "mirdeepsquared/train-simple-model.keras", "mirdeepsquared/best-hyperparameters.yaml", "train-best-results-fixed-holdout-bug.csv", parallelism=2)

    # model = train_simple_motifs(df)
    # model.save("mirdeepsquared/train-simple-model-motifs.keras")
    # model = train_density_map(df)
    # model.save("mirdeepsquared/train-simple-model-density-map.keras")
    # model = train_simple_numerical_features(df)
    # model.save("mirdeepsquared/train-simple-model-numerical-features.keras")
    # model = train_simple_structure(df)
    # model.save("mirdeepsquared/train-simple-model-precursors-max.keras")
