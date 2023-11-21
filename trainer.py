from tensorflow import keras
# Make training reproducable
keras.utils.set_random_seed(42)
import tensorflow as tf
tf.config.experimental.enable_op_determinism()
from mirdeepsquared.common import list_of_pickle_files_in, read_dataframes
# from mirdeepsquared.train_simple_density_map import train_density_map
from mirdeepsquared.train import train_main
from tensorflow import keras

if __name__ == '__main__':
    train_files = list_of_pickle_files_in("resources/dataset/split/train")  # with dataset-backup (.pkl files from commit 4b9cf56) the accuracy is better because the entries were not filtered with the mirgene db file for some reason (Issue #1)
    df = read_dataframes(train_files)
    train_main("resources/dataset/split/train/", "mirdeepsquared/train-simple-model.keras", "mirdeepsquared/default-hyperparameters.yaml", "trainer-train-results.csv", parallelism=2)
    # model = train_density_map(df)
    # model = train_simple_numerical_features(df)
    # model = train_simple_structure(df)
    # model.save("mirdeepsquared/train-simple-model-density-map.keras")
