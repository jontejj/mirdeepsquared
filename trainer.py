from tensorflow import keras
#Make training reproducable
keras.utils.set_random_seed(42)
import tensorflow as tf
tf.config.experimental.enable_op_determinism()
from mirdeepsquared.common import list_of_pickle_files_in, read_dataframes, train_main
from tensorflow import keras

if __name__ == '__main__':
    human = "resources/dataset/split/train" #list_of_pickle_files_in("resources/dataset") #with dataset-backup (.pkl files from commit 4b9cf56) the accuracy is better because the entries were not filtered with the mirgene db file for some reason (Issue #1)
    other_species = list_of_pickle_files_in("resources/dataset/other_species")
    df = read_dataframes(human) # human + other_species) #

    train_main("resources/dataset/split/train/", "mirdeepsquared/train-simple-model.keras", "mirdeepsquared/default-hyperparameters.yaml", "trainer-train-results.csv")
    #model = train_density_map(df)
    #model = train_simple_numerical_features(df)
    #model = train_simple_structure(df)
    #model.save("mirdeepsquared/train-simple-model.keras")
