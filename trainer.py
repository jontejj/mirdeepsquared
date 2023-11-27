from mirdeepsquared.train_simple_density_map import DensityMapModel
from mirdeepsquared.train_simple_numerical_feature import NumericalModel
from mirdeepsquared.train_simple_structure import StructureModel
from tensorflow import keras
# Make training reproducable
keras.utils.set_random_seed(42)
import tensorflow as tf
tf.config.experimental.enable_op_determinism()
from mirdeepsquared.common import list_of_pickle_files_in, prepare_data, read_dataframes, split_data_once
# from mirdeepsquared.train import train_main
from mirdeepsquared.motifs_bayes_model import MotifModel
from tensorflow import keras

if __name__ == '__main__':
    train_files = list_of_pickle_files_in("resources/dataset/split/train")  # with .pkl files from commit 4b9cf56 the accuracy was better because the entries were not filtered with the mirgene db file for some reason (Issue #1)
    df = read_dataframes(train_files)
    print("False positives:" + str(len(df[(df['false_positive'] == True)])))
    print("True positives:" + str(len(df[(df['false_positive'] == False)])))

    # train_main("resources/dataset/split/train/", "models/Big_Model_best_hyperparameters.keras", "mirdeepsquared/best-hyperparameters.yaml", "train-best-results-for-ensemble.csv", parallelism=2)

    train, val = split_data_once(prepare_data(df))

    density_map_model = DensityMapModel()
    density_map_model.train(train, val)
    density_map_model.save("models/DensityMapModel_bigger.keras")

    motifs = MotifModel()
    motifs.train(train, val)
    motifs.save("models/MotifModel_bayes.pkl")

    structure_model = StructureModel()
    structure_model.train(train, val)
    structure_model.save("models2/StructureModel_simple.keras")

    numerical_model = NumericalModel()
    numerical_model.train(train, val)
    numerical_model.save("models/NumericalModel_simple.keras")

