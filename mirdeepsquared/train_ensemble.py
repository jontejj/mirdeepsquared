from pathlib import Path
from mirdeepsquared.motifs_bayes_model import MotifModel
from mirdeepsquared.train_simple_density_map import DensityMapModel
from mirdeepsquared.train_simple_numerical_feature import NumericalModel
from mirdeepsquared.train_simple_structure import StructureModel
from tensorflow import keras
# Make training reproducable
keras.utils.set_random_seed(42)
import tensorflow as tf
tf.config.experimental.enable_op_determinism()
from mirdeepsquared.common import list_of_pickle_files_in, prepare_data, read_dataframes, split_data_once
from tensorflow import keras


def train_ensemble(dataset_path, model_output_path):
    model_output_dir = Path(model_output_path)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    train_files = list_of_pickle_files_in(dataset_path)  # with .pkl files from commit 4b9cf56 the accuracy was better because the entries were not filtered with the mirgene db file for some reason (Issue #1)
    df = read_dataframes(train_files)
    print("False positives:" + str(len(df[(df['false_positive'] == True)])))
    print("True positives:" + str(len(df[(df['false_positive'] == False)])))

    # TODO: how to use the same split for the big model? Train again after figuring out the best parameters?
    # train_main("resources/dataset/split/train/", "models/Big_Model_best_hyperparameters.keras", "mirdeepsquared/best-hyperparameters.yaml", "train-best-results-for-ensemble.csv", parallelism=2)

    train, val = split_data_once(prepare_data(df))
    motifs = MotifModel()
    motifs.train(train, val)
    motifs.save(model_output_path + "/MotifModel_motifs.pkl")

    density_map_model = DensityMapModel()
    density_map_model.train(train, val)
    density_map_model.save(model_output_path + "/DensityMapModel_with_location_of_mature_star_and_hairpin.keras")

    structure_model = StructureModel()
    structure_model.train(train, val)
    structure_model.save(model_output_path + "/StructureModel_simple.keras")

    numerical_model = NumericalModel()
    numerical_model.train(train, val)
    numerical_model.save(model_output_path + "/NumericalModel_simple.keras")
