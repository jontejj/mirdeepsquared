from tensorflow import keras
# Make training reproducable
keras.utils.set_random_seed(42)
import tensorflow as tf
tf.config.experimental.enable_op_determinism()
from pathlib import Path
from mirdeepsquared.motifs_bayes_model import MotifModel
from mirdeepsquared.density_map_model import DensityMapModel
from mirdeepsquared.numerical_model import NumericalModel
from mirdeepsquared.structure_model import StructureModel
from mirdeepsquared.estimation_model import EstimationModel
from mirdeepsquared.common import list_of_pickle_files_in, prepare_data, read_dataframes, split_data_once


def train_ensemble(dataset_path, model_output_path):
    model_output_dir = Path(model_output_path)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    train_files = list_of_pickle_files_in(dataset_path)  # with .pkl files from commit 4b9cf56 the accuracy was better because the entries were not filtered with the mirgene db file for some reason (Issue #1)
    df = read_dataframes(train_files)
    print("False positives:" + str(len(df[(df['false_positive'] == True)])))
    print("True positives:" + str(len(df[(df['false_positive'] == False)])))

    train, val = split_data_once(prepare_data(df))

    train_no_generated = train[~train['location'].str.endswith('_generated')]
    motifs = MotifModel()
    motifs.train(train_no_generated, val)
    motifs.save(model_output_dir / "MotifModel_motifs.pkl")

    estimation_model = EstimationModel()
    estimation_model.train(train_no_generated, val)
    estimation_model.save(model_output_dir / "EstimationModel_model.pkl")

    density_map_model = DensityMapModel()
    density_map_model.train(train, val)
    density_map_model.save(model_output_dir / "DensityMapModel_with_location_of_mature_star_and_hairpin.keras")

    structure_model = StructureModel()
    structure_model.train(train, val)
    structure_model.save(model_output_dir / "StructureModel_simple.keras")

    numerical_model = NumericalModel()
    numerical_model.train(train, val)
    numerical_model.save(model_output_dir / "NumericalModel_simple.keras")
