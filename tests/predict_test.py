# Make training reproducable, and test results stable
from mirdeepsquared.train_ensemble import train_ensemble
from tensorflow import keras
keras.utils.set_random_seed(42)
import tensorflow as tf
tf.config.experimental.enable_op_determinism()

from mirdeepsquared.common import list_of_pickle_files_in, prepare_data, read_dataframes, split_into_different_files
from mirdeepsquared.mirgene_db_filter import main_mirgene_filter
from mirdeepsquared.train import train_main
from mirdeepsquared.predict import model_weights_from_file, true_positives, predict_main
from mirdeepsquared.predict_cmd import parse_args
import multiprocessing
import pytest


class TestPredict:

    @pytest.fixture(autouse=True, scope="class")
    def prepare_dataset(cls):
        main_mirgene_filter("resources/dataset/true_positives/true_positives_TCGA_LUSC_all.pkl", "resources/ALL-precursors_in_mirgene_db.fas", "resources/dataset/true_positives_TCGA_LUSC_only_precursors_in_mirgene_db.pkl", stringent=True)
        main_mirgene_filter("resources/dataset/true_positives/true_positives_TCGA_BRCA.pkl", "resources/ALL-precursors_in_mirgene_db.fas", "resources/dataset/true_positives_TCGA_BRCA_only_precursors_in_mirgene_db.pkl", stringent=True)

    @pytest.fixture(scope="class")
    def model_weights(cls):
        return model_weights_from_file("mirdeepsquared/model_weights.yaml")

    def train(self, tmp_path):
        models_path = tmp_path / "models"
        models_path.mkdir()
        model_path = str(models_path / "BigModel_testpredict.keras")
        train_results_file = str(tmp_path / "train_results_tmp.csv")
        split_main_path = str(tmp_path / "split-data")
        split_into_different_files("resources/dataset", split_main_path, 0.8, 42)
        train_main(split_main_path + "/train", model_path, "tests/two-hyperparameters.yaml", train_results_file, parallelism=min([multiprocessing.cpu_count() - 1, 1]))
        return model_path

    def test_false_positives(self, tmp_path, model_weights):
        model_path = self.train(tmp_path)

        holdout_df = prepare_data(read_dataframes(list_of_pickle_files_in(str(tmp_path / "split-data/holdout"))))
        train_df = prepare_data(read_dataframes(list_of_pickle_files_in(str(tmp_path / "split-data/train"))))
        expected_holdout_len = 0.25 * len(train_df)
        assert len(holdout_df) > expected_holdout_len - 1 and len(holdout_df) < expected_holdout_len + 1

        args = parse_args(["resources/false_positives/result_08_11_2023_t_19_35_00.csv", "resources/false_positives/08_11_2023_t_19_35_00_output.mrd", "-m", model_path])
        true_positives_in_seen_data = predict_main(args)
        assert len(true_positives_in_seen_data) < 35

        expected_true_positives = set(holdout_df[(holdout_df['false_positive'] == False)]['location'])

        predicted_true_positives_holdout = set(true_positives(model_path, holdout_df, model_weights, threshold=0.5))
        difference = set(expected_true_positives) ^ set(predicted_true_positives_holdout)
        assert (1 - (len(difference) / len(holdout_df.values))) * 100 > 92

    def test_train_ensemble(self, tmp_path, model_weights):
        model_path = str(tmp_path / "models")
        split_main_path = str(tmp_path / "split-data")
        split_into_different_files("resources/dataset", split_main_path, 0.8, 42)
        train_ensemble(split_main_path + "/train", model_path)

        holdout_df = prepare_data(read_dataframes(list_of_pickle_files_in(split_main_path + "/holdout")))
        expected_true_positives = set(holdout_df[(holdout_df['false_positive'] == False)]['location'])
        predicted_true_positives_holdout = set(true_positives(model_path, holdout_df, model_weights, threshold=0.5))
        difference = set(expected_true_positives) ^ set(predicted_true_positives_holdout)
        assert (1 - (len(difference) / len(holdout_df.values))) * 100 > 95

    # TODO: write a test for resuming training
