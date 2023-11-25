# Make training reproducable, and test results stable
from tensorflow import keras
keras.utils.set_random_seed(42)
import tensorflow as tf
tf.config.experimental.enable_op_determinism()

from mirdeepsquared.common import list_of_pickle_files_in, prepare_data, read_dataframes, split_into_different_files
from mirdeepsquared.train import train_main
from mirdeepsquared.predict import true_positives, predict_main
from mirdeepsquared.predict_cmd import parse_args
import multiprocessing


class TestPredict:

    def train(self, tmp_path):
        model_path = str(tmp_path / "keras.model")
        train_results_file = str(tmp_path / "train_results_tmp.csv")
        split_main_path = str(tmp_path / "split-data")
        split_into_different_files("resources/dataset", split_main_path, 0.8)
        train_main(split_main_path + "/train", model_path, "tests/two-hyperparameters.yaml", train_results_file, parallelism=min([multiprocessing.cpu_count() - 1, 1]))
        return model_path

    def test_false_positives(self, tmp_path):
        model_path = self.train(tmp_path)

        holdout_df = prepare_data(read_dataframes(list_of_pickle_files_in(str(tmp_path / "split-data/holdout"))))
        train_df = prepare_data(read_dataframes(list_of_pickle_files_in(str(tmp_path / "split-data/train"))))
        assert len(holdout_df) == 0.25 * len(train_df)

        args = parse_args(["resources/false_positives/result_08_11_2023_t_19_35_00.csv", "resources/false_positives/08_11_2023_t_19_35_00_output.mrd", "-m", model_path])
        true_positives_in_seen_data = predict_main(args)
        # TODO: improve!
        assert len(true_positives_in_seen_data) < 35

        expected_true_positives = set(holdout_df[(holdout_df['false_positive'] == False)]['location'])

        predicted_true_positives_holdout = set(true_positives(model_path, holdout_df))
        difference = set(expected_true_positives) ^ set(predicted_true_positives_holdout)
        assert (1 - (len(difference) / len(holdout_df.values))) * 100 > 97
