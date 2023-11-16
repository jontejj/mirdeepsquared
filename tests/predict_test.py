from .context import mirdeepsquared
#Make training reproducable
from tensorflow import keras
keras.utils.set_random_seed(42)
import tensorflow as tf
tf.config.experimental.enable_op_determinism()
from mirdeepsquared.train import list_of_pickle_files_in, read_dataframes, train_main
#from mirdeepsquared.train_simple_density_map import train_density_map
import shutil, tempfile
from os import path
from mirdeepsquared.predict import predict_main
from mirdeepsquared.predict_cmd import parse_args

class TestPredict:

    def train(self, model_path, train_results_file):
        #df = read_dataframes(list_of_pickle_files_in("resources/dataset"))
        model, history = train_main("resources/dataset", model_path, "tests/two-hyperparameters.yaml", train_results_file)
        model.save(model_path)

    def test_false_positives(self, tmp_path):
        model_path = str(tmp_path / "keras.model")
        train_results_file = str(tmp_path / "train_results_tmp.csv")
        self.train(model_path, train_results_file)
        args = parse_args(["resources/false_positives/result_08_11_2023_t_19_35_00.csv", "resources/false_positives/08_11_2023_t_19_35_00_output.mrd", "-m",  model_path])
        false_positives = predict_main(args)
        #TODO: improve!
        assert len(false_positives) < 50
