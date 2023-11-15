from .context import mirdeepsquared
#Make training reproducable
from tensorflow import keras
keras.utils.set_random_seed(42)
import tensorflow as tf
tf.config.experimental.enable_op_determinism()
from mirdeepsquared.train import list_of_pickle_files_in, read_dataframes
from mirdeepsquared.train_simple_density_map import train_density_map
import shutil, tempfile
from os import path
from mirdeepsquared.predict import predict_main
from mirdeepsquared.predict_cmd import parse_args

class TestPredict:

    def train(self, model_path):
        df = read_dataframes(list_of_pickle_files_in("resources/dataset"))
        model, history = train_density_map(df)
        model.save(model_path)

    def test_false_positives(self, tmp_path):
        model_path = str(tmp_path / "keras.model")
        self.train(model_path)
        args = parse_args(["resources/false_positives/result_08_11_2023_t_19_35_00.csv", "resources/false_positives/08_11_2023_t_19_35_00_output.mrd", "-m",  model_path])
        false_positives = predict_main(args)
        #TODO: improve!
        assert len(false_positives) < 30
