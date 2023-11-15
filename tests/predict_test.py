from mirdeepsquared.train import list_of_pickle_files_in, read_dataframes
from mirdeepsquared.train_simple_density_map import train_density_map
import unittest
import os
from context import mirdeepsquared
from mirdeepsquared.predict import parse_args, predict_main

TEST_MODEL_FILE = "test_false_positives.keras"
class TestExtractFeatures(unittest.TestCase):

    def setUp(self) -> None:
        df = read_dataframes(list_of_pickle_files_in("resources/dataset"))
        model, history = train_density_map(df)
        model.save(TEST_MODEL_FILE)
        return super().setUp()

    def test_false_positives(self):
        args = parse_args(["resources/false_positives/result_08_11_2023_t_19_35_00.csv", "resources/false_positives/08_11_2023_t_19_35_00_output.mrd", "-m", TEST_MODEL_FILE])
        false_positives = predict_main(args)
        self.assertEqual(len(false_positives), 0)

    def tearDown(self) -> None:
        os.remove(TEST_MODEL_FILE)
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()