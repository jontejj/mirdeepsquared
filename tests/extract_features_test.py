import unittest

from context import mirdeepsquared
from mirdeepsquared.extract_features import parse_args, extract_features_main

class TestExtractFeatures(unittest.TestCase):

    def test_false_positives(self):
        args = parse_args(["resources/false_positives/result_08_11_2023_t_19_35_00.csv", "resources/false_positives/08_11_2023_t_19_35_00_output.mrd", "not_used.pkl", "-fp"])
        df = extract_features_main(args)
        self.assertEqual(len(df[(df['predicted_as_novel'] == True) & (df['false_positive'] == True)]), 1199)

    def test_filter_file(self):
        args = parse_args(["resources/false_positives/result_08_11_2023_t_19_35_00.csv", "resources/false_positives/08_11_2023_t_19_35_00_output.mrd", "not_used.pkl", "-fp", "-f", "tests/location_filter.txt"])
        df = extract_features_main(args)
        self.assertEqual(df['location'][0], "chr8_8_8338")
        self.assertEqual(len(df), 1)

    def test_mirgene_db_filter(self):
        args = parse_args(["resources/result_30_10_2023_t_15_05_15.csv", "resources/output.mrd", "not_used.pkl", "-fp", "-m", "tests/mirgene_db_example.txt"])
        df = extract_features_main(args)
        self.assertEqual(len(df.loc[(df['location'] == 'chrII:11534525-11540624_7')]), 1)
        self.assertEqual(len(df), 1)
    """
    Note: this does not test the real true positive file as it is too big to be included in the repository
    """
    def test_true_positives(self):
        args = parse_args(["resources/result_30_10_2023_t_15_05_15.csv", "resources/output.mrd", "not_used.pkl", "-tp"])
        df = extract_features_main(args)
        self.assertEqual(len(df[(df['predicted_as_novel'] == False) & (df['false_positive'] == False)]), 7)


if __name__ == '__main__':
    unittest.main()