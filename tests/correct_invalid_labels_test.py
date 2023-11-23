from mirdeepsquared.extract_features import parse_args, extract_features_main
from mirdeepsquared.correct_invalid_labels import correct_label
import pandas as pd


class TestCorrectInvalidLabel:
    def test_correct_invalid_label(self, tmp_path):
        tmp_pickle_file = str(tmp_path / "tmp_pickle_file.pkl")
        args = parse_args(["resources/result_30_10_2023_t_15_05_15.csv", "resources/output.mrd", "not_used.pkl", "-tp", "--section", "known"])
        df = extract_features_main(args)
        df.to_pickle(tmp_pickle_file)
        correct_label(tmp_pickle_file, "chrII:11534525-11540624_11", True)
        df = pd.read_pickle(tmp_pickle_file)

        changed_sample = df.loc[(df['location'] == 'chrII:11534525-11540624_11')]
        assert changed_sample['false_positive'].values[0] == True

        unchanged_sample = df.loc[(df['location'] == 'chrII:11534525-11540624_17')]
        assert unchanged_sample['false_positive'].values[0] == False
