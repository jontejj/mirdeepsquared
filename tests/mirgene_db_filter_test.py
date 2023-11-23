from mirdeepsquared.extract_features import parse_args, extract_features_main
from mirdeepsquared.mirgene_db_filter import filter_out_sequences_not_in_mirgene_db


class TestMirgeneDbFilter:
    def test_mirgene_db_filter(self):
        args = parse_args(["resources/result_30_10_2023_t_15_05_15.csv", "resources/output.mrd", "not_used.pkl", "-fp"])
        df = extract_features_main(args)
        df = filter_out_sequences_not_in_mirgene_db(df, "tests/mirgene_db_example.txt")
        assert len(df.loc[(df['location'] == 'chrII:11534525-11540624_7')]) == 1
        assert len(df) == 1
