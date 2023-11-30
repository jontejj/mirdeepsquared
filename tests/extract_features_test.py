from mirdeepsquared.extract_features import parse_args, extract_features_main
from mirdeepsquared.common import extract_precursor_from_exp_and_pri_seq


class TestExtractFeatures:

    def test_false_positives(self):
        args = parse_args(["resources/false_positives/result_08_11_2023_t_19_35_00.csv", "resources/false_positives/08_11_2023_t_19_35_00_output.mrd", "not_used.pkl", "-fp", "--section", "novel"])
        df = extract_features_main(args)
        assert len(df[(df['predicted_as_novel'] == True) & (df['false_positive'] == True)]) == 1199

    """
    Note: this does not test the real true positive file as it is too big to be included in the repository
    """
    def test_true_positives(self):
        args = parse_args(["resources/result_30_10_2023_t_15_05_15.csv", "resources/output.mrd", "not_used.pkl", "-tp", "--section", "known"])
        df = extract_features_main(args)
        assert len(df[(df['predicted_as_novel'] == False) & (df['false_positive'] == False)]) == 7

    def test_extracting_precursor_from_exp_and_pri_seq_star_first(self):
        exp = 'fffffffffffffffffffffffffffffffSSSSSSSSSSSSSSSSSSSSSSSllllllllllllllllMMMMMMMMMMMMMMMMMMMMMMffffffffffffffffff'
        pri_seq = 'caacuauuauucucggaucagaucgagccauugcugguuucuuccacagugguacuuuccauuagaacuaucaccggguggaaacuagcaguggcucgaucuuuuccacu'
        precursor = extract_precursor_from_exp_and_pri_seq(exp, pri_seq)
        assert precursor == 'ugcugguuucuuccacagugguacuuuccauuagaacuaucaccggguggaaacuagcagu'

    def test_extracting_precursor_from_exp_and_pri_seq_mature_first(self):
        exp = 'fffffffffffffffffffffffffffffffMMMMMMMMMMMMMMMMMMMMMMllllllllllllllllSSSSSSSSSSSSSSSSSSSSSSSffffffffffffffffff'
        pri_seq = 'caacuauuauucucggaucagaucgagccauugcugguuucuuccacagugguacuuuccauuagaacuaucaccggguggaaacuagcaguggcucgaucuuuuccacu'
        precursor = extract_precursor_from_exp_and_pri_seq(exp, pri_seq)
        assert precursor == 'ugcugguuucuuccacagugguacuuuccauuagaacuaucaccggguggaaacuagcagu'
