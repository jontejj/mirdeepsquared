from mirdeepsquared.common import find_motifs


class TestFeatureExtraction:
    def test_motifs_mature_first(self):
        # From https://mirgenedb.org/show/hsa/Mir-30-P2b
        mature_first_motifs = find_motifs('ffffffffffffffffffffffffffffffMMMMMMMMMMMMMMMMMMMMMMMMllllllllllllllllSSSSSSSSSSSSSSSSSSSSSSffffffffffffffffffffffffffffff', 'CCAUCCUAGAGAGCACUGAGCGACAGAUACUGUAAACAUCCUACACUCUCAGCUGUGGAAAGUAAGAAAGCUGGGAGAAGGCUGUUUACUCUUUCUGCCUUGGAAGUCAACUAAAGAGAAAU'.lower())
        assert mature_first_motifs == [True, True, True]

    def test_motifs_star_first(self):
        # From https://mirgenedb.org/show/hsa/Mir-8-P1b
        star_first_motifs = find_motifs('ffffffffffffffffffffffffffffffSSSSSSSSSSSSSSSSSSSSSSlllllllllllllllllMMMMMMMMMMMMMMMMMMMMMMMffffffffffffffffffffffffffffff', 'CCCAAGGUGGGCGGGCUGGGCGGGGGCCCUCGUCUUACCCAGCAGUGUUUGGGUGCGGUUGGGAGUCUCUAAUACUGCCGGGUAAUGAUGGAGGCCCCUGUCCCUGUGUCAGCAACAUCCAU'.lower())
        assert star_first_motifs == [True, True, True]

    def test_motifs_no_cnnc(self):
        no_cnnc_motif = find_motifs('ffffffffffffffffffffffffffffffSSSSSSSSSSSSSSSSSSSSSSlllllllllllllllllMMMMMMMMMMMMMMMMMMMMMMMffffffffffffffffffffffffffffff', 'CCCAAGGUGGGCGGGCUGGGCGGGGGCCCUCGUCUUACCCAGCAGUGUUUGGGUGCGGUUGGGAGUCUCUAAUACUGCCGGGUAAUGAUGGAGGCCCCUGUCCCUGUGUGAGCAACAUCCAU'.lower())
        assert no_cnnc_motif == [True, True, False]

    def test_motifs_no_gug_no_cnnc(self):
        no_gug_no_cnnc_motif = find_motifs('ffffffffffffffffffffffffffffffSSSSSSSSSSSSSSSSSSSSSSlllllllllllllllllMMMMMMMMMMMMMMMMMMMMMMMffffffffffffffffffffffffffffff', 'CCCAAGGUGGGCGGGCUGGGCGGGGGCCCUCGUCUUACCCAGCAGUGUUUGGGAGCGGUUGGGAGUCUCUAAUACUGCCGGGUAAUGAUGGAGGCCCCUGUCCCUGUGUGAGCAACAUCCAU'.lower())
        assert no_gug_no_cnnc_motif == [True, False, False]

    def test_no_motifs(self):
        no_motifs = find_motifs('ffffffffffffffffffffffffffffffSSSSSSSSSSSSSSSSSSSSSSlllllllllllllllllMMMMMMMMMMMMMMMMMMMMMMMffffffffffffffffffffffffffffff', 'CCCAAGGUGGGCGGGCAGGGCGGGGGCCCUCGUCUUACCCAGCAGUGUUUGGGAGCGGUUGGGAGUCUCUAAUACUGCCGGGUAAUGAUGGAGGCCCCUGUCCCUGUGUGAGCAACAUCCAU'.lower())
        assert no_motifs == [False, False, False]
