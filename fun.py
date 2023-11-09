import os
#import tensorflow
#import screed # a library for reading in FASTA/FASTQ
#from tensorflow import keras
#import numpy as np
import pandas as pd
import re
import numpy as np


    
if __name__ == '__main__':
    #df1 = pd.read_pickle("not_false_positives_small.pkl")
    #df2 = pd.read_pickle("false_positives_small.pkl")
    df3 = pd.read_pickle("resources/dataset/not_false_positives_TCGA_LUSC.pkl")
    print(df3['location'].values)
    #df = pd.concat([df1, df2], axis=0)

    #print(df['mm_struct'])
    #print(df['mm_offset'])
    #pri_struct = "..............(((..(((((((((((((((((((((((.((.(.((((((.(((......))).)))))).))).)))))))))))))))))))))))..)))..."
    #mm_struct = "(((((((((((.((.(.(((((("
    #mm_offset = 60
    #build_matrix(pri_struct, mm_struct, mm_offset)
    

    #input = "'seq_378026_x1                  ...............................ugcugguuucuuccacagugg..........................................................\t0\n'"
    #m = re.search(r"[A-Za-z]{3}_(\d*)_x(\d*)\s+([\.ucagUCAGN]*)\t\d*\n", input)
    #if m is not None:
    #    print(m)
    #    print(int(m.group(2)))
    #print(int("xxx"))
    pass
