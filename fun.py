import os
#import tensorflow
#import screed # a library for reading in FASTA/FASTQ
#from tensorflow import keras
#import numpy as np
#import pandas as pd
import re
    
if __name__ == '__main__':

    input = "'seq_378026_x1                  ...............................ugcugguuucuuccacagugg..........................................................\t0\n'"
    m = re.search(r"[A-Za-z]{3}_(\d*)_x(\d*)\s+([\.ucagUCAGN]*)\t\d*\n", input)
    if m is not None:
        print(m)
        print(int(m.group(2)))
    print(int("xxx"))
    pass
