import os
#import tensorflow
import screed # a library for reading in FASTA/FASTQ
#from tensorflow import keras
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

#import keras
#from keras.layers import Conv2D
#from keras.layers import Conv1D

def build_kmers(sequence, ksize):
    kmers = []
    n_kmers = len(sequence) - ksize + 1

    for i in range(n_kmers):
        kmer = sequence[i:i + ksize]
        kmers.append(kmer)

    return kmers

def read_kmers_from_file(filename, ksize):
    all_kmers = []
    for record in screed.open(filename):
        sequence = record.sequence
        kmers = build_kmers(sequence, ksize)
        all_kmers += kmers

    return all_kmers

def kmers_from_list(list, ksize):
    all_kmers = []
    for sequence in list:
        kmers = build_kmers(sequence, ksize)
        all_kmers += kmers

    return all_kmers

#Convert A->1 C->2 T->3 G->4 U->5

#def convert_kmer_to_numpy(kmer)
    
if __name__ == '__main__':

    #x = np.arange(15, dtype=np.int8).reshape(3, 5)
    #x[1:, ::2] = -99
    #print(x)

    df1 = pd.read_pickle("not_false_positives_small.pkl")
    df2 = pd.read_pickle("false_positives_small.pkl")

    df = pd.concat([df1, df2], axis=0)
    #print(df['mm_struct'])

    #1. Convert consensus_sequence into a list of numbers
    #2. Build kmers and use nlp?
    df['consensus_sequence_kmers'] = df.apply(lambda x: build_kmers(x['consensus_sequence'], 6), axis=1)
    print(df.head())

    consensus_texts = list(df['consensus_sequence_kmers'])
    for item in range(len(consensus_texts)):
        consensus_texts[item] = ' '.join(consensus_texts[item])

    print(df.head())

    cv = CountVectorizer(ngram_range=(4,4))
    X_miRNA = cv.fit_transform(consensus_texts)
    y_data = df['false_positive'].values

    X_train, X_test, y_train, y_test = train_test_split(X_miRNA, 
                                                    y_data, 
                                                    test_size = 0.20, 
                                                    random_state=42)

    classifier = MultinomialNB(alpha=0.1)
    classifier.fit(X_train, y_train)

    pred=classifier.predict(X_test)

    print(confusion_matrix(y_test,pred))

    #kmers = kmers_from_list(df['consensus_sequence'].tolist(), 6)
    #truth_labels = df['false_positive']
    #kmers = read_kmers_from_file("resources/known-mature-sequences-h_sapiens.fas", 20)
    #print(kmers[:5])

    #model = keras.models.Sequential()

    #model.add(Conv2D(1, kernel_size=(3,3), input_shape = (128, 128, 3)))

    #model.summary()
    pass
