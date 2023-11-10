from train import prepare_data, read_dataframes
from tensorflow import keras
from keras.saving import load_model

import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

if __name__ == '__main__':

    #df = read_dataframes(["not_false_positives_small.pkl", "false_positives_small.pkl"])
    #df = read_dataframes(["resources/dataset/not_false_positives_TCGA_LUSC.pkl", "resources/dataset/false_positives_SRR2496781-84.pkl"])
    df = read_dataframes(["resources/dataset/true_positives_TCGA_LUSC.pkl", "resources/dataset/false_positives_SRR2496781-84_bigger.pkl"])

    print("False positives:" + str(len(df[(df['false_positive']==True)])))
    print("True positives:" + str(len(df[(df['false_positive']==False)])))
    
    X_train, Y_train, X_val, Y_val, X_test, Y_test = prepare_data(df)

    model = load_model("best-model-not-seen-test.keras")
    model.summary()

    #TODO: check feature importance by shuffling values for each column in df and checking the impact on the F1-score

    pred = model.predict(X_test)
    pred = (pred>=0.50) #If probability is equal or higher than 0.50, It's most likely a false positive (True)
    print("Confusion matrix:")
    print(confusion_matrix(Y_test,pred))
    print("Accuracy: " + str(accuracy_score(Y_test,pred)))
    print("F1-score: " + str(f1_score(Y_test,pred)))
