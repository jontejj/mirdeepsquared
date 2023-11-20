from keras.saving import load_model

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
#TODO: put these methods in another file to improve boot time when predicting
from mirdeepsquared.train import read_dataframes, prepare_data, split_data, to_xy_with_location, list_of_pickle_files_in

if __name__ == '__main__':
    df = read_dataframes(list_of_pickle_files_in("resources/dataset"))
    #df = read_dataframes(["resources/dataset/true_positives_TCGA_LUSC.pkl", "resources/dataset/false_positives_SRR2496781-84_bigger.pkl"])

    print("False positives:" + str(len(df[(df['false_positive']==True)])))
    print("True positives:" + str(len(df[(df['false_positive']==False)])))
    
    train, val, test = split_data(prepare_data(df))
    X_test, Y_test, locations_test = to_xy_with_location(test)
    X_val, Y_val, locations_val = to_xy_with_location(val)
    #TODO: use estimated_probability_uncertainty to decide which model to use (ensemble)
    model = load_model("mirdeepsquared/train-simple-model.keras")

    pred = model.predict(X_test) #[X_test[1], X_test[2]]) #
    pred = (pred>=0.50) #If probability is equal or higher than 0.50, It's most likely a false positive (True)
    print("Test Confusion matrix:")
    print(confusion_matrix(Y_test,pred))
    print("Test Accuracy: " + str(accuracy_score(Y_test,pred)))
    print("Test F1-score: " + str(f1_score(Y_test,pred)))
    for i in range(0, len(pred)):
        if pred[i] != Y_test[i]:
            source_pickle = df[(df['location'] == locations_test[i])]['source_pickle'].array[0]
            print(f'Predicted: {not pred[i]} positive for {locations_test[i]}, real is: {not bool(Y_test[i])} positive. Source pickle: {source_pickle}')

    pred = model.predict(X_val)
    pred = (pred>=0.50) #If probability is equal or higher than 0.50, It's most likely a false positive (True)
    print("Val Confusion matrix:")
    print(confusion_matrix(Y_val,pred))
    print("Val Accuracy: " + str(accuracy_score(Y_val,pred)))
    print("Val F1-score: " + str(f1_score(Y_val,pred)))
    for i in range(0, len(pred)):
        if pred[i] != Y_val[i]:
            source_pickle = df[(df['location'] == locations_val[i])]['source_pickle'].array[0]
            print(f'Predicted: {not pred[i]} positive for {locations_val[i]}, real is: {not bool(Y_val[i])} positive. Source pickle: {source_pickle}')

