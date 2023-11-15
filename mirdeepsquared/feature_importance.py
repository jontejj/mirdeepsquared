from .train import prepare_data, split_data, read_dataframes
from keras.saving import load_model
from sklearn.metrics import f1_score

import random
from statistics import mean

#Idea adapted from https://builtin.com/data-science/feature-importance

if __name__ == '__main__':

    #, 'estimated_probability', 'estimated_probability_uncertainty',
    used_features= ['mature_read_count', 'star_read_count', 'significant_randfold', 'consensus_sequence_as_sentence',
       'mature_vs_star_read_ratio', 'structure_as_1D_array', 'read_density_map_percentage_change']

    model = load_model("best-model-not-seen-test.keras")

    df = read_dataframes(["resources/dataset/true_positives_TCGA_LUSC.pkl", "resources/dataset/false_positives_SRR2496781-84_bigger.pkl"])
    X_train, Y_train, X_val, Y_val, X_test, Y_test, locations_train, locations_val, locations_test = split_data(prepare_data(df))
    pred = model.predict(X_val)
    pred = (pred>=0.50)
    original_F1 = f1_score(Y_val,pred)

    print("Original F1-score: " + str(original_F1))

    shuffled_feature_f1 = {}

    for feature in used_features:
        F1_with_feature_shuffled = []
        for i in range(0,3):
            df = read_dataframes(["resources/dataset/true_positives_TCGA_LUSC.pkl", "resources/dataset/false_positives_SRR2496781-84_bigger.pkl"])
            df = prepare_data(df)
            random.shuffle(df[feature].values)
            X_train, Y_train, X_val, Y_val, X_test, Y_test, locations_train, locations_val, locations_test = split_data(df)

            pred = model.predict(X_val)
            pred = (pred>=0.50)
            F1_with_feature_shuffled.append(f1_score(Y_val,pred))

        print(f'Average F1-score with {feature} shuffled: ' + str(mean(F1_with_feature_shuffled)))
        shuffled_feature_f1[feature] = mean(F1_with_feature_shuffled)
    sorted_by_importance = sorted(shuffled_feature_f1.items(), key=lambda x:x[1])
    print(sorted_by_importance)
