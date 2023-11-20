from mirdeepsquared.common import prepare_data, read_dataframes, to_xy_with_location
from keras.saving import load_model
from sklearn.metrics import f1_score

import random
from statistics import mean

#Idea adapted from https://builtin.com/data-science/feature-importance

if __name__ == '__main__':

    #, 'estimated_probability', 'estimated_probability_uncertainty',
    used_features= ['mature_read_count', 'star_read_count', 'significant_randfold', 'consensus_sequence_as_sentence',
       'mature_vs_star_read_ratio', 'structure_as_1D_array', 'read_density_map_percentage_change', 'location_of_mature_star_and_hairpin']

    holdout = "resources/dataset/split/split.holdout.pkl"

    model = load_model("best-not-seen-test-model-6.keras")
    df = read_dataframes(holdout)
    X_test, Y_test, _ = to_xy_with_location(prepare_data(df))
    pred = model.predict(X_test)
    pred = (pred>=0.50)
    original_F1 = f1_score(X_test,pred)

    print("Original F1-score: " + str(original_F1))

    shuffled_feature_f1 = {}

    for feature in used_features:
        F1_with_feature_shuffled = []
        for i in range(0,3):
            df = read_dataframes(holdout)
            df = prepare_data(df)
            random.shuffle(df[feature].values)
            X, Y, _ = to_xy_with_location(df)
            pred = model.predict(X)
            pred = (pred>=0.50)
            F1_with_feature_shuffled.append(f1_score(Y,pred))

        print(f'Average F1-score with {feature} shuffled: ' + str(mean(F1_with_feature_shuffled)))
        shuffled_feature_f1[feature] = mean(F1_with_feature_shuffled)
    sorted_by_importance = sorted(shuffled_feature_f1.items(), key=lambda x:x[1])
    print(sorted_by_importance)
