from mirdeepsquared.common import prepare_data, read_dataframes, Y_values
from mirdeepsquared.predict import predict
from sklearn.metrics import f1_score

import random
from statistics import mean

# Idea adapted from https://builtin.com/data-science/feature-importance
if __name__ == '__main__':
    used_features = ['combined_numerics', 'consensus_sequence_as_sentence', 'structure_as_1D_array', 'pri_seq_encoded', 'read_density_map_percentage_change', 'read_density_map_moving_average', 'location_of_mature_star_and_hairpin', 'motifs']
    model_path = "models/"
    holdout = ["resources/dataset/split/holdout/holdout.pkl"]
    df = read_dataframes(holdout)
    df = prepare_data(df)
    pred = predict(model_path, df)
    pred = (pred >= 0.50)
    original_F1 = f1_score(Y_values(df), pred)

    print("Original F1-score: " + str(original_F1))

    shuffled_feature_f1 = {}

    for feature in used_features:
        F1_with_feature_shuffled = []
        for i in range(0, 3):
            df = read_dataframes(holdout)
            df = prepare_data(df)
            random.shuffle(df[feature].values)
            pred = predict(model_path, df)
            pred = (pred >= 0.50)
            F1_with_feature_shuffled.append(f1_score(Y_values(df), pred))

        print(f'Average F1-score with {feature} shuffled: ' + str(mean(F1_with_feature_shuffled)))
        shuffled_feature_f1[feature] = mean(F1_with_feature_shuffled)
    sorted_by_importance = sorted(shuffled_feature_f1.items(), key=lambda x: x[1])
    print(sorted_by_importance)
