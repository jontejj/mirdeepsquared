from mirdeepsquared.train import parse_args, train_both_ensemble_and_big_model

if __name__ == '__main__':
    args = parse_args(["resources/dataset/split/train-val/train", "-o", "mirdeepsquared/models16", "-hp", "mirdeepsquared/best-hyperparameters.yaml", "-tr", "trainer-results.csv"])
    train_both_ensemble_and_big_model(args)
