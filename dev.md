# Procedure for releasing a new version
Bump version in setup.py
```
virtualenv ~/.virtualenvs/mirdeepsquared-rel -p python3.9
source ~/.virtualenvs/mirdeepsquared-rel/bin/activate
rm -rf build/ dist/
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build
python3 -m pip install twine
pip install -r requirements.txt
python3 -m pip install -e .
python mirdeepsquared/correct_invalid_labels.py
python generate_data.py
cp resources/dataset/generated/false_positives_with_empty_read_density_maps.pkl resources/dataset/
cp resources/dataset/true_positives/true_positives_TCGA_BRCA.pkl resources/dataset/
python split_dataset.py resources/dataset/ resources/dataset/split -f 0.9
cp resources/dataset/other_species/true_positives/mouse/mouse.mature.pkl resources/dataset/split/train
python3 mirdeepsquared/train.py resources/dataset/split/train -o mirdeepsquared/models/ -hp mirdeepsquared/best-hyperparameters.yaml -tr trainer-results.csv
python3 -m build
twine upload dist/*
```