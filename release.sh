#!/usr/bin/env bash

# Procedure for releasing a new version of mirdeepsquared.
# Note: remember to first bump the version in setup.py!

set -euxo pipefail

rm -rf ~/.virtualenvs/mirdeepsquared-rel
virtualenv ~/.virtualenvs/mirdeepsquared-rel -p python3.9
source ~/.virtualenvs/mirdeepsquared-rel/bin/activate
rm -rf build/ dist/
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build
python3 -m pip install twine
pip install -r requirements.txt
python3 -m pip install -e .
./prepare_default_dataset.sh
python3 mirdeepsquared/train.py resources/dataset/split/train -o mirdeepsquared/models/ -hp mirdeepsquared/best-hyperparameters.yaml -tr trainer-results.csv
python3 -m build
twine upload dist/*
echo "Successfully released!"