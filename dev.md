# Procedure for releasing a new version
Bump version in setup.py
```
rm -rf build/ dist/
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build
python3 -m pip install twine
python split_dataset.py resources/dataset/ resources/dataset/split
python3 trainer.py
python3 -m build
twine upload dist/*
```