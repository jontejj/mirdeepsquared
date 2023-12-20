#!/usr/bin/env bash
set -euxo pipefail

python mirdeepsquared/mirgene_db_filter.py resources/dataset/true_positives/true_positives_TCGA_LUSC_all.pkl resources/ALL-precursors_in_mirgene_db.fas resources/dataset/true_positives_TCGA_LUSC_only_precursors_in_mirgene_db.pkl --stringent
python mirdeepsquared/mirgene_db_filter.py resources/dataset/true_positives/true_positives_TCGA_BRCA.pkl resources/ALL-precursors_in_mirgene_db.fas resources/dataset/true_positives_TCGA_BRCA_only_precursors_in_mirgene_db.pkl --stringent
python mirdeepsquared/correct_invalid_labels.py
python generate_data.py
cp resources/dataset/generated/false_positives_with_empty_read_density_maps.pkl resources/dataset/
python mirdeepsquared/mirgene_db_filter.py resources/dataset/other_species/true_positives/mouse/mouse.mature.pkl resources/ALL-precursors_in_mirgene_db.fas resources/dataset/mouse.mature_only_mirgene_db.pkl -s
python split_dataset.py resources/dataset/ resources/dataset/split -f 0.9 -r 42
python split_dataset.py resources/dataset/split/train resources/dataset/split/train-val -f 0.8 -r 42
python mirdeepsquared/mirgene_db_filter.py resources/dataset/other_species/true_positives/zebrafish/zebrafish.mature.2nd.run.pkl resources/ALL-precursors_in_mirgene_db.fas resources/dataset/split/holdout/zebrafish.mature.2nd.run_only_in_mirgene_db.pkl -s