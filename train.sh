#!/bin/bash
python link_prediction.py
cp models/link_predictor-9.model models/link_predictor.model

python context_selection.py 
cp models/context_selector-9.model models/context_selector.model

python make_drop_style.py

cp iirc/drop_train.json numnet_plus/drop_dataset/drop_dataset_train.json
cp iirc/drop_dev.json numnet_plus/drop_dataset/drop_dataset_dev.json

cd numnet_plus
./run_train.sh
