# IIRC-baseline
Code for the IIRC baseline used in [IIRC: A Dataset of Incomplete Information Reading Comprehension Questions](https://www.semanticscholar.org/paper/IIRC%3A-A-Dataset-of-Incomplete-Information-Reading-Ferguson-Gardner/01a1f2df34d947d7aa5698ca6fb31c03d15a5183).

(1/13) NOTE: I think I broke the link prediction model when cleaning up the code, I'm looking into this, and will (hopefully) push a fix shortly.

If you use the code, please cite the following paper:

```
@inproceedings{Ferguson2020IIRCAD,
  title={IIRC: A Dataset of Incomplete Information Reading Comprehension Questions},
  author={J. Ferguson and Matt Gardner and Hannaneh Hajishirzi and Tushar Khot and Pradeep Dasigi},
  booktitle={EMNLP},
  year={2020}
}
```

The final step in the pipeline uses a slightly modified version of [NumNet+](https://github.com/llamazing/numnet_plus).

## Usage

`./setup.sh`

This will download the IIRC data and the NumNet+ config.

`./train.sh`

This first trains the link predictor, then the context retrieval, and then finally NumNet+ on IIRC. These models can be trained independently, and the context retrieval training is very slow.

`numnet_plus/run_eval.sh`

This will evaluate a model saved in `numnet_plus/model.pt` on the data in `numnet_plus/drop_dataset/drop_dataset_dev.json`. This same evaluation is also run at the end of training.