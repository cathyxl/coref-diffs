# CorefDiffs: Co-referential and Differential Knowledge Flow in Document Grounded Conversations

This project hosts the code and dataset for CorefDiffs.


## Download data

Download the [dataset](https://drive.google.com/file/d/1AF7qM6lkhW3CkQD4ni1Jsr1I8y5ecGbP/view?usp=sharing) and [model weights](https://drive.google.com/file/d/1RkZFJ_jGWp8T7QiIpAuDOEaSFl_spt-k/view?usp=sharing) for WoW.

The selector checkpoints is saved to 'models/wow'. The dataset is saved to 'data/wow'.



## Inference
To test the selector
```bash
./scripts/test_selector.sh
```

## Training
To train the selector
```bash
./scripts/train_selector.sh
```
