# CorefDiffs: Co-referential and Differential Knowledge Flow in Document Grounded Conversations

This project hosts the code and dataset for Coref-Diffs.


## Download data

Download the dataset and pre-trained weights for WoW and Holl-E.

The selector checkpoints for WoW is saved to 'models/wow'. The dataset for WoW is saved to 'data/wow'



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