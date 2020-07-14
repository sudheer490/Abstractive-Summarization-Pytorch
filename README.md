# Abstractive-Summarization-Pytorch
Pytorch implementation of "A Deep Reinforced Model for Abstractive Summarization" paper and pointer generator network 

## Model Description
* LSTM based Sequence-to-Sequence model for Abstractive Summarization
* Pointer mechanism for handling Out of Vocabulary (OOV) words [See et al. (2017)](https://arxiv.org/pdf/1704.04368.pdf)
* Intra-temporal and Intra-decoder attention for handling repeated words [Paulus et al. (2018)](https://arxiv.org/pdf/1705.04304.pdf)
* Self-critic policy gradient training along with MLE training [Paulus et al. (2018)](https://arxiv.org/pdf/1705.04304.pdf)
## Prerequisites
* Pytorch
* Tensorflow
* Python 2 & 3
* [rouge](!pip install rouge) 
## Data
We used telugu datasets of news articles from prajashakthi.For Every article we manually created summaries.We kept both the summary and article in a single file with a seprator(@highlight).The dataset contains 2104 training examples, 117 validation examples and 120 testing examples.
## Creating ```.bin``` files and vocab file
* The model accepts data in the form of ```.bin``` files.
* We convert ```.txt``` file into ```.bin``` files and chunked them further, run (requires Python 3 & Tensorflow):```python make_data_files.py```
* You will find the data in ```data/chunked``` folder and vocab file in ```data``` folder
## Training
* As suggested in [Paulus et al. (2018)](https://arxiv.org/pdf/1705.04304.pdf), first pretrain the seq-to-seq model using MLE (with Python 3):
```
python train.py --train_mle=yes --train_rl=no --mle_weight=1.0
```
* Next, find the best saved model on validation data by running (with Python 3):
```
python eval.py --task=validate --start_from=0000500.tar
```
* After finding the best model (lets say ```0001500.tar```) with high rouge-l f score, load it and run (with Python 3):
```
python train.py --train_mle=yes --train_rl=yes --mle_weight=0.25 --load_model=0001500.tar --new_lr=0.0001
```
for MLE + RL training (or)
```
python train.py --train_mle=no --train_rl=yes --mle_weight=0.0 --load_model=0001500.tar --new_lr=0.0001
```
for RL training

## Validation
* To perform validation of RL training, run (with Python 3):
```
python eval.py --task=validate --start_from=0001500.tar
```
## Testing
* After finding the best model of RL training (lets say ```00002500.tar```), evaluate it on test data & get all rouge metrics by running (with Python 3):
```
python eval.py --task=test --load_model=00002500.tar
```
