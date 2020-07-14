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
## Results
* Rouge scores obtained by using best MLE trained model on test set:  
scores: {  
```'rouge-1':``` {'f': 0.3198728609234364, 'p': 0.40719348401816186, 'r': 0.2828704975863946},  
```'rouge-2':``` {'f': 0.18738412533800142, 'p': 0.24378047546954465, 'r': 0.16365593514992774},  
```'rouge-l':``` {'f': 0.3449876685273812, 'p': 0.5195584179761905, 'r': 0.27020822099624625}  
}
* Rouge scores obtained by using best MLE + RL trained model on test set:  
scores: {  
```'rouge-1':``` {'f': 0.30394816372035105, 'p': 0.38923155673042714, 'r': 0.2669144746884053},  
```'rouge-2':``` {'f': 0.17695009059376737, 'p': 0.23290476075407537, 'r': 0.15331403457211143},  
```'rouge-l':``` {'f': 0.31247062769929973, 'p': 0.45185872431258434, 'r': 0.2496851899023605}  
}
* Rouge scores obtained by using best RL trained model on test set:  
scores: {  
```'rouge-1':``` {'f': 0.3173991113843275, 'p': 0.41245745240064535, 'r': 0.27614900215772026},  
```'rouge-2':``` {'f': 0.18852843587777787, 'p': 0.25046859555012757, 'r': 0.1622990297381499},  
```'rouge-l':``` {'f': 0.34214496333785943, 'p': 0.5147891204773996, 'r': 0.2667430801426903}  
}
