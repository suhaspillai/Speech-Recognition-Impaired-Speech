Speech Recognition for speakers with speech impairment due to diseases like (Cerebral Palsy, Parkinson or Amyotrophic Lateral Sclerosis ALS). The base architecture is [DeepSpeech](http://arxiv.org/pdf/1512.02595v1.pdf) from Baidu. The initial pretrained model is from [SeanNaren](https://github.com/SeanNaren/deepspeech.torch.git) which was trained on small AN4 dataset. 

## Features
* The model is trained on 1000 hours of [Librispeech](http://www.danielpovey.com/files/2015_icassp_librispeech.pdf) data on normal speakers.
* Implemented Learning Hidden Unit Contribution Layer for speaker adaptation based on [Learning Hidden Unit Contributions For Unsupervised Speaker Adaptation Of Neural Network Acoustic Models](http://homepages.inf.ed.ac.uk/srenals/ps-slt14.pdf)
* The model is trained and tested on [TORGO](http://dl.acm.org/citation.cfm?id=2423820) speech database for dysarthric speakers (i.e speakers with speech disorders due to Cerebral Palsy) 
* Implemented Beam Search Decoding using Connectionist Temporal Classification (CTC) and Character Language Model based on Andrew Mass and Ziang Xie [Lexicon-Free Conversational Speech Recognition with Neural Networks](http://ai.stanford.edu/~amaas/papers/ctc_clm_naacl_2015.pdf) paper.

## Installation

For installation follow instruction guide [here](https://github.com/SeanNaren/deepspeech.torch.git)

## Data Preparation

Download dataset from [data](http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html).

There are total 8 speakers with speech impairment and 7 without speech impairment (normal speakers). Data preparation scripts are in python_scripts folder. Following command will create train and test files. This will also remove some unwanted files, which are not required.

```
python create_data.py path_to_speakers_folder path_to destination_folder test_speaker
```
For example,
```
python create_data.py /home/Torgo/data /home/Torgo/destination F01
```
path_to_speakers_folder: Here all the speaker folders are located
path_to_destination: Here test and train folders are located.
test_speaker: This speaker's data will be stored in test folder, while other speaker's data in train folder.

For Data Augmentation,
The following command will create augmentated data, this script uses [sox](http://sox.sourceforge.net/) 

```
python create_augmentated_data.py source_folder destination_folder 
```
This will create speech files with tempo and speed perturbation and amplified speech files.

The input to the model is in lmdb format, running the following command will store the data in lmdb format for fast data loading.

```
th MakeLMDB.lua -roottpath /home/torgo/data -lmdbPath /home/torgo/lmdb -windowSize 0.020 -stride 0.01 -sampleRate 16000 -audioExtension wav -processes 16 
```

Since, people with speech impairment generally have slow speaking rate, you can try changing the windowSize between 0.010 / 0.015 / 0.020 based on the speaker with speech impairment. 

## Training

First train on mixture of (impaired + normal) speakers data keeping test speaker aside, adding dropout only to conv layers.
```

th Train.lua

```
Use the trained model to adapt to the new speaker by using small subset of test speaker data.

```
th Train_SA.lua 

```
This will add LHU layer and train LHU layer to adapt to test speaker data.

## Testing

For testing, use the adapted model

```
th Test.lua

```



