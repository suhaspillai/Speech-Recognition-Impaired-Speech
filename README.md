Speech Recognition for speakers with speech impairment due to diseases like (Cerebral Palsy, Parkinson or Amyotrophic Lateral Sclerosis ALS). The base architecture is [DeepSpeech](http://arxiv.org/pdf/1512.02595v1.pdf) from Baidu. The intial pretrained model is from [SeanNaren](https://github.com/SeanNaren/deepspeech.torch.git) which was trained on small AN4 dataset. 

The model is trained on 1000 hours of [Librispeech] (http://www.danielpovey.com/files/2015_icassp_librispeech.pdf) data on normal speakers. 

## Features
* Trained on LibriSpeech 100hrs data.
* Modified for Phoneme Recognition.
* Model trained on torgo clean speech dataset gives PER of 8%

It is still a work in progress.
