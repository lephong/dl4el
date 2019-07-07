dl4el: A distant learning approach to entity linking
========

A Python implementation of ACL2019 paper 

[1] Phong Le and Ivan Titov. [Distant Learning for Entity Linking with Automatic Noise Detection](https://arxiv.org/abs/1905.07189). ACL 2019.

Written and maintained by Phong Le (lephong.xyz [at] gmail.com)

### Installation

Requirements: Python 3.7, Pytorch 0.4, CUDA 8

### Usage

The following instruction is for replicating the experiments reported in [1]. 
Note that training and testing need lots of RAM (about 30GB) because 
some files related to Freebase have to be loaded. 


#### Data

Download data from [here](https://drive.google.com/open?id=1un-UQGPFVpDVxeXtijz6eA5xYkBCe_eV) 
and unzip to the main folder (i.e. your-path/dl4el).

The new training set (170k sentences, based on the New York Time corpus) is at

    data/freebase/el_annotation/el_annotated_170k.json

The dev and test sets (based on AIDA-CoNLL corpus) are at

    data/EL/AIDA/test[a-b].json


#### Train

To train, from the main folder run

    python3 -m jrk.el_main

**IMPORTANT: if you want to train the model from scratch, you have to remove the current saved model (if exists, by `rm model.*`).**

Using a GTX 1080 Ti GPU it will take about 3 hours for 20 epochs. The output is a model saved in two files: 
`model.config` and `model.state_dict` . 

For more options, please have a look at `jrk/el_hyperparams.py` 

#### Evaluation

Execute

    python3 -m jrk.el_main --mode eval 


