dl4el: A distant learning approach for entity linking
========

A Python implementation of the approach proposed in

[1] Phong Le and Ivan Titov (2019). [Distant learning for Entity linking with Noise prediction](https://arxiv.org/pdf/anonymous.pdf).

Written and maintained by Phong Le (lephong.xyz [at] gmail.com)

### Installation

Requirements: Python 3.7, Pytorch 0.4, CUDA 8

### Usage

The following instruction is for replicating the experiments reported in [1]. 
Note that training and testing take lots of RAM (about 30GB) because 
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

Using a GTX 1080 Ti GPU it will take about 3 hours for 20 epochs. The output is a model saved in two files: 
`model.config` and `model.state_dict` . 

For more options, please have a look at `jrk/el_hyperparams.py` 

#### Evaluation

Execute

    python3 -m jrk.el_main --model eval 


