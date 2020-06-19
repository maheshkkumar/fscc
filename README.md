# Few-Shot Scene Adaptive Crowd Counting Using Meta-Learning

### Introduction
This repository is contains the PyTorch implementation for "Few-Shot Scene Adaptive Crowd Counting Using Meta-Learning" by [Mahesh Kumar Krishna Reddy](http://cs.umanitoba.ca/~kumarkm/), Mohammad Hossain, [Mrigank Rochan](http://cs.umanitoba.ca/~mrochan/), and [Yang Wang](http://cs.umanitoba.ca/~ywang/). If you make use of this code in your work, please cite the paper.

[Code](https://github.com/maheshkkumar/crowd_meta) | [Paper](https://arxiv.org/abs/2002.00264)

### Abstract
<p align="center">
  <img src="./image/introduction.png" data-canonical-src="./image/introduction.png" width="300" height="200">
</p>

We consider the problem of few-shot scene adaptive crowd counting. Given a target camera scene, our goal is to adapt a model to this specific scene with only a few labeled images of that scene. The solution to this problem has potential applications in numerous real-world scenarios, where we ideally like to deploy a crowd counting model specially adapted to a target camera. We accomplish this challenge by taking inspiration from the recently introduced learning-to-learn paradigm in the context of few-shot regime. In training, our method learns the model parameters in a way that facilitates the fast adaptation to the target scene. At test time, given a target scene with a small number of labeled data, our method quickly adapts to that scene with a few gradient updates to the learned parameters. Our extensive experimental results show that the proposed approach outperforms other alternatives in few-shot scene adaptive crowd counting. 

### Setup
```python
pip install -r requirements.txt
```

### Datasets
The details related to all the crowd counting datasets can be found in the following links.
1. [WorldExpo'10](http://www.ee.cuhk.edu.hk/~xgwang/expo.html)
2. [Mall](https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html)
3. [UCSD](http://www.svcl.ucsd.edu/projects/peoplecnt/)

### Training

First, to generate a pre-trained backbone CSRNet network, please refer to [CSRNet documentation](https://github.com/leeyeehoo/CSRNet-pytorch). Then, the command line arguments for the meta-learning `train.py` is as follows:
```python
>> python train.py --help

usage: train.py [-h] [-d DATASET] -trp DATA_PATH [-nt NUM_TASKS] [-ni NUM_INSTANCES] [-mb META_BATCH] [-bb BASE_BATCH] [-mlr META_LR] [-blr BASE_LR] [-e EPOCHS] [-bu BASE_UPDATES] [-exp EXPERIMENT] -log LOG_NAME

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Name of the dataset
  -trp DATA_PATH, --data_path DATA_PATH
                        Path of the dataset
  -nt NUM_TASKS, --num_tasks NUM_TASKS
                        Number of tasks for training
  -ni NUM_INSTANCES, --num_instances NUM_INSTANCES
                        Number of instances per task for training
  -mb META_BATCH, --meta_batch META_BATCH
                        Batch size for meta network
  -bb BASE_BATCH, --base_batch BASE_BATCH
                        Batch size for base network
  -mlr META_LR, --meta_lr META_LR
                        Meta learning rate
  -blr BASE_LR, --base_lr BASE_LR
                        Base learning rate
  -e EPOCHS, --epochs EPOCHS
                        Number of training epochs
  -bu BASE_UPDATES, --base_updates BASE_UPDATES
                        Iterations for base network to train
  -exp EXPERIMENT, --experiment EXPERIMENT
                        Experiment number
  -log LOG_NAME, --log_name LOG_NAME
                        Name of logging file
```

You can either train the network using the `run.sh` or by using the following command:
```python
python train.py --dataset=<dataset_name> \
                --data_path=<data_path> \
                --num_tasks=<num_tasks> \
                --num_instances=<num_instances> \
                --meta_batch=<meta_batch> \
                --base_batch=<base_batch> \
                --meta_lr=0.001 \
                --base_lr=0.001 \
                --epochs=1000 \
                --base_updates=<base_updates> \
                --exp=<experiment_name> \
                --log=<log_path>
```

### Citation
```
@inproceedings{reddy2020few,
    author      =  {Reddy, Mahesh Kumar Krishna and Hossain, Mohammad and Rochan, Mrigank and Wang, Yang},
    title       =  {Few-Shot Scene Adaptive Crowd Counting Using Meta-Learning},
    booktitle   =  {The IEEE Winter Conference on Applications of Computer Vision (WACV)},
    month       =  {March},
    year        =  {2020}
}
```

### Acknowledgements
We have borrowed code from the following repositories:
1. https://github.com/leeyeehoo/CSRNet-pytorch
2. https://github.com/katerakelly/pytorch-maml

### License

The project is licensed under MIT license (please refer to LICENSE.txt for more details).