# Few-Shot Scene Adaptive Crowd Counting Using Meta-Learning

### Introduction
This repository is contains the PyTorch implementation for "Few-Shot Scene Adaptive Crowd Counting Using Meta-Learning" by Mahesh Kumar Krishna Reddy, Mohammad Hossain, Mrigank Rochan, and Yang Wang. If you make use of this code in your work, please cite the paper.

[[Code](github.com/maheshkkumar/crowd_meta)] [[Paper](https://arxiv.org/abs/2002.00264)]

```
@inproceedings{reddy2020few,
    author      =  {Reddy, Mahesh Kumar Krishna and Hossain, Mohammad and Rochan, Mrigank and Wang, Yang},
    title       =  {Few-Shot Scene Adaptive Crowd Counting Using Meta-Learning},
    booktitle   =  {The IEEE Winter Conference on Applications of Computer Vision (WACV)},
    month       =  {March},
    year        =  {2020}
}
```

### Abstract

![Problem Setup](image/introduction.png)

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
You can train the network using `run.sh` and by customising the command line arguments in it.

### Acknowledgements
We have borrowed code from the following repositories:
1. https://github.com/leeyeehoo/CSRNet-pytorch
2. https://github.com/katerakelly/pytorch-maml