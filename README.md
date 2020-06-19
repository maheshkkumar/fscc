# Few-shot scene adaptive crowd counting using meta-learning

This is the PyTorch implementation of the code for [Few-Shot Scene Adaptive Crowd Counting Using Meta-Learning](https://arxiv.org/abs/2002.00264) in WACV 2020.

### Setup
```python
pip install -r requirements.txt
```

### Dataset
1. [WorldExpo'10](http://www.ee.cuhk.edu.hk/~xgwang/expo.html)
2. [Mall](https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html)
3. [UCSD](http://www.svcl.ucsd.edu/projects/peoplecnt/)

### Training

You can train the network using `run.sh` and by customising the command line arguments in it.

### References
If you found our work helpful, please cite our work.
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