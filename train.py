import argparse
import logging
import random

import numpy as np
import torch

from networks.meta_learner import MetaLearner

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)


class InitiateTraining(object):
    def __init__(self, args):
        self.dataset = args.dataset
        self.data_path = args.data_path
        self.num_tasks = args.num_tasks
        self.num_instances = args.num_instances
        self.meta_bs = args.meta_batch
        self.base_bs = args.base_batch
        self.meta_lr = args.meta_lr
        self.base_lr = args.base_lr
        self.epochs = args.epochs
        self.base_updates = args.base_updates
        self.experiment = args.experiment
        self.meta_learner = MetaLearner(dataset=self.dataset, data_path=self.data_path, num_tasks=self.num_tasks,
                                        num_instances=self.num_instances, meta_batch=self.meta_bs,
                                        meta_lr=self.meta_lr, base_batch=self.base_bs, base_lr=self.base_lr,
                                        meta_updates=self.epochs, base_updates=self.base_updates,
                                        experiment=self.experiment)

    def start_training(self):
        self.meta_learner.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='Name of the dataset', default='WorldExpo', type=str)
    parser.add_argument('-trp', '--data_path', help='Path of the dataset', required=True, type=str)
    parser.add_argument('-nt', '--num_tasks', help='Number of tasks for training', default=10, type=int)
    parser.add_argument('-ni', '--num_instances', help='Number of instances per task for training', default=5, type=int)
    parser.add_argument('-mb', '--meta_batch', help='Batch size for meta network', default=32, type=int)
    parser.add_argument('-bb', '--base_batch', help='Batch size for base network', default=1, type=int)
    parser.add_argument('-mlr', '--meta_lr', help='Meta learning rate', default=1e-5, type=float)
    parser.add_argument('-blr', '--base_lr', help='Base learning rate', default=1e-5, type=float)
    parser.add_argument('-e', '--epochs', help='Number of training epochs', default=15000, type=int)
    parser.add_argument('-bu', '--base_updates', help='Iterations for base network to train', default=1, type=int)
    parser.add_argument('-exp', '--experiment', help='Experiment number', default=0, type=int)
    parser.add_argument('-log', '--log_name', help='Name of logging file', type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(filename=args.log_name, level=logging.INFO)
    logging.info('Started training')

    st = InitiateTraining(args)
    st.start_training()

    logging.info('Finished training')
