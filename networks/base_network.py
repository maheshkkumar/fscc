import logging
from collections import OrderedDict

from dataloader.train_dataloader import GetDataLoader
from .backbone import CSRMetaNetwork
from .network_utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BaseNetwork(CSRMetaNetwork):
    def __init__(self, loss_function, base_updates, base_lr, base_batch, meta_batch, num_of_channels=3):
        super(BaseNetwork, self).__init__(loss_function, num_of_channels)

        self.loss_function = loss_function
        self.base_updates = base_updates
        self.base_lr = base_lr
        self.base_batch = base_batch
        self.meta_batch = meta_batch
        self.num_of_channels = num_of_channels
        self.get_loader = GetDataLoader()

        for param in self.frontend.parameters():
            param.requires_grad = False

    def network_forward(self, x, weights=None):
        return super(BaseNetwork, self).forward(x, weights)

    def forward_pass(self, _input, _output, weights=None):
        _input = _input.to(device)
        _target = _output.float().to(device)

        output = self.network_forward(_input, weights)
        loss = self.loss_function(output, _target)
        return loss, output

    def forward(self, task):
        gradients = None

        train_loader = self.get_loader.get_data(task)
        validation_loader = self.get_loader.get_data(task, mode='validation')

        # testing the base network before training
        train_pre_loss, train_pre_accuracy, train_pre_mse = evaluate(self, train_loader, mode='training')
        validation_pre_accuracy, validation_pre_mse = evaluate(self, validation_loader)

        base_weights = OrderedDict(
            (name, parameter) for (name, parameter) in self.named_parameters() if parameter.requires_grad)

        for idx, data in enumerate(train_loader):

            _input, _target = data
            _input = _input.to(device)

            _target = _target.float().unsqueeze(0).to(device)

            if idx == 0:
                trainable_weights = [p for n, p in self.named_parameters() if p.requires_grad]
                loss, _ = self.forward_pass(_input, _target)
                gradients = torch.autograd.grad(loss, trainable_weights, create_graph=True)
            else:
                trainable_weights = [v for k, v in base_weights.items() if 'frontend' not in k]
                loss, _ = self.forward_pass(_input, _target, base_weights)
                gradients = torch.autograd.grad(loss, trainable_weights, create_graph=True)

            base_weights = OrderedDict((name, parameter - self.base_lr * gradient) for ((name, parameter), gradient) in
                                       zip(base_weights.items(), gradients))

        # testing the base network after training to evaluate fast adaptation
        train_post_loss, train_post_accuracy, train_post_mse = evaluate(self, train_loader, mode='training',
                                                                        weights=base_weights)
        validation_post_accuracy, validation_pose_mse = evaluate(self, validation_loader,
                                                                 weights=base_weights)

        logging.info("==========================")
        logging.info("(Meta-training) pre train loss: {}, MAE: {}, MSE: {}".format(train_pre_loss, train_pre_accuracy,
                                                                                   train_pre_mse))
        logging.info(
            "(Meta-training) post train loss: {}, MAE: {}, MSE: {}".format(train_post_loss, train_post_accuracy,
                                                                           train_post_mse))
        logging.info(
            "(Meta-training) pre-training test MAE: {}, MSE: {}".format(validation_pre_accuracy, validation_pre_mse))
        logging.info(
            "(Meta-training) post-training test MAE: {}, MSE: {}".format(validation_post_accuracy, validation_pose_mse))
        logging.info("==========================")

        # updating the meta network with the accumulated gradients from training base network
        # this operation is performed by running a dummy forward pass through the meta network on the validation dataset

        _input, _target = validation_loader.__iter__().next()
        _target = _target.float().unsqueeze(0).to(device)
        loss, _ = self.forward_pass(_input, _target, base_weights)
        loss /= self.meta_batch
        trainable_weights = {n: p for n, p in self.named_parameters() if p.requires_grad}
        gradients = torch.autograd.grad(loss, trainable_weights.values())
        meta_gradients = {name: grad for ((name, _), grad) in zip(trainable_weights.items(), gradients)}
        metrics = (train_post_loss, train_post_accuracy, train_post_mse, validation_post_accuracy, validation_pose_mse)
        return metrics, meta_gradients
