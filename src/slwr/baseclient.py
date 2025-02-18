import os
import time

import torch

from slwr.client.numpy_client import NumPyClient

from src.utils.parameters import get_parameters, set_parameters
from src.utils.stochasticity import set_seed
from src.utils.other import import_given_string
from src.model.utils import init_optimizer


class BaseClient(NumPyClient):
    def __init__(self, model, trainset, valset, train_fn, eval_fn):
        super().__init__()
        self.model = model
        self.trainset = trainset
        self.valset = valset
        self.train_fn = import_given_string(train_fn)
        self.eval_fn = import_given_string(eval_fn)

    def get_parameters(self, config):
        # can be overridden by different algorithms
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        assert {"lr", "optimizer_name", "batch_size", "lte"} <= set(config.keys())
        set_seed(config["round"])
        self.server_model_proxy.torch()
        self.set_parameters(parameters)

        optimizer = init_optimizer(self.model, config)
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=config["batch_size"],
            shuffle=True,
            pin_memory=torch.cuda.is_available()
        )

        start_time = time.time()
        losses_dict_sum = {}
        for _ in range(config["lte"]):
            losses_dict = self.train_fn(self.model, self.server_model_proxy, trainloader, optimizer)
            if len(losses_dict_sum) == 0:
                losses_dict_sum = losses_dict
            else:
                for key in losses_dict:
                    losses_dict_sum[key] += losses_dict[key]
        return (
            self.get_parameters({}),
            len(self.trainset),
            losses_dict_sum | { "train_time": time.time() - start_time }
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.server_model_proxy.torch()

        valloader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=32,
            shuffle=False,
            pin_memory=torch.cuda.is_available()
        )

        out_dict = self.eval_fn(self.model, self.server_model_proxy, valloader)

        return out_dict["loss"], len(self.valset), out_dict
