import time

import torch

from slwr.client.numpy_client import NumPyClient

from src.utils.parameters import get_parameters, set_parameters
from src.model.utils import init_optimizer
from src.model.training_procedures import train_ce
from src.model.evaluation_procedures import evaluate_model


class Client(NumPyClient):
    def __init__(self, model, trainset, valset, last_client_layer):
        super().__init__()
        self.model = model
        self.trainset = trainset
        self.valset = valset
        self.last_client_layer = last_client_layer

    def get_parameters(self, config):
        raise Exception("Should not be called")

    def fit(self, parameters, config):
        assert {"lr", "optimizer_name", "batch_size", "lte"} <= set(config.keys())
        self.server_model_proxy.torch()
        set_parameters(self.model, parameters)
        optimizer = init_optimizer(self.model, config)
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=config["batch_size"],
            shuffle=True,
            pin_memory=torch.cuda.is_available()
        )

        loss = 0.
        start_time = time.time()
        for _ in range(config["lte"]):
            loss += train_ce(self.model, self.server_model_proxy, trainloader, optimizer)

        return (
            get_parameters(self.model),
            len(self.trainset),
            {
                "loss": loss / config["lte"],
                "train_time": time.time() - start_time
            }
        )

    def evaluate(self, parameters, config):
        _ = (config, )
        set_parameters(self.model, parameters)
        self.server_model_proxy.torch()

        valloader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=32,
            shuffle=False,
            pin_memory=torch.cuda.is_available()
        )

        out_dict = evaluate_model(self.model, self.server_model_proxy, valloader)

        return out_dict["loss"], len(self.valset), out_dict

    def get_properties(self, config):
        _ = (config, )
        return {
            "last_client_layer": self.last_client_layer,
            "num_client_params": len(get_parameters(self.model))
        }
