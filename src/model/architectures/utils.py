import torch.nn as nn

from src.model.architectures.resnet import resnet18
from src.utils.stochasticity import TempRng


def instantiate_model(model_name, seed, **kwargs):
    with TempRng(seed):
        if model_name == "resnet18":
            model = resnet18(**kwargs)
        else:
            raise ValueError(f"Model {model_name} not supported")

    return model


def instantiate_general_model(client_partitions, server_partitions, for_client, **kwargs):
    partitions = client_partitions if for_client else server_partitions
    if isinstance(partitions, str):
        return instantiate_model(**kwargs, partition=partitions)
    elif isinstance(partitions, dict):
        model = nn.ModuleDict({
            k: instantiate_model(**kwargs, partition=v)
            for k, v in partitions.items()
        })
        model.is_complete_model = False
        return model
