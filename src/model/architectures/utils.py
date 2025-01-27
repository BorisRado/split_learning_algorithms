from src.model.architectures.resnet import resnet18
from src.utils.stochasticity import TempRng


def instantiate_model(model_name, seed, **kwargs):
    with TempRng(seed):
        if model_name == "resnet18":
            model = resnet18(**kwargs)
        else:
            raise ValueError(f"Model {model_name} not supported")

    return model
