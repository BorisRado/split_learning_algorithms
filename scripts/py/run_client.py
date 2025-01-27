import os

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from slwr.client.app import start_client

from src.data.loading import get_dataset_from_cfg
from src.model.architectures.utils import instantiate_model
from src.utils.environment_variables import EnvironmentVariables as EV
from src.utils.stochasticity import set_seed
from src.utils.other import get_from_cfg_or_env_var
from src.slwr.client import Client



@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run(cfg):
    print(OmegaConf.to_yaml(cfg))

    client_idx = get_from_cfg_or_env_var(cfg, "client_idx", EV.CLIENT_ID)
    server_ip = get_from_cfg_or_env_var(cfg, "server_ip", EV.SERVER_ADDRESS)
    device_type = get_from_cfg_or_env_var(cfg, "device_type", EV.DEVICE_TYPE)
    device_capacity = cfg.device_capacities[device_type]
    last_client_layer = cfg.model.last_layers[device_capacity]

    dataset = get_dataset_from_cfg(
        cfg.dataset,
        cfg.partitioning,
        cfg.general.seed,
        client_idx
    )

    model = instantiate_model(
        model_name=cfg.model.model_name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.dataset.num_classes,
        partition="client",
        seed=cfg.general.seed,
        last_client_layer=last_client_layer,
    )

    client = Client(
        model=model,
        trainset=dataset["train"],
        valset=dataset["test"],
        last_client_layer=last_client_layer,
    )
    start_client(
        server_address=server_ip,
        client=client,
    )




if __name__ == "__main__":
    run()
