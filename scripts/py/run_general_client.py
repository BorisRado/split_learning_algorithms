import hydra
import torch.nn as nn
from omegaconf import OmegaConf

from slbd.client.app import start_client

from src.data.loading import get_dataset_from_cfg
from src.model.architectures.utils import instantiate_general_model
from src.utils.environment_variables import EnvironmentVariables as EV
from src.utils.stochasticity import set_seed
from src.utils.other import get_from_cfg_or_env_var, set_torch_flags
from src.slwr.baseclient import BaseClient as Client



@hydra.main(version_base=None, config_path="../../conf", config_name="multialgo_config")
def run(cfg):
    print(OmegaConf.to_yaml(cfg))
    set_torch_flags()
    set_seed(cfg.general.seed)

    client_idx = int(get_from_cfg_or_env_var(cfg, "client_idx", EV.CLIENT_ID))
    server_ip = get_from_cfg_or_env_var(cfg, "server_ip", EV.SERVER_ADDRESS)

    dataset = get_dataset_from_cfg(
        cfg.dataset,
        cfg.partitioning,
        cfg.general.seed,
        client_idx
    )

    model = instantiate_general_model(
        **OmegaConf.to_container(cfg.algorithm.model),
        model_name=cfg.model.model_name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.dataset.num_classes,
        seed=cfg.general.seed,
        for_client=True
    )

    client = Client(
        model=model,
        trainset=dataset["train"],
        valset=dataset["test"],
        **OmegaConf.to_container(cfg.algorithm.client_kwargs),
    )

    start_client(
        server_address=server_ip,
        client=client,
    )




if __name__ == "__main__":
    run()
