import time

import hydra
from omegaconf import OmegaConf
import torch

from src.model.architectures.utils import instantiate_model
from src.model.utils import init_optimizer
from src.utils.stochasticity import set_seed
from src.data.loading import get_dataset_from_cfg
from src.model.training_procedures import train_ce
from src.model.evaluation_procedures import evaluate_model


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run(cfg):
    print(OmegaConf.to_yaml(cfg))

    dataset = get_dataset_from_cfg(cfg.dataset, cfg.partitioning, cfg.general.seed, 0)
    model = instantiate_model(
        model_name=cfg.model.model_name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.dataset.num_classes,
        partition=None,
        seed=cfg.general.seed,
        last_client_layer=None,
    )

    trainloader = torch.utils.data.DataLoader(
        dataset["train"],
        batch_size=cfg.client_train_config.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    valloader = torch.utils.data.DataLoader(
        dataset["test"],
        batch_size=cfg.client_train_config.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = init_optimizer(model, cfg.optimizer)

    set_seed(cfg.general.seed)
    train_times = []
    for _ in range(cfg.general.num_rounds):
        start_time = time.time()
        train_ce(model, None, trainloader, optimizer)
        train_times.append(time.time() - start_time)

        eval_dict = evaluate_model(model, None, valloader)
        print(eval_dict)

    exp_folder = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    with open(exp_folder + "/train_times.yaml", "w") as f:
        OmegaConf.save({"train_times": train_times}, f)


if __name__ == "__main__":
    run()
