from dotenv import load_dotenv
from omegaconf import OmegaConf
import hydra
from flwr.server import ServerConfig
from slwr.server.app import start_server

from src.utils.stochasticity import set_seed
from src.utils.wandb import init_wandb, finish_wandb
from src.utils.other import get_from_cfg_or_env_var
from src.utils.environment_variables import EnvironmentVariables as EV
from src.slwr.strategy import Strategy
from src.slwr.server_model import ServerModel
from src.model.architectures.utils import instantiate_model


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run(cfg):
    print(OmegaConf.to_yaml(cfg))
    num_clients = int(get_from_cfg_or_env_var(cfg, "num_clients", EV.NUM_CLIENTS))
    server_model_fn = lambda: ServerModel(
        num_classes=cfg.dataset.num_classes,
    )

    if "log_to_wandb" in cfg and cfg.log_to_wandb:
        init_wandb(cfg, {"num_clients": num_clients})

    model = instantiate_model(
        model_name=cfg.model.model_name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.dataset.num_classes,
        partition=None,
        seed=cfg.general.seed,
        last_client_layer=None,
    )

    optim_dict = OmegaConf.to_container(cfg.optimizer)
    strategy = Strategy(
        num_clients=num_clients,
        model=model,
        server_model_train_config=optim_dict | {"model_name": cfg.model.model_name},
        server_model_evaluate_config={"model_name": cfg.model.model_name},
        client_train_config=optim_dict | OmegaConf.to_container(cfg.client_train_config),
        fraction_fit=cfg.strategy_config.fraction_fit,
        fraction_evaluate=cfg.strategy_config.fraction_evaluate,
        init_server_model_fn=server_model_fn,
    )

    set_seed(cfg.general.seed)
    history = start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=ServerConfig(num_rounds=cfg.general.num_rounds),
    )
    fit_metrics = {
        key: [x[1] for x in value]
        for key, value in history.metrics_distributed_fit.items()
    }
    eval_metrics = {
        key: [x[1] for x in value]
        for key, value in history.metrics_distributed.items()
    }

    exp_folder = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    with open(exp_folder + "/fit_metrics.yaml", "w") as f:
        OmegaConf.save(fit_metrics, f)
    with open(exp_folder + "/eval_metrics.yaml", "w") as f:
        OmegaConf.save(eval_metrics, f)
    finish_wandb()


if __name__ == "__main__":
    load_dotenv()
    run()
