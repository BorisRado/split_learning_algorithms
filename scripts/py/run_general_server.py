from dotenv import load_dotenv
from omegaconf import OmegaConf
import hydra
from flwr.server import ServerConfig
from slwr.server.app import start_server

from src.utils.stochasticity import set_seed
from src.utils.wandb import init_wandb, finish_wandb
from src.utils.other import get_from_cfg_or_env_var, set_torch_flags
from src.utils.environment_variables import EnvironmentVariables as EV
from src.slwr.simple_strategy import Strategy
from src.slwr.server_model import ServerModel
from src.model.architectures.utils import instantiate_general_model


@hydra.main(version_base=None, config_path="../../conf", config_name="multialgo_config")
def run(cfg):
    print(OmegaConf.to_yaml(cfg))
    set_torch_flags()
    num_clients = int(get_from_cfg_or_env_var(cfg, "num_clients", EV.NUM_CLIENTS))
    server_model_fn = lambda: ServerModel(
        num_classes=cfg.dataset.num_classes,
    )

    server_port = cfg.server_port if "server_port" in cfg else 8080
    if "log_to_wandb" in cfg and cfg.log_to_wandb:
        init_wandb(cfg, {"num_clients": num_clients})

    model_init_kwargs = {
        "model_name": cfg.model.model_name,
        "pretrained": cfg.model.pretrained,
        "num_classes": cfg.dataset.num_classes,
        "seed": cfg.general.seed,
    } | OmegaConf.to_container(cfg.algorithm.model)
    print(model_init_kwargs)
    client_model = instantiate_general_model(for_client=True, **model_init_kwargs)
    server_model = instantiate_general_model(for_client=False, **model_init_kwargs)

    server_model_args = {
        "model_name": cfg.model.model_name,
        "server_partitions": OmegaConf.to_container(cfg.algorithm.model)["server_partitions"],
        "last_client_layer": cfg.algorithm.model.last_client_layer,
    }

    optim_dict = OmegaConf.to_container(cfg.optimizer)
    strategy = Strategy(
        num_clients=num_clients,
        client_model=client_model,
        server_model=server_model,
        process_all_clients_as_batch=cfg.algorithm.process_all_clients_as_batch,
        server_model_train_config=optim_dict | server_model_args,
        server_model_evaluate_config=server_model_args,
        client_train_config=optim_dict | OmegaConf.to_container(cfg.client_train_config),
        fraction_fit=cfg.strategy_config.fraction_fit,
        fraction_evaluate=cfg.strategy_config.fraction_evaluate,
        init_server_model_fn=server_model_fn,
        num_training_server_models=cfg.algorithm.num_training_server_models
    )

    set_seed(cfg.general.seed)
    history = start_server(
        server_address=f"0.0.0.0:{server_port}",
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
