import wandb
from hydra.core.config_store import OmegaConf


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_value(config, key):
    key_list = key.split(".")
    v = config
    for k in key_list:
        v = v[k]
    return v


def init_wandb(cfg, further_config_dict):
    config = OmegaConf.to_container(cfg)
    flat_config = flatten_dict(config) | further_config_dict
    wandb.init(
        config=flat_config,
    )


def finish_wandb():
    if wandb.run is not None:
        wandb.finish()
