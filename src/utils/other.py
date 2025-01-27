import os


def get_from_cfg_or_env_var(cfg, key, ev_key):
    if key in cfg:
        return cfg[key]
    else:
        return os.environ[ev_key]
