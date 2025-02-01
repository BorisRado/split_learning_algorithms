import os
import importlib


def get_from_cfg_or_env_var(cfg, key, ev_key):
    if key in cfg:
        return cfg[key]
    else:
        return os.environ[ev_key]


def import_given_string(path):
    module_path, function_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, function_name)
