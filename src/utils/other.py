import os
import importlib

import torch


def get_from_cfg_or_env_var(cfg, key, ev_key):
    if key in cfg:
        return cfg[key]
    else:
        return os.environ[ev_key]


def import_given_string(path):
    module_path, function_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, function_name)


def set_torch_flags():
    # these flags help us to achieve better numerical stability and hence obtain consistent
    # and reproducible results. Note, that we may still get different results if we run
    # the code on different GPUs, but so long as we use consistent GPUs, we should get
    # always the same values
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_flush_denormal(True)
