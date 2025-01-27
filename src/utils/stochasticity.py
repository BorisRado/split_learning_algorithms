import numpy as np
import random
import torch


def set_seed(seed=1602):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class TempRng():
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        # store current RNG state
        self.np_state = np.random.get_state()
        self.random_state = random.getstate()
        self.torch_state = torch.random.get_rng_state()

        # set new rng state
        set_seed(seed=self.seed)

    def __exit__(self, exception_type, exception_value, exception_traceback):
        # restore previous version
        np.random.set_state(self.np_state)
        random.setstate(self.random_state)
        torch.random.set_rng_state(self.torch_state)
