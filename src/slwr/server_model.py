import numpy as np
import torch
import torch.nn.functional as F

from slwr.server.server_model.numpy_server_model import NumPyServerModel
from slwr.server.server_model.utils import pytorch_format

from src.utils.parameters import get_parameters, set_parameters
from src.model.architectures.utils import instantiate_model
from src.model.utils import init_optimizer
from src.utils.stochasticity import StatefulRng


class ServerModel(NumPyServerModel):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None  # instantiated in configure_fit
        self.optimizer = None  # instantiated in configure_fit
        self.round_loss = 0.
        self.stateful_rng = StatefulRng(10)

    @pytorch_format
    def serve_grad_request(self, embeddings, labels):
        embeddings, labels = embeddings.to(self.device), labels.to(self.device)
        embeddings.requires_grad_(True)

        with self.stateful_rng:
            output = self.model(embeddings)
        loss = F.cross_entropy(output, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.round_loss += loss.item()
        return embeddings.grad

    @pytorch_format
    def get_logits(self, embeddings):
        embeddings = embeddings.to(self.device)
        with torch.no_grad():
            output = self.model(embeddings)
        return output.cpu()

    def get_parameters(self):
        return get_parameters(self.model)

    def _init_server_model(self, parameters, config):
        model = instantiate_model(
            model_name=config["model_name"],
            seed=10, # weight are immediately overridden
            pretrained=False,
            num_classes=self.num_classes,
            partition="server",
            last_client_layer=config["last_client_layer"],
        )
        set_parameters(model, parameters)
        model.to(self.device)
        return model

    def configure_fit(self, parameters, config):
        assert {"lr", "optimizer_name", "last_client_layer", "model_name"} <= set(config.keys())

        self.round_loss = 0.
        self.model = self._init_server_model(parameters, config)
        self.optimizer = init_optimizer(self.model, config)
        self.model.train()

    def configure_evaluate(self, parameters, config):
        self.model = self._init_server_model(parameters, config)
        self.model.eval()

    def get_fit_result(self):
        del self.optimizer
        return get_parameters(self.model), {}

    def get_round_loss(self):
        return [np.array(self.round_loss),]
