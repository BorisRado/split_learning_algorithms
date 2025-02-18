import numpy as np
import torch
import torch.nn.functional as F

from slwr.server.server_model.numpy_server_model import NumPyServerModel
from slwr.server.server_model.utils import pytorch_format

from src.utils.parameters import get_parameters, set_parameters
from src.model.architectures.utils import instantiate_model, instantiate_general_model
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
        self.num_processed_batches = 0

    def _update_server_model_and_get_grad(self, embeddings, labels):
        embeddings, labels = embeddings.to(self.device), labels.to(self.device)
        embeddings.requires_grad_(True)

        with self.stateful_rng:
            output = self.model(embeddings)
        loss = F.cross_entropy(output, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.round_loss += loss.item()
        self.num_processed_batches += 1
        return embeddings.grad

    @pytorch_format
    def serve_grad_request(self, embeddings, labels):
        return self._update_server_model_and_get_grad(embeddings, labels)

    @pytorch_format
    def get_logits(self, embeddings):
        embeddings = embeddings.to(self.device)
        with torch.no_grad():
            output = self.model(embeddings)
        return output.cpu()

    def get_parameters(self):
        return get_parameters(self.model)

    def _init_server_model(self, parameters, config):
        model_init_kwargs = {
            "model_name": config["model_name"],
            "seed": 10, # weight are immediately overridden
            "pretrained": False,
            "num_classes": self.num_classes,
            "last_client_layer": config["last_client_layer"],
        }
        print(model_init_kwargs)
        if "server_partitions" in config:
            model = instantiate_general_model(
                client_partitions=None,
                server_partitions=config["server_partitions"],
                for_client=False,
                **model_init_kwargs,
            )
        else:
            model = instantiate_model(partition="server", **model_init_kwargs)
        set_parameters(model, parameters)
        model.to(self.device)
        return model

    def configure_fit(self, parameters, config):
        assert {"lr", "optimizer_name", "last_client_layer", "model_name"} <= set(config.keys())

        self.round_loss = 0.
        self.model = self._init_server_model(parameters, config)
        self.optimizer = init_optimizer(self.model, config)
        self.model.train()
        self.num_processed_batches = 0

    def configure_evaluate(self, parameters, config):
        self.model = self._init_server_model(parameters, config)
        self.model.eval()

    def get_fit_result(self):
        del self.optimizer
        return get_parameters(self.model), {"num_examples": self.num_processed_batches}

    def get_round_loss(self):
        return [np.array(self.round_loss) / self.num_processed_batches,]

    @pytorch_format
    def u_forward(self, embeddings):
        self.client_embeddings = embeddings.to(self.device)
        self.client_embeddings.requires_grad_(True)
        self.num_processed_batches += 1
        with self.stateful_rng:
            self.server_embeddings = self.model(self.client_embeddings)
        return self.server_embeddings

    @pytorch_format
    def u_backward(self, gradient):
        self.optimizer.zero_grad()
        self.server_embeddings.backward(gradient.to(self.device))
        self.optimizer.step()
        return self.client_embeddings.grad

    @pytorch_format
    def u_forward_inference(self, embeddings):
        embeddings = embeddings.to(self.device)
        with torch.no_grad():
            embeddings = self.model(embeddings)
        return embeddings

    @pytorch_format
    def update_server_model(self, embeddings, labels):
        self._update_server_model_and_get_grad(embeddings, labels)

    def locfedmix_gradient(self, embeddings, labels):
        # the loss is the sum of the normal CE loss and the smashed loss
        ce_gradients = self.serve_grad_request(embeddings=embeddings, labels=labels)

        embeddings = [torch.from_numpy(e)  for e in embeddings]
        labels = [F.one_hot(torch.from_numpy(l), num_classes=self.num_classes)  for l in labels]

        # get the gradients for the mixed-up data
        for idx, base_emb in enumerate(embeddings):
            other_idx = torch.randint(0, len(embeddings), (1,)).item()
            mixup_emb = (base_emb + embeddings[other_idx]) / 2
            mixup_labels = (labels[idx] + labels[other_idx]) / 2
            ce_gradients[idx] += self._update_server_model_and_get_grad(mixup_emb, mixup_labels).cpu().numpy()
        return ce_gradients

    def serve_splitavg_gradient(self, embeddings, labels):
        sizes = [e.shape[0] for e in embeddings]
        embeddings = torch.vstack([torch.from_numpy(e) for e in embeddings])
        labels = torch.hstack([torch.from_numpy(l) for l in labels])
        grad = self._update_server_model_and_get_grad(embeddings, labels).detach().cpu()
        average_gradient = grad.mean(dim=0).numpy()
        assert average_gradient.shape == embeddings[0].shape
        split_gradient = torch.split(grad, sizes, dim=0)
        split_gradient = [g.numpy() for g in split_gradient]
        with self.stateful_rng:
            avg_indices = torch.randperm(len(sizes))[:len(sizes) // 2]
        for idx in avg_indices:
            split_gradient[idx][:] = average_gradient
        for idx in avg_indices:
            assert np.all(split_gradient[idx][1:] == split_gradient[idx][0])
        return split_gradient
