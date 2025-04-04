import time

import wandb
import numpy as np
import pandas as pd
from flwr.common import (
    FitIns,
    EvaluateIns,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from slbd.server.strategy.strategy import Strategy as SlbdStrategy
from slbd.common import (
    ServerModelEvaluateIns,
    ServerModelFitIns,
)
from slbd.server.server_model.utils import ClientRequestGroup

from src.utils.parameters import get_parameters


class Strategy(SlbdStrategy):
    def __init__(
        self,
        num_clients,
        model,
        server_model_train_config,
        server_model_evaluate_config,
        client_train_config,
        fraction_fit,
        fraction_evaluate,
        init_server_model_fn,
    ):
        self.num_clients = num_clients
        self.server_model_train_config = server_model_train_config
        self.client_train_config = client_train_config
        self.num_train_clients = int(fraction_fit * self.num_clients)
        self.num_evaluate_clients = int(fraction_evaluate * self.num_clients)
        self.init_server_model_fn = init_server_model_fn
        self.server_model_evaluate_config = server_model_evaluate_config

        self.client_to_num_parameters = {}
        self.client_to_last_layer = {}

        self.whole_model_parameters = get_parameters(model)
        self.required_server_models = set()
        self.trained_client_parameters = {}
        self.start_training_time = None
        self._rount_start_time = None

    def init_server_model_fn(self):
        return self.init_server_model_fn().to_server_model()

    def initialize_parameters(self, client_manager):
        client_manager.wait_for(self.num_clients)

        table_data = []
        for client_proxy in client_manager.all().values():
            table_data.append(client_manager.get_het_client_properties(client_proxy))

        if wandb.run is not None:
            props_df = pd.DataFrame(table_data)
            wandb.run.summary["clients"] = wandb.Table(dataframe=props_df)

        return []

    def initialize_server_parameters(self):
        return []

    def _configure_clients(self, server_round, client_manager, clients, is_train):
        self._rount_start_time = time.time()
        ins_cls = FitIns if is_train else EvaluateIns

        all_ins = []
        self.required_server_models = []
        client_config = self.client_train_config.copy() if is_train else {}
        client_config["round"] = server_round

        for client in clients:
            client_props = client_manager.get_het_client_properties(client)
            num_layers = client_props["num_client_params"]
            last_layer = client_props["last_client_layer"]

            arrays = self.whole_model_parameters[:num_layers]

            self.required_server_models.append((client.cid, num_layers, last_layer))

            fit_ins = (
                client,
                ins_cls(ndarrays_to_parameters(arrays), client_config)
            )
            all_ins.append(fit_ins)
        return all_ins

    def configure_fit(self, server_round, parameters, client_manager):
        if server_round == 1:
            assert self.start_training_time is None
            self.start_training_time = time.time()
        assert self.start_training_time is not None

        clients = client_manager.sample(self.num_train_clients)
        return self._configure_clients(server_round, client_manager, clients, is_train=True)

    def configure_evaluate(self, server_round, parameters, client_manager):
        clients = client_manager.sample(self.num_evaluate_clients)
        return self._configure_clients(server_round, client_manager, clients, is_train=False)

    def _configure_server_models(self, is_train):
        all_ins = []
        ins_cls = ServerModelFitIns if is_train else ServerModelEvaluateIns
        config = self.server_model_train_config if is_train else self.server_model_evaluate_config
        for cid, num_layers, last_layer in self.required_server_models:
            if num_layers == len(self.whole_model_parameters):
                # client will fully train a model locally
                # => no need to instantiate a server model for these clients
                continue

            ins = ins_cls(
                self.whole_model_parameters[num_layers:],
                {"last_client_layer": last_layer} | config,
                sid=cid
            )

            all_ins.append(ins)
        return all_ins

    def configure_server_fit(self, server_round, parameters, cids):
        return self._configure_server_models(is_train=True)

    def configure_server_evaluate(self, server_round, parameters, cids):
        return self._configure_server_models(is_train=False)

    def aggregate_fit(self, server_round, results, failures):
        _ = (server_round, failures,)

        for client, fit_res in results:
            self.trained_client_parameters[client.cid] = (
                parameters_to_ndarrays(fit_res.parameters),
                fit_res.num_examples
            )
        aggregated_metrics = aggregated_metrics = self._aggregate_custom_metrics(results)
        print("Current training loss", aggregated_metrics["train_loss"])
        return [], aggregated_metrics

    def _aggregate_custom_metrics(self, results):
        tot_num_examples = sum([r[1].num_examples for r in results])
        keys = results[0][1].metrics.keys()
        metrics = {
            k: sum([r.metrics[k] * r.num_examples for _, r in results]) / tot_num_examples
            for k in keys
        }
        metrics_stds = {
            f"{k}_std": np.std([r.metrics[k] for _, r in results]).item()
            for k in keys
        }
        metrics["elapsed_time"] = time.time() - self.start_training_time
        metrics["round_time"] = time.time() - self._rount_start_time
        if wandb.run is not None:
            wandb.log(metrics)
        return metrics | metrics_stds

    def aggregate_server_fit(self, server_round, results):
        _ = (server_round, )
        for res in results:
            self.trained_client_parameters[res.sid][0].extend(res.parameters)

        weighted_results = list(self.trained_client_parameters.values())
        self.whole_model_parameters = aggregate(weighted_results)

        self.trained_client_parameters = {} # clean resources
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        _ = (server_round, failures, )
        loss_aggregated = weighted_loss_avg([
            (evaluate_res.num_examples, evaluate_res.loss)
            for _, evaluate_res in results
        ])

        aggregated_metrics = self._aggregate_custom_metrics(results)
        print("Current test accuracy", aggregated_metrics["accuracy"])
        return loss_aggregated, aggregated_metrics

    def route_client_request(self, cid, method_name):
        # each client batc is processed independently
        request_group = ClientRequestGroup(sid=cid)
        request_group.mark_as_ready()
        return request_group, None

    def mark_ready_requests(self):
        return None

    def mark_client_as_done(self, cid):
        return None
