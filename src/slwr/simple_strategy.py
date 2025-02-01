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

from slwr.server.strategy.strategy import Strategy as SlwrStrategy
from slwr.common import (
    ServerModelEvaluateIns,
    ServerModelFitIns,
)
from slwr.server.server_model.utils import ClientRequestGroup

from src.utils.parameters import get_parameters


class Strategy(SlwrStrategy):
    def __init__(
        self,
        num_clients,
        client_model,
        server_model,
        process_all_clients_as_batch,
        server_model_train_config,
        server_model_evaluate_config,
        client_train_config,
        fraction_fit,
        fraction_evaluate,
        init_server_model_fn,
        num_training_server_models,
    ):
        self.num_clients = num_clients
        self.server_model_train_config = server_model_train_config
        self.client_train_config = client_train_config
        self.num_train_clients = int(fraction_fit * self.num_clients)
        self.num_evaluate_clients = int(fraction_evaluate * self.num_clients)
        self.init_server_model_fn = init_server_model_fn
        self.process_all_clients_as_batch = process_all_clients_as_batch
        self.server_model_evaluate_config = server_model_evaluate_config
        self.num_training_server_models = num_training_server_models \
            if num_training_server_models > 0 else self.num_train_clients

        self.client_model_params = get_parameters(client_model)
        self.server_model_params = get_parameters(server_model)

        self.start_training_time = None
        self._rount_start_time = None
        self.cid_to_sid_mapping = {}
        self.request_group = None

    def init_server_model_fn(self):
        return self.init_server_model_fn().to_server_model()

    def initialize_parameters(self, client_manager):
        client_manager.wait_for(self.num_clients)
        return ndarrays_to_parameters(self.client_model_params)

    def initialize_server_parameters(self):
        params = self.server_model_params
        del self.server_model_params
        return params

    def _configure_clients(self, parameters, server_round, clients, is_train):
        self._rount_start_time = time.time()
        ins_cls = FitIns if is_train else EvaluateIns

        all_ins = []
        client_config = self.client_train_config.copy() if is_train else {}
        client_config["round"] = server_round

        for client in clients:
            fit_ins = (
                client,
                ins_cls(parameters, client_config)
            )
            all_ins.append(fit_ins)
        return all_ins

    def configure_fit(self, server_round, parameters, client_manager):
        if server_round == 1:
            assert self.start_training_time is None
            self.start_training_time = time.time()
        assert self.start_training_time is not None

        clients = client_manager.sample(self.num_train_clients)
        self.triaining_client_ids = [c.cid for c in clients]
        assert len(self.triaining_client_ids) % self.num_training_server_models == 0
        return self._configure_clients(parameters, server_round, clients, is_train=True)

    def configure_evaluate(self, server_round, parameters, client_manager):
        clients = client_manager.sample(self.num_evaluate_clients)
        return self._configure_clients(parameters, server_round, clients, is_train=False)

    def configure_server_fit(self, server_round, parameters, cids):
        _ = (server_round,)
        unique_sids = cids[:self.num_training_server_models]
        print(unique_sids)
        sids = unique_sids * (len(cids) // self.num_training_server_models)
        print(sids)
        print(cids)
        self.cid_to_sid_mapping = {cid: sid for cid, sid in zip(cids, sids)}
        print("Number of server models", len(unique_sids))
        print(self.cid_to_sid_mapping)
        return [ServerModelFitIns(parameters, self.server_model_train_config, sid=cid) for cid in unique_sids]

    def configure_server_evaluate(self, server_round, parameters, cids):
        self.cid_to_sid_mapping = {cid: "" for cid in cids}
        return [ServerModelEvaluateIns(parameters, self.server_model_evaluate_config, sid="")]

    def aggregate_fit(self, server_round, results, failures):
        _ = (server_round, failures,)
        self._num_round_active_clients = -1
        for failure in failures:
            print("===ERROR===", str(failure))

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        aggregated_metrics = aggregated_metrics = self._aggregate_custom_metrics(results)
        print("Current training loss", aggregated_metrics["train_loss"])
        return parameters_aggregated, aggregated_metrics

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
        weights_results = [
            (res.parameters, res.config["num_examples"])
            for res in results
        ]
        parameters_aggregated = aggregate(weights_results)

        return parameters_aggregated


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
        if self.request_group is None:
            self.request_group = ClientRequestGroup(sid=self.cid_to_sid_mapping[cid])

        if (
            not self.process_all_clients_as_batch or
            self.request_group.get_num_batches() == self.num_train_clients - 1
        ):
            self.request_group.mark_as_ready()
            request_group = self.request_group
            self.request_group = None
        else:
            request_group = self.request_group
        return request_group, None

    def mark_ready_requests(self):
        return None

    def mark_client_as_done(self, cid):
        return None
