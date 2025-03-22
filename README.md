# Split Learning with heterogeneous client-side model sizes

Implementation for reproducing results contained in the paper `Towards a Unified Framework for Split Learning` (Radovič, Boris and Canini, Marco and Horváth, Samuel and Pejović, Veljko and Vepakomma, Praneeth) @ EuroMLSys'25.

Scripts to obtain the results are in the `scripts/slurm` folder:

* `run_simulations.sh`: run experiments for train time with respect to number of clients with the SplitFed v1 algorithm;
* `train_centralized.sh`: run full training on the devices (what in the paper is referred to as "CPU full model" and "CPU partial model");
* `compare_train_fns.sh`: run all the algorithms we implemented.

For the experiment on CoLExT, the configuration file for running the experiment is included in `colext_config.yaml`.

Note, that results are logged to `wandb`, so you have to include a `.env` file with your wandb API key.
