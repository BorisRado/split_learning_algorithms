#!/bin/bash -l

#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=cpu_intel_xeon_silver_4112

# fill the remaining as necessary
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2GB
#SBATCH --out=logs/centralized_cpu.log


mamba activate slower
export PYTHONPATH=$PYTHONPATH:../slower_repo


CONFIG="general.num_rounds=20 \
    model.pretrained=false \
    partitioning.num_partitions=100"

folder="cpu_full_model"
folder_config="hydra.run.dir=outputs/centralized/${folder}"
srun python -u scripts/py/train_model_centralized.py $CONFIG $folder_config

folder="cpu_layer1"
folder_config="hydra.run.dir=outputs/centralized/${folder}"
srun python -u scripts/py/train_model_centralized.py $CONFIG +last_client_layer=layer1 $folder_config

folder="cpu_layer2"
folder_config="hydra.run.dir=outputs/centralized/${folder}"
srun python -u scripts/py/train_model_centralized.py +last_client_layer=layer2 $CONFIG $folder_config
