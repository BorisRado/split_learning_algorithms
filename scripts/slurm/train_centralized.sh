#!/bin/bash -l

#SBATCH --time=20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# fill the remaining as necessary
#SBATCH ...

mamba activate test


folder_config="..."
CONFIG="general.num_rounds=5 \
    model.pretrained=false \
    partitioning.num_partitions=100"

srun scripts/py/train_model_centralized.py $CONFIG $folder_config
