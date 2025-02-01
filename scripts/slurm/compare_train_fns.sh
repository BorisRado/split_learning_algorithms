#!/bin/bash -l

#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1GB
#SBATCH --gpus=1
#SBATCH --constraint=gpu_a100
#SBATCH --out=logs/train_fns_a100.log

#SBATCH hetjob
#SBATCH --ntasks=8
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2GB
#SBATCH --constraint=cpu_intel_xeon_silver_4112

mamba activate slower
export PYTHONPATH=$PYTHONPATH:../slower_repo

num_clients=8

server_ip=$(srun --het-group=0 hostname)
echo "First group hosts: ---${server_ip}---"

for algorithm in splitfedv1 splitfedv2 streamsl ushaped fsl locfedmix; do
for pretrained in true false; do

folder_config="hydra.run.dir=outputs/distributed/${algorithm}_${pretrained}"

CONFIG="+num_clients=${num_clients} \
    general.num_rounds=40 \
    model.pretrained=$pretrained \
    +server_ip=\"${server_ip}:8080\" \
    partitioning.num_partitions=50   \
    algorithm=$algorithm \
    +log_to_wandb=true \
    strategy_config.fraction_evaluate=1.0"

srun \
    --ntasks=1 \
    --het-group=0 \
python -u scripts/py/run_general_server.py $CONFIG $folder_config &

sleep 10  # give some time to the server to start up

for ((i=0; i<num_clients; i++)); do
    srun \
        --ntasks=1 \
        --het-group=1 \
        --nodes=1 \
        --output=/dev/null \
        --error=/dev/null \
    python -u scripts/py/run_general_client.py $CONFIG +client_idx=$i &
done

wait

done
done
