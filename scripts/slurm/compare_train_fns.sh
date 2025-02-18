#!/bin/bash -l

#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1GB
#SBATCH --gpus=1
#SBATCH --constraint=gpu_a100
#SBATCH --out=logs/train_fns_a100_5.log

#SBATCH hetjob
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1GB

mamba activate slower
export PYTHONPATH=$PYTHONPATH:../slower_repo

num_clients=8

server_ip=$(srun --het-group=0 hostname)
echo "First group hosts: ---${server_ip}---"

server_port=8085

for algorithm in splitavg splitfedv1 splitfedv2 splitfed2com streamsl ushaped fsl locfedmix; do
for pretrained in false; do

if [ "$pretrained" = true ]; then
    num_rounds=60
else
    num_rounds=200
fi

folder_config="hydra.run.dir=outputs/distributed/${algorithm}_${pretrained}"

CONFIG="+num_clients=${num_clients} \
    +server_port=$server_port \
    general.num_rounds=$num_rounds \
    model.pretrained=$pretrained \
    +server_ip=\"${server_ip}:$server_port\" \
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
