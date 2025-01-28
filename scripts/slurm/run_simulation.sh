#!/bin/bash -l

#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2GB
#SBATCH --gpus=1
#SBATCH --partition=debug
#SBATCH --constraint=gpu_p100
#SBATCH --out=logs/gpu_p100.log

#SBATCH hetjob
#SBATCH --ntasks=16
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2GB
#SBATCH --constraint=cpu_intel_xeon_silver_4112

mamba activate slower
export PYTHONPATH=$PYTHONPATH:../slower_repo

server_ip=$(srun --het-group=0 hostname)
echo "First group hosts: ---${server_ip}---"

server_gpu_type="p100"
for last_client_layer in layer1 layer2; do
for num_clients in 1 2 4 6 8 10 12 14 16; do

folder_config="hydra.run.dir=outputs/distributed/${num_clients}_${server_gpu_type}_${last_client_layer}"
CONFIG="+num_clients=${num_clients} \
    general.num_rounds=20 \
    model.pretrained=false \
    +server_ip=\"${server_ip}:8080\" \
    partitioning.num_partitions=100 \
    strategy_config.fraction_evaluate=0.0 \
    +device_type=weak_device \
    model.last_layers.weak=${last_client_layer} \
    +server_gpu_type=$server_gpu_type"

srun \
    --ntasks=1 \
    --het-group=0 \
python -u scripts/py/run_server.py $CONFIG $folder_config &

sleep 10  # give some time to the server to start up

for ((i=0; i<num_clients; i++)); do
    srun \
        --ntasks=1 \
        --het-group=1 \
        --output=/dev/null \
        --error=/dev/null \
    python -u scripts/py/run_client.py $CONFIG +client_idx=$i &
done
wait

done
done
