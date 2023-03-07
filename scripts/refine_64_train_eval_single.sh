#!/bin/bash

dataset=$1
seed=$2
gpu=$3

echo "======================================================================================="
echo "============== Training refine-64 model on: $dataset (gpu ids: $gpu) =============="
echo "======================================================================================="

screen -S train-"$dataset"-64_"$seed" -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu" python ../main.py ../configs/"$dataset"/refine64/config"$seed".yaml; \
                                                   CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/refine64/config"$seed".yaml ./logs_"$dataset"_refine-64_"$seed"/lightning_logs/checkpoints ./logs_"$dataset"_encoder-decoder-64_"$seed"/lightning_logs/checkpoints eval-eval NA; \
                                                   python ../log_summary.py ./logs_"$dataset"_refine-64_"$seed"; \
                                                   exec sh";