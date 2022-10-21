#!/bin/bash

dataset=$1
gpu=$2

echo "==========================================================================================="
echo "================================ STARTING ROLLOUT SINGLE MODEL ============================"
echo "================================ Dataset: ${dataset} ======================================"
echo "================================ GPU Number: ${gpu} ======================================="
echo "==========================================================================================="




screen -S ROLLOUT_SINGLE -dm bash -c "echo =======================================================================================================;\
                                    echo ============== Long-term model rollout encoder-decoder model on: $dataset (gpu id: $gpu) ==============;\
                                    echo =======================================================================================================;\
                                    CUDA_VISIBLE_DEVICES="$gpu" python ../pred.py model-rollout ../configs/"$dataset"/model/config1.yaml ./logs_"$dataset"_encoder-decoder_1/lightning_logs/checkpoints NA NA 60; \
                                    CUDA_VISIBLE_DEVICES="$gpu" python ../pred.py model-rollout ../configs/"$dataset"/model/config2.yaml ./logs_"$dataset"_encoder-decoder_2/lightning_logs/checkpoints NA NA 60; \
                                    CUDA_VISIBLE_DEVICES="$gpu" python ../pred.py model-rollout ../configs/"$dataset"/model/config3.yaml ./logs_"$dataset"_encoder-decoder_3/lightning_logs/checkpoints NA NA 60; \
                                    echo ============== DONE Encoder-decoder rollout ==============;\
                                    echo;\
                                    echo =======================================================================================================;\
                                    echo ============== Long-term model rollout encoder-decoder-64 model on: $dataset (gpu id: $gpu) ===========;\
                                    echo =======================================================================================================;\
                                    CUDA_VISIBLE_DEVICES="$gpu" python ../pred.py model-rollout ../configs/"$dataset"/model64/config1.yaml ./logs_"$dataset"_encoder-decoder-64_1/lightning_logs/checkpoints NA NA 60; \
                                    CUDA_VISIBLE_DEVICES="$gpu" python ../pred.py model-rollout ../configs/"$dataset"/model64/config2.yaml ./logs_"$dataset"_encoder-decoder-64_2/lightning_logs/checkpoints NA NA 60; \
                                    CUDA_VISIBLE_DEVICES="$gpu" python ../pred.py model-rollout ../configs/"$dataset"/model64/config3.yaml ./logs_"$dataset"_encoder-decoder-64_3/lightning_logs/checkpoints NA NA 60; \
                                    echo ============== DONE Encoder-decoder-64 rollout ==============;\
                                    echo;\
                                    echo =======================================================================================================;\
                                    echo ============== Long-term model rollout refine-64 model on: $dataset (gpu id: $gpu) ====================;\
                                    echo =======================================================================================================;\
                                    CUDA_VISIBLE_DEVICES="$gpu" python ../pred.py model-rollout ../configs/"$dataset"/refine64/config1.yaml ./logs_"$dataset"_encoder-decoder-64_1/lightning_logs/checkpoints ./logs_"$dataset"_refine-64_1/lightning_logs/checkpoints NA 60; \
                                    CUDA_VISIBLE_DEVICES="$gpu" python ../pred.py model-rollout ../configs/"$dataset"/refine64/config2.yaml ./logs_"$dataset"_encoder-decoder-64_2/lightning_logs/checkpoints ./logs_"$dataset"_refine-64_2/lightning_logs/checkpoints NA 60; \
                                    CUDA_VISIBLE_DEVICES="$gpu" python ../pred.py model-rollout ../configs/"$dataset"/refine64/config3.yaml ./logs_"$dataset"_encoder-decoder-64_3/lightning_logs/checkpoints ./logs_"$dataset"_refine-64_3/lightning_logs/checkpoints NA 60; \
                                    echo ============== DONE Encoder-decoder-64 rollout ==============;\
                                    echo;\
                                    echo ============================================================================================;\
                                    echo ========= Evaluating stability of long term prediction on: $dataset (gpu id: $gpu) =========;\
                                    echo ============================================================================================;\
                                    CUDA_VISIBLE_DEVICES="$gpu" python ../stability.py ../configs/"$dataset"/latentpred/config1.yaml ./logs_"$dataset"_latent-prediction_1/lightning_logs/checkpoints/ ./logs_"$dataset"_encoder-decoder-64_1/lightning_logs/checkpoints/ ./logs_"$dataset"_refine-64_1/lightning_logs/checkpoints/ ./logs_"$dataset"_encoder-decoder_1/prediction_long_term/model_rollout/ 60; \
                                    CUDA_VISIBLE_DEVICES="$gpu" python ../stability.py ../configs/"$dataset"/latentpred/config1.yaml ./logs_"$dataset"_latent-prediction_1/lightning_logs/checkpoints/ ./logs_"$dataset"_encoder-decoder-64_1/lightning_logs/checkpoints/ ./logs_"$dataset"_refine-64_1/lightning_logs/checkpoints/ ./logs_"$dataset"_encoder-decoder-64_1/prediction_long_term/model_rollout/ 60; \
                                    CUDA_VISIBLE_DEVICES="$gpu" python ../stability.py ../configs/"$dataset"/latentpred/config1.yaml ./logs_"$dataset"_latent-prediction_1/lightning_logs/checkpoints/ ./logs_"$dataset"_encoder-decoder-64_1/lightning_logs/checkpoints/ ./logs_"$dataset"_refine-64_1/lightning_logs/checkpoints/ ./logs_"$dataset"_refine-64_1/prediction_long_term/model_rollout/ 60; \
                                    CUDA_VISIBLE_DEVICES="$gpu" python ../stability.py ../configs/"$dataset"/latentpred/config2.yaml ./logs_"$dataset"_latent-prediction_2/lightning_logs/checkpoints/ ./logs_"$dataset"_encoder-decoder-64_2/lightning_logs/checkpoints/ ./logs_"$dataset"_refine-64_2/lightning_logs/checkpoints/ ./logs_"$dataset"_encoder-decoder_2/prediction_long_term/model_rollout/ 60; \
                                    CUDA_VISIBLE_DEVICES="$gpu" python ../stability.py ../configs/"$dataset"/latentpred/config2.yaml ./logs_"$dataset"_latent-prediction_2/lightning_logs/checkpoints/ ./logs_"$dataset"_encoder-decoder-64_2/lightning_logs/checkpoints/ ./logs_"$dataset"_refine-64_2/lightning_logs/checkpoints/ ./logs_"$dataset"_encoder-decoder-64_2/prediction_long_term/model_rollout/ 60; \
                                    CUDA_VISIBLE_DEVICES="$gpu" python ../stability.py ../configs/"$dataset"/latentpred/config2.yaml ./logs_"$dataset"_latent-prediction_2/lightning_logs/checkpoints/ ./logs_"$dataset"_encoder-decoder-64_2/lightning_logs/checkpoints/ ./logs_"$dataset"_refine-64_2/lightning_logs/checkpoints/ ./logs_"$dataset"_refine-64_2/prediction_long_term/model_rollout/ 60; \
                                    CUDA_VISIBLE_DEVICES="$gpu" python ../stability.py ../configs/"$dataset"/latentpred/config3.yaml ./logs_"$dataset"_latent-prediction_3/lightning_logs/checkpoints/ ./logs_"$dataset"_encoder-decoder-64_3/lightning_logs/checkpoints/ ./logs_"$dataset"_refine-64_3/lightning_logs/checkpoints/ ./logs_"$dataset"_encoder-decoder_3/prediction_long_term/model_rollout/ 60; \
                                    CUDA_VISIBLE_DEVICES="$gpu" python ../stability.py ../configs/"$dataset"/latentpred/config3.yaml ./logs_"$dataset"_latent-prediction_3/lightning_logs/checkpoints/ ./logs_"$dataset"_encoder-decoder-64_3/lightning_logs/checkpoints/ ./logs_"$dataset"_refine-64_3/lightning_logs/checkpoints/ ./logs_"$dataset"_encoder-decoder-64_3/prediction_long_term/model_rollout/ 60; \
                                    CUDA_VISIBLE_DEVICES="$gpu" python ../stability.py ../configs/"$dataset"/latentpred/config3.yaml ./logs_"$dataset"_latent-prediction_3/lightning_logs/checkpoints/ ./logs_"$dataset"_encoder-decoder-64_3/lightning_logs/checkpoints/ ./logs_"$dataset"_refine-64_3/lightning_logs/checkpoints/ ./logs_"$dataset"_refine-64_3/prediction_long_term/model_rollout/ 60; \
                                    echo ============== DONE Evaluating Stability of Prediction ==============;\
                                    echo;\
                                    echo ===========================================================================================;\
                                    echo ================================ STARTING ROLLOUT SINGLE MODEL ============================;\
                                    echo ================================ Dataset: ${dataset} ======================================;\
                                    echo ================================ GPU Number: ${gpu} =======================================;\
                                    echo ===========================================================================================;\
                                    exec sh";