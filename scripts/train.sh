#!/bin/bash

dataset=$1
gpu=$2

echo "==========================================================================================="
echo "================================ STARTING TRAIN SCRIPT ===================================="
echo "==========================================================================================="

screen -S TRAIN -dm bash -c "echo ===========================================================================================;\
                                            echo ============================ Training encoder-decoder-64 model ============================;\
                                            echo ===========================================================================================;\
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../main.py ../configs/"$dataset"/model64/config1.yaml;\
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../main.py ../configs/"$dataset"/model64/config2.yaml;\
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../main.py ../configs/"$dataset"/model64/config3.yaml;\
                                            echo ============== DONE Training encoder-decoder-64 model ==============;
                                            echo;\
                                            echo ===========================================================================================;\
                                            echo ============================ Training encoder-decoder model ============================;\
                                            echo ===========================================================================================;\
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../main.py ../configs/"$dataset"/model/config1.yaml; \
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../main.py ../configs/"$dataset"/model/config2.yaml; \
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../main.py ../configs/"$dataset"/model/config3.yaml; \
                                            echo ============== DONE Training encoder-decoder model ==============;
                                            echo;\
                                            echo ===========================================================================================;\
                                            echo ============================ Eval encoder-decoder-64 model ================================;\
                                            echo ===========================================================================================;\
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/model64/config1.yaml ./logs_"$dataset"_encoder-decoder-64_1/lightning_logs/checkpoints NA eval-eval NA; \
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/model64/config2.yaml ./logs_"$dataset"_encoder-decoder-64_2/lightning_logs/checkpoints NA eval-eval NA; \
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/model64/config3.yaml ./logs_"$dataset"_encoder-decoder-64_3/lightning_logs/checkpoints NA eval-eval NA; \
                                            echo ============== DONE Eval encoder-decoder model ==============;
                                            echo;\
                                            echo ===========================================================================================;\
                                            echo ============================ Eval encoder-decoder model ===================================;\
                                            echo ===========================================================================================;\
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/model/config1.yaml ./logs_"$dataset"_encoder-decoder_1/lightning_logs/checkpoints NA eval-eval NA; \
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/model/config2.yaml ./logs_"$dataset"_encoder-decoder_2/lightning_logs/checkpoints NA eval-eval NA; \
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/model/config3.yaml ./logs_"$dataset"_encoder-decoder_3/lightning_logs/checkpoints NA eval-eval NA; \
                                            echo ============== DONE Eval encoder-decoder model ==============;\
                                            echo;\
                                            echo ===========================================================================================;\
                                            echo ============================ Forward Pass encoder-decoder model ===========================;\
                                            echo ===========================================================================================;\

                                            CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/model/config1.yaml ./logs_"$dataset"_encoder-decoder_1/lightning_logs/checkpoints NA eval-train NA; \
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/model/config2.yaml ./logs_"$dataset"_encoder-decoder_2/lightning_logs/checkpoints NA eval-train NA; \
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/model/config3.yaml ./logs_"$dataset"_encoder-decoder_3/lightning_logs/checkpoints NA eval-train NA; \
                                            echo ============== DONE Forward Pass encoder-decoder model ==============;\
                                            echo;\
                                            echo ===========================================================================================;\
                                            echo ============================ Forward Pass encoder-decoder-64 model ========================;\
                                            echo ===========================================================================================;\
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/model64/config1.yaml ./logs_"$dataset"_encoder-decoder-64_1/lightning_logs/checkpoints NA eval-train NA; \
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/model64/config2.yaml ./logs_"$dataset"_encoder-decoder-64_2/lightning_logs/checkpoints NA eval-train NA; \
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/model64/config3.yaml ./logs_"$dataset"_encoder-decoder-64_3/lightning_logs/checkpoints NA eval-train NA; \
                                            echo ============== DONE Forward Pass encoder-decoder 64 model ==============;\
                                            echo;\
                                            echo ===========================================================================================;\
                                            echo ============================ Estimate Intrinsic Dimension =================================;\
                                            echo ===========================================================================================;\
                                            OMP_NUM_THREADS=4 python ../analysis/eval_intrinsic_dimension.py ../configs/"$dataset"/model/config1.yaml model-latent NA; \
                                            OMP_NUM_THREADS=4 python ../analysis/eval_intrinsic_dimension.py ../configs/"$dataset"/model/config2.yaml model-latent NA; \
                                            OMP_NUM_THREADS=4 python ../analysis/eval_intrinsic_dimension.py ../configs/"$dataset"/model/config3.yaml model-latent NA; \
                                            echo ============================ DONE Estimate Intrinsic Dimension =============================;\
                                            echo;\
                                            echo ===========================================================================================;\
                                            echo ============================ Calculate Final Intrinsic Dimension ==========================;\
                                            echo ===========================================================================================;\
                                            python ../utils/dimension.py "$dataset";\
                                            echo ======================= DONE Calculate Final Intrinsic Dimension ==========================;\
                                            echo;\
                                            echo ===========================================================================================;\
                                            echo ================================= Train Refine-64  ========================================;\
                                            echo ===========================================================================================;\
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../main.py ../configs/"$dataset"/refine64/config1.yaml; \
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../main.py ../configs/"$dataset"/refine64/config2.yaml; \
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../main.py ../configs/"$dataset"/refine64/config3.yaml; \
                                            echo ================================= DONE Train Refine-64  ===================================;\
                                            echo;\
                                            echo ===========================================================================================;\
                                            echo ================================= Eval Refine-64  =========================================;\
                                            echo ===========================================================================================;\
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/refine64/config1.yaml ./logs_"$dataset"_refine-64_1/lightning_logs/checkpoints ./logs_"$dataset"_encoder-decoder-64_1/lightning_logs/checkpoints eval-eval NA; \
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/refine64/config2.yaml ./logs_"$dataset"_refine-64_2/lightning_logs/checkpoints ./logs_"$dataset"_encoder-decoder-64_2/lightning_logs/checkpoints eval-eval NA; \
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/refine64/config3.yaml ./logs_"$dataset"_refine-64_3/lightning_logs/checkpoints ./logs_"$dataset"_encoder-decoder-64_3/lightning_logs/checkpoints eval-eval NA; \
                                            echo ================================= DONE Eval Refine-64  ===================================;\
                                            echo;\
                                            echo ===========================================================================================;\
                                            echo ============================== Eval Gather Refine-64  =====================================;\
                                            echo ===========================================================================================;\
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/refine64/config1.yaml ./logs_"$dataset"_refine-64_1/lightning_logs/checkpoints ./logs_"$dataset"_encoder-decoder-64_1/lightning_logs/checkpoints eval-refine-train NA; \
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/refine64/config2.yaml ./logs_"$dataset"_refine-64_2/lightning_logs/checkpoints ./logs_"$dataset"_encoder-decoder-64_2/lightning_logs/checkpoints eval-refine-train NA; \
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/refine64/config3.yaml ./logs_"$dataset"_refine-64_3/lightning_logs/checkpoints ./logs_"$dataset"_encoder-decoder-64_3/lightning_logs/checkpoints eval-refine-train NA; \
                                            echo ============================== DONE Eval Gather Refine-64  ================================;\
                                            echo;\
                                            echo ===========================================================================================;\
                                            echo =============================== Train Latent Pred  ========================================;\
                                            echo ===========================================================================================;\
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../main.py latentpred ../configs/"$dataset"/latentpred/config1.yaml ./logs_"$dataset"_encoder-decoder-64_1/lightning_logs/checkpoints/ ./logs_"$dataset"_refine-64_1/lightning_logs/checkpoints/; \
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../main.py latentpred ../configs/"$dataset"/latentpred/config2.yaml ./logs_"$dataset"_encoder-decoder-64_2/lightning_logs/checkpoints/ ./logs_"$dataset"_refine-64_2/lightning_logs/checkpoints/; \
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../main.py latentpred ../configs/"$dataset"/latentpred/config3.yaml ./logs_"$dataset"_encoder-decoder-64_3/lightning_logs/checkpoints/ ./logs_"$dataset"_refine-64_3/lightning_logs/checkpoints/; \
                                            echo ================================ DONE Train Latent Pred  ==================================;\
                                            echo;\
                                            echo ===========================================================================================;\
                                            echo ================================ TRAINING COMPLETE!!!  ====================================;\
                                            echo ===========================================================================================;\
                                            exec sh";
