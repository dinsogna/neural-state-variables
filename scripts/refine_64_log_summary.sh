#!/bin/bash

dataset=$1

python ../log_summary.py ./logs_"$dataset"_refine-64_1;
echo "========================================================================================"
python ../log_summary.py ./logs_"$dataset"_refine-64_2;
echo "========================================================================================"
python ../log_summary.py ./logs_"$dataset"_refine-64_3;