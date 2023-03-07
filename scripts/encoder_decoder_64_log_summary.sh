#!/bin/bash

dataset=$1

python  ../log_summary.py ./logs_"$dataset"_encoder-decoder-64_1/;
python  ../log_summary.py ./logs_"$dataset"_encoder-decoder-64_2/;
python  ../log_summary.py ./logs_"$dataset"_encoder-decoder-64_3/;