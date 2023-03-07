#!/bin/bash

dataset=$1
stage=$2

rm -r ./logs_"$dataset"_"$stage"_1/;
rm -r ./logs_"$dataset"_"$stage"_2/;
rm -r ./logs_"$dataset"_"$stage"_3/;