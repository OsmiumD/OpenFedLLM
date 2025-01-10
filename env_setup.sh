#!/bin/bash

fingpt_dir=evaluation/FinGPT
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PYTHONPATH:$fingpt_dir
export MODELS_HOME='/mnt/data1/big_file/dingyh/models'
export OUTPUT_DIR='/home/dingyh/OpenFedLLM/output'
source $HOME/anaconda3/bin/activate fedllm
