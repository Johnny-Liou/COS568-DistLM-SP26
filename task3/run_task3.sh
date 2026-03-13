#!/bin/bash
# Task 3: Distributed training with DistributedDataParallel (DDP)
# Master node is node3 (IP: 10.10.1.1), which runs with --local_rank 0
# Run this script on EACH node with the correct --local_rank:
#   node3 (10.10.1.1): bash run_task3.sh 0   ← master / rank 0
#   node0 (10.10.1.2): bash run_task3.sh 1
#   node1 (10.10.1.x): bash run_task3.sh 2
#   node2 (10.10.1.x): bash run_task3.sh 3

export GLUE_DIR=$HOME/COS568-DistLM-SP26/glue_data
export TASK_NAME=RTE
export GLOO_SOCKET_IFNAME=enp65s0f0np0   # CloudLab experimental network interface

LOCAL_RANK=${1:-0}   # first argument is the rank of this node

python3 run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --output_dir /tmp/${TASK_NAME}_3/ \
  --overwrite_output_dir \
  --master_ip 10.10.1.1 \
  --master_port 12347 \
  --world_size 4 \
  --local_rank $LOCAL_RANK
