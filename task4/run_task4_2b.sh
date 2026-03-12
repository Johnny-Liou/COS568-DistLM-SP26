#!/bin/bash
# Task 4: Profiling all_reduce (based on Task 2b)
# Master node is node3 (IP: 10.10.1.1), which runs with --local_rank 0
# Run this script on EACH node with the correct --local_rank:
#   node3 (10.10.1.1): bash run_task4_2b.sh 0   ← master / rank 0
#   node0 (10.10.1.2): bash run_task4_2b.sh 1
#   node1 (10.10.1.x): bash run_task4_2b.sh 2
#   node2 (10.10.1.x): bash run_task4_2b.sh 3

export GLUE_DIR=$HOME/COS568-DistLM-SP26/glue_data
export TASK_NAME=RTE
export GLOO_SOCKET_IFNAME=enp1s0d1

LOCAL_RANK=${1:-0}

python3 run_glue_2b.py \
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
  --output_dir /tmp/${TASK_NAME}_4_2b/ \
  --master_ip 10.10.1.1 \
  --master_port 12349 \
  --world_size 4 \
  --local_rank $LOCAL_RANK
