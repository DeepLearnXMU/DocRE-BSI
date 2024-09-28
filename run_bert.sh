#! /bin/bash
export CUDA_VISIBLE_DEVICES=2
python3 -u train.py \
  --data_dir ./data \
  --transformer_type bert \
  --model_name_or_path ./model/bert \
  --save_path ./model/save_model/1016/ \
  --load_path ./model/save_model/1016/ \
  --train_file train_annotated.json \
  --dev_file dev.json \
  --test_file test.json \
  --train_batch_size 14 \
  --test_batch_size 20 \
  --gradient_accumulation_steps 1 \
  --num_labels 4 \
  --num_decoder_layers 2 \
  --learning_rate 7e-5 \
  --max_grad_norm 1.0 \
  --warmup_ratio 0.06 \
  --num_train_epochs 1002 \
  --seed 66 \
  --num_class 97 \
  | tee logs/1016.train.log 2>&1
