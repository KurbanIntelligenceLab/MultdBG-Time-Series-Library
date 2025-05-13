#!/bin/bash
source /l/anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /nobackup/meocakir/conda/timesnetEnv

wandb online

export CUDA_VISIBLE_DEVICES=0
nohup python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_96_96 --model TimesNet --data custom --features M --seq_len 12 --label_len 6 --pred_len 720 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 16 --d_ff 32 --top_k 5 --des 'Exp' --itr 1 \
--k 4 --disc 20 25 --d_graph 8 --learning_rate 0.0001 --dBG_enc_layers 3 --dBG --use_gdc > /dev/null 2>&1 &

