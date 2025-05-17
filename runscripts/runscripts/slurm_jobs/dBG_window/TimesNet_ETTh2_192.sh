#!/bin/bash
#SBATCH -J dBG_ETTh2_192
#SBATCH -p gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=meocakir@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5:00:00
#SBATCH --mem=42G
#SBATCH -A r00432

module load conda
conda activate TSLibEnv

wandb online

model_name=TimesNet


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 12 \
  --label_len 6 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 5 --k 4 --disc 20 25 30 --dBG --use_gdc --d_graph 16 --dBG_enc_layers 3 --node_feat_size 8 --dBG_heads 16 --dBG_topk 16
