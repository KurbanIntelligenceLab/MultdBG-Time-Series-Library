#!/bin/bash
#SBATCH -J dBG_ETTm2_336
#SBATCH -p gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=meocakir@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
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
  --data_path ETTm2.csv \
  --model_id ETTm2_96_336 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 12 \
  --label_len 6 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 32 \
  --d_ff 32 \
  --top_k 5 \
  --itr 5 --k 4 --disc 20 25 30 --dBG --use_gdc --d_graph 16 --dBG_enc_layers 2 --node_feat_size 16 --dBG_heads 8 --dBG_topk 32

