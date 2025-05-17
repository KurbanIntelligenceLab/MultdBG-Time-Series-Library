#!/bin/bash
#SBATCH -J dBG_ETTh1_96
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
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 12 \
  --label_len 6 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 5 \
  --top_k 5 --k 4 --disc 20 25 30 --dBG --use_gdc --d_graph 16 --dBG_enc_layers 2 --node_feat_size 4 --dBG_heads 16 --dBG_topk 32
