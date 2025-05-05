export CUDA_VISIBLE_DEVICES=0

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
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 8 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 5 \
  --top_k 5 \
  --k 4 \
  --disc 20 \
  --d_graph 8 \
  --learning_rate 0.00005 \
  --dBG > /dev/null 2>&1 &

export CUDA_VISIBLE_DEVICES=1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 8 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 5 \
  --top_k 5 \
  --k 4 \
  --disc 20 \
  --d_graph 8 \
  --learning_rate 0.00005 \
  --dBG > /dev/null 2>&1 &

export CUDA_VISIBLE_DEVICES=2


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 8 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 5 \
  --top_k 5 \
    --k 4 \
  --disc 20 \
  --d_graph 8 \
  --learning_rate 0.00005 \
  --dBG > /dev/null 2>&1 &

export CUDA_VISIBLE_DEVICES=3


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 8 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 5 \
  --top_k 5 \
  --k 4 \
  --disc 20 \
  --d_graph 8 \
  --learning_rate 0.00005 \
  --dBG > /dev/null 2>&1 &