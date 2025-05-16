
model_name=Nonstationary_Transformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_12_96 \
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
  --des 'Exp' \
  --itr 3 \
  --p_hidden_dims 256 256 \
  --p_hidden_layers 2 \
  --d_model 128

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_12_192 \
  --model $model_name \
  --data ETTh1 \
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
  --des 'Exp' \
  --itr 3 \
  --p_hidden_dims 256 256 \
  --p_hidden_layers 2 \
  --d_model 128

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_12_336 \
  --model $model_name \
  --data ETTh1 \
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
  --itr 3 \
  --p_hidden_dims 256 256 \
  --p_hidden_layers 2 \
  --d_model 128

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_12_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 12 \
  --label_len 6 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 3 \
  --p_hidden_dims 256 256 \
  --p_hidden_layers 2 \
  --d_model 128