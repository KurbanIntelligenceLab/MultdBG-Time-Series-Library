
model_name=TimeXer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_12_96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 12 \
  --label_len 6 \
  --pred_len 96 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 256 \
  --batch_size 4 \
  --des 'Exp' \
  --itr 3 \
  --patch_len 12

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_12_192 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 12 \
  --label_len 6 \
  --pred_len 192 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 256 \
  --d_ff 256 \
  --batch_size 4 \
  --des 'Exp' \
  --itr 3 \
  --patch_len 12

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_12_336 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 12 \
  --label_len 6 \
  --pred_len 336 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 256 \
  --d_ff 1024 \
  --batch_size 4 \
  --des 'Exp' \
  --itr 3 \
  --patch_len 12

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_12_720 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 12 \
  --label_len 6 \
  --pred_len 720 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 256 \
  --d_ff 512 \
  --batch_size 4 \
  --des 'Exp' \
  --itr 3 \
  --patch_len 12