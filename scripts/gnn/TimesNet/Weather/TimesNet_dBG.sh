export CUDA_VISIBLE_DEVICES=4

model_name=TimesNet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 16 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 5 \
  --k 4 \
  --disc 15 \
  --d_graph 16 \
  --learning_rate 0.00005 \
  --dBG > /dev/null 2>&1 &

export CUDA_VISIBLE_DEVICES=5

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 16 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 5 \
  --train_epochs 1 \
  --k 4 \
  --disc 15 \
  --d_graph 16 \
  --learning_rate 0.00005 \
  --dBG > /dev/null 2>&1 &

export CUDA_VISIBLE_DEVICES=6

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 16 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 5 \
  --k 4 \
  --disc 15 \
  --d_graph 16 \
  --learning_rate 0.00005 \
  --dBG > /dev/null 2>&1 &

export CUDA_VISIBLE_DEVICES=7

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 16 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 5 \
  --train_epochs 1 \
  --k 4 \
  --disc 15 \
  --d_graph 16 \
  --learning_rate 0.00005 \
  --dBG > /dev/null 2>&1 &