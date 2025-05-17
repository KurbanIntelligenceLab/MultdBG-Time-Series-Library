model_name=TimesNet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
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
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 5 \
  --top_k 5 5 \
  --top_k 5 --k 4 --disc 20 25 30 --dBG --use_gdc --d_graph 16 --dBG_enc_layers 2 --node_feat_size 4 --dBG_heads 16 --dBG_topk 32


