import os
import argparse
from itertools import product

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', type=int, required=True, help='Input sequence length')
parser.add_argument('--label_len', type=int, required=True, help='Label length')
parser.add_argument('--pred_len', type=int, required=True, help='Prediction horizon')
args = parser.parse_args()

# --- Hyperparameter Grids ---
d_graph_values = [8, 16]
dBG_enc_layers_values = [2, 3]
node_feat_size_values = [4, 8, 16]
dBG_heads_values = [4, 8, 16]
dBG_topk_values = [4, 16, 32]

# --- SLURM Job Setup ---
os.makedirs("slurm_jobs", exist_ok=True)

base_command = (
    f"python -u run.py --task_name long_term_forecast --is_training 1 "
    f"--root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_{args.seq_len}_{args.pred_len} "
    f"--model TimesNet --data custom --features M --seq_len {args.seq_len} --label_len {args.label_len} --pred_len {args.pred_len} "
    f"--e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 "
    f"--d_model 16 --d_ff 32 --top_k 5 --des 'Exp' --itr 1 --k 4 --disc 20 25 30 "
    f"--dBG --use_gdc"
)


job_id = 0
for d_graph, layers, feat_size, heads, topk in product(
    d_graph_values, dBG_enc_layers_values, node_feat_size_values, dBG_heads_values, dBG_topk_values
):
    job_id += 1
    job_name = f"dBG_{job_id}"
    script_path = f"slurm_jobs/{job_name}.sh"

    full_command = (
        f"{base_command} --d_graph {d_graph} --dBG_enc_layers {layers} "
        f"--node_feat_size {feat_size} --dBG_heads {heads} --dBG_topk {topk}"
    )

    with open(script_path, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -p gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=meocakir@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH -A r00432

module load conda
conda activate TSLibEnv

wandb online

{full_command}
""")
