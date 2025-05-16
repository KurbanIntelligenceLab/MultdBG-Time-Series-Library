import os
import re
import argparse
from pathlib import Path

def update_script_content(content, seq_len, label_len, itr):
    # Remove CUDA_VISIBLE_DEVICES line
    lines = content.splitlines()
    lines = [line for line in lines if not line.strip().startswith("export CUDA_VISIBLE_DEVICES=")]
    content = "\n".join(lines)

    # Update parameters
    content = re.sub(r'--seq_len\s+\d+', f'--seq_len {seq_len}', content)
    content = re.sub(r'--label_len\s+\d+', f'--label_len {label_len}', content)
    content = re.sub(r'--itr\s+\d+', f'--itr {itr}', content)
    content = re.sub(
        r'--model_id\s+\S+_(\d+)_\d+',
        lambda m: re.sub(r'_(\d+)_\d+', f'_{seq_len}_' + m.group(0).split('_')[-1], m.group(0)),
        content
    )
    return content

def main(input_dir, seq_len, label_len, itr):
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"{input_dir} not found")

    for file_path in input_path.glob("*.sh"):
        with open(file_path, 'r') as f:
            content = f.read()

        model_match = re.search(r'model_name=(\w+)', content)
        if not model_match:
            print(f"Skipping {file_path.name} (model_name not found)")
            continue

        model_name = model_match.group(1)
        new_content = update_script_content(content, seq_len, label_len, itr)

        save_dir = Path(f"runscripts/runscripts/slurm_jobs/raw/{model_name}/{seq_len}/{label_len}")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / file_path.name

        with open(save_path, 'w') as f:
            f.write(new_content)

        print(f"Saved updated script to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing shell scripts")
    parser.add_argument("--seq_len", type=int, required=True, help="Sequence length")
    parser.add_argument("--label_len", type=int, required=True, help="Label length")
    parser.add_argument("--itr", type=int, required=True, help="Number of iterations")
    args = parser.parse_args()

    main(args.input_dir, args.seq_len, args.label_len, args.itr)
