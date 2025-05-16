import os
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
import time

def find_scripts(base_dir):
    return list(Path(base_dir).rglob("*.sh"))

def launch_script(script_path, gpu_id):
    cmd = f"bash {script_path}"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return subprocess.Popen(cmd, shell=True, env=env)

def main(root_dir, max_gpus=8, max_jobs_per_gpu=2):
    scripts = find_scripts(root_dir)
    if not scripts:
        print("No scripts found.")
        return

    gpu_jobs = defaultdict(list)
    running_procs = []

    for script in scripts:
        # Wait if all GPUs are saturated
        while True:
            # Remove finished jobs
            running_procs = [(p, g) for p, g in running_procs if p.poll() is None]

            # Count current GPU loads
            gpu_jobs = defaultdict(list)
            for p, g in running_procs:
                gpu_jobs[g].append(p)

            # Find available GPU
            for gpu_id in range(max_gpus):
                if len(gpu_jobs[gpu_id]) < max_jobs_per_gpu:
                    proc = launch_script(script, gpu_id)
                    running_procs.append((proc, gpu_id))
                    print(f"[GPU {gpu_id}] Launched: {script}")
                    break
            else:
                time.sleep(2)  # Wait and retry if all are busy
                continue
            break

    # Wait for all to finish
    for proc, _ in running_procs:
        proc.wait()

    print("All scripts completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True, help="Directory containing .sh scripts in subdirectories")
    parser.add_argument("--gpus", type=int, default=8, help="Number of available GPUs")
    parser.add_argument("--max_jobs_per_gpu", type=int, default=1, help="Max parallel jobs per GPU")
    args = parser.parse_args()

    main(args.root_dir, max_gpus=args.gpus, max_jobs_per_gpu=args.max_jobs_per_gpu)
