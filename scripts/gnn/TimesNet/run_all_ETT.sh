#!/bin/bash

# Navigate to each dataset folder and run both scripts

for dataset in ETTh1 ETTh2 ETTm1 ETTm2; do
  echo "Running scripts in $dataset..."
  bash "scripts/gnn/TimesNet/$dataset/TimesNet_${dataset}.sh"
  bash "scripts/gnn/TimesNet/$dataset/TimesNet_${dataset}_dBG.sh"
done

echo "All scripts executed."