#!/bin/bash
set -e
cd /home/itsva/deepspace_v5
source .venv/bin/activate

LOG=/home/itsva/deepspace_v5/run_500K_ft_eval.log
exec > >(tee -a "$LOG") 2>&1

echo "=============================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting 500K fine-tune + eval"
echo "=============================="

echo "--- Finetune seed 0 ---"
python src/train/train_v6b3.py finetune --size 500K --seed 0 --snr-source linear

echo "--- Finetune seed 1 ---"
python src/train/train_v6b3.py finetune --size 500K --seed 1 --snr-source linear

echo "--- Finetune seed 2 (from pretrain seed 1) ---"
python src/train/train_v6b3.py finetune --size 500K --seed 2 --from-seed 1 --snr-source linear

echo "--- Eval seed 0 ---"
python src/train/train_v6b3.py eval --size 500K --seed 0 --snr-source linear

echo "--- Eval seed 1 ---"
python src/train/train_v6b3.py eval --size 500K --seed 1 --snr-source linear

echo "--- Eval seed 2 ---"
python src/train/train_v6b3.py eval --size 500K --seed 2 --snr-source linear

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All 500K ft+eval done."
