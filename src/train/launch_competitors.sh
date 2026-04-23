#!/usr/bin/env bash
# Phase 5: Train all 3 competitor baselines × 3 seeds (~110 min total).
# Run from project root: bash src/train/launch_competitors.sh
set -e
source .venv/bin/activate
cd "$(dirname "$0")/../.."

MODELS=("bi_transformer" "bi_mamba2" "mambanet")

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "========================================"
    echo "  Competitor: $MODEL"
    echo "========================================"
    for SEED in 0 1 2; do
        echo ""
        echo "  --- seed $SEED ---"
        python src/train/train_competitor.py --model "$MODEL" --seed "$SEED" "$@"
    done
done

echo ""
echo "All competitor runs complete. Summary:"
python - <<'EOF'
import csv, statistics
from pathlib import Path

results = {}
for path in sorted(Path("results").glob("*_s*_test.csv")):
    parts = path.stem.split("_s")
    model = parts[0]
    seed  = parts[1].split("_")[0]
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["condition"] == "OVERALL":
                results.setdefault(model, []).append(float(row["ber"]))

print(f"\n{'Model':<18} {'Seeds':>25}  {'Mean':>7}  {'Std':>6}")
print("-" * 65)
for model, bers in sorted(results.items()):
    seeds_str = "  ".join(f"{b*100:.2f}%" for b in bers)
    mean = statistics.mean(bers) * 100
    std  = statistics.stdev(bers) * 100 if len(bers) > 1 else 0
    print(f"  {model:<16} {seeds_str:>25}  {mean:>6.2f}%  {std:>5.2f}%")

print(f"\n  {'baseline (Zhu)':<16} {'3.12%':>25}  {'3.12':>6}%")
print(f"  {'V5 ensemble':<16} {'2.76%':>25}  {'2.76':>6}%")
EOF
