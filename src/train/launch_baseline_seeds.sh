#!/usr/bin/env bash
# Run Zhu baseline training with 3 seeds for fair comparison with multi-seed competitors.
# Run from project root: bash src/train/launch_baseline_seeds.sh
set -e
source .venv/bin/activate
cd "$(dirname "$0")/../.."

for SEED in 0 1 2; do
    echo ""
    echo "========================================"
    echo "  Zhu Baseline seed=$SEED"
    echo "========================================"
    python src/train/train_baseline.py --seed "$SEED" "$@"
done

echo ""
echo "All 3 seeds complete. Building ensemble..."
python src/eval/eval_baseline_ensemble.py

echo ""
echo "Summary:"
python - <<'EOF'
import csv, statistics
from pathlib import Path

per_seed = {}
for path in sorted(Path("results").glob("baseline_s*_test.csv")):
    seed = path.stem.split("_s")[1].split("_")[0]
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["condition"] == "ALL":
                per_seed[seed] = float(row["ber"]) * 100

print(f"{'Seed':<6}  {'BER%':>8}")
for s, ber in sorted(per_seed.items()):
    print(f"  s{s}    {ber:>8.3f}%")

ens_path = Path("results/baseline_ensemble_test.csv")
if ens_path.exists():
    with open(ens_path) as f:
        for row in csv.DictReader(f):
            if row["condition"] == "ALL":
                print(f"\nEnsemble BER: {float(row['ber'])*100:.3f}%")
EOF
