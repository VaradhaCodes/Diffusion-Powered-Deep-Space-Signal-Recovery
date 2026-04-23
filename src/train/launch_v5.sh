#!/usr/bin/env bash
# Launch V5 training for all 3 seeds sequentially.
# Run from project root: bash src/train/launch_v5.sh
set -e
source .venv/bin/activate
cd "$(dirname "$0")/../.."

for SEED in 0 1 2; do
    echo ""
    echo "========================================"
    echo "  V5 training — seed $SEED"
    echo "========================================"
    python src/train/train_v5.py --seed "$SEED" "$@"
done

echo ""
echo "All 3 seeds complete. Summarising ..."
python - <<'EOF'
import csv, glob
from pathlib import Path

rows = []
for path in sorted(Path("results").glob("v5_s*_test.csv")):
    seed = path.stem.split("_s")[1].split("_")[0]
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["condition"] == "OVERALL":
                rows.append((f"seed={seed}", float(row["ber"])))

if rows:
    bers = [r[1] for r in rows]
    import statistics
    print(f"\n{'Seed':<12} {'BER':>8}")
    for label, b in rows:
        print(f"  {label:<10} {b*100:>6.2f}%")
    print(f"  {'mean':<10} {statistics.mean(bers)*100:>6.2f}%")
    print(f"  {'std':<10} {statistics.stdev(bers)*100:>6.2f}%")
    print(f"\nBaseline BER = 3.12%")
EOF
