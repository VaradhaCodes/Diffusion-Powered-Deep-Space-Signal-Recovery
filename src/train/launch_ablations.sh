#!/usr/bin/env bash
# Phase 6: Ablation runs — 3 variants × 3 seeds = 9 runs (~75 min).
# Run from project root: bash src/train/launch_ablations.sh
set -e
source .venv/bin/activate
cd "$(dirname "$0")/../.."

echo "=== Ablation A1: MambaNet-NoFiLM (remove SNR conditioning) ==="
for SEED in 0 1 2; do
    echo "  --- seed $SEED ---"
    python src/train/train_competitor.py --model mambanet_no_film --seed "$SEED"
done

echo ""
echo "=== Ablation A2: MambaNet-2ch (remove feature engineering, raw IQ only) ==="
for SEED in 0 1 2; do
    echo "  --- seed $SEED ---"
    python src/train/train_competitor.py --model mambanet_2ch --seed "$SEED"
done

echo ""
echo "=== Ablation A3: MambaNet-NoPretrain (finetune on Zhu only, no synthetic) ==="
for SEED in 0 1 2; do
    echo "  --- seed $SEED ---"
    python src/train/train_competitor.py --model mambanet_no_pretrain --seed "$SEED" \
        --skip-pretrain
done

echo ""
echo "All ablation runs complete. Computing ensemble BERs ..."
python - <<'PYEOF'
import csv, sys, torch, statistics
from pathlib import Path

ROOT = Path('.')
sys.path.insert(0, str(ROOT))
DEVICE = torch.device('cuda')

from src.models.competitors import build_model
from src.data_zhu import zhu_test_dataset, TEST_CONDITIONS
from src.train.train_v5 import _calibrate_snr_estimator, estimate_snr
from torch.utils.data import DataLoader, Subset

slope, intercept = _calibrate_snr_estimator()
TEST_N = 700

ablations = {
    'mambanet_no_film':     ('mambanet_no_film',     False),
    'mambanet_2ch':         ('mambanet_2ch',          False),
    'mambanet_no_pretrain': ('mambanet_no_pretrain',  False),
}

ensemble_results = {}
for abl_name, (model_name, _) in ablations.items():
    ckpts = sorted(ROOT.glob(f'checkpoints/{abl_name}_s*_ft_best.pt'))
    if not ckpts:
        print(f"  No checkpoints for {abl_name}")
        continue
    models = []
    for ck in ckpts:
        m = build_model(model_name).to(DEVICE)
        m.load_state_dict(torch.load(ck, map_location=DEVICE)['model'])
        m.eval()
        models.append(m)

    all_ber = []
    for cond in TEST_CONDITIONS:
        ds  = zhu_test_dataset(cond)
        sub = Subset(ds, list(range(TEST_N)))
        tl  = DataLoader(sub, batch_size=256, shuffle=False, num_workers=2)
        bs = n = 0
        with torch.no_grad():
            for x, y in tl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                snr   = estimate_snr(x, slope, intercept)
                lg_sum = torch.zeros(len(x), 100, device=DEVICE)
                for m in models:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        lg, _ = m(x, snr)
                    lg_sum += lg.float()
                prob = torch.sigmoid(lg_sum / len(models))
                b = ((prob > 0.5).float() != y).float().mean().item()
                bs += b * len(x); n += len(x)
        all_ber.append(bs / n)
    overall = sum(all_ber) / len(all_ber)
    ensemble_results[abl_name] = overall
    # Save
    rows = [(c, round(b, 6)) for c, b in zip(TEST_CONDITIONS, all_ber)]
    rows.append(('OVERALL', round(overall, 6)))
    csv_path = ROOT / 'results' / f'{abl_name}_ensemble_test.csv'
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerows([['condition','ber']] + rows)
    print(f"  {abl_name:<28} {overall*100:.3f}%  → {csv_path.name}")

print("\n\n=== ABLATION TABLE ===")
refs = [('MambaNet full',       0.02275, 'Phase 5 (full model)'),
        ('+ no attention (BiMamba2)', 0.02725, 'Phase 5'),
        ('Zhu baseline',        0.03120, 'paper')]
for name, ber, note in refs:
    delta = (ber - 0.02275) * 100
    print(f"  {name:<28} {ber*100:>6.3f}%  ({delta:+.3f}pp vs full)  [{note}]")

for name, ber in sorted(ensemble_results.items(), key=lambda x: x[1]):
    delta = (ber - 0.02275) * 100
    component = name.replace('mambanet_', '').replace('_', ' ')
    print(f"  - {component:<26} {ber*100:>6.3f}%  ({delta:+.3f}pp vs full)")
PYEOF
