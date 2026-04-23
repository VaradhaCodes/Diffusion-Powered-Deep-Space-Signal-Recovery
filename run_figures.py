#!/usr/bin/env python3
"""
Phase 8 figure generation master runner.
Runs all 5 figure scripts. Each checks for its own output artifact
and produces a 300 DPI PNG in figures/.
"""
import subprocess, sys, os

SCRIPTS = [
    "src/figures/fig1_geometry.py",
    "src/figures/fig2_model_comparison.py",
    "src/figures/fig3_per_condition.py",
    "src/figures/fig4_training_curves.py",
    "src/figures/fig5_ablations.py",
]

ROOT = os.path.dirname(os.path.abspath(__file__))
venv_py = os.path.join(ROOT, ".venv", "bin", "python")
python  = venv_py if os.path.exists(venv_py) else sys.executable

ok, failed = [], []
for script in SCRIPTS:
    abs_script = os.path.join(ROOT, script)
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print('='*60)
    result = subprocess.run([python, abs_script], capture_output=False,
                            cwd=ROOT)
    if result.returncode == 0:
        ok.append(script)
    else:
        failed.append(script)
        print(f"[FAIL] {script} returned code {result.returncode}")

print(f"\n{'='*60}")
print(f"Done. {len(ok)}/{len(SCRIPTS)} succeeded.")
if failed:
    print("Failed:")
    for f in failed:
        print(f"  {f}")
    sys.exit(1)
else:
    print("All figures generated successfully.")
    print("Output in: figures/")
