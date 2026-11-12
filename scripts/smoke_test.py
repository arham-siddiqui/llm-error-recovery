"""Smoke test: unit tests (mock backend) + 2 real examples per dataset through the real backend.

    python scripts/smoke_test.py --config config.yaml
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args = sys.argv[1:]
print("[smoke] unit tests (mock backend - NOT model results)")
subprocess.run([sys.executable, "-m", "pytest", "-q", os.path.join(ROOT, "tests")], check=True)
print("[smoke] 2 examples/dataset through the configured real backend(s)")
subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_pilot.py"), "--n", "2",
                "--output-dir", "results/smoke", *args], check=True)
print("[smoke] OK")
