"""Run: backend/.venv/bin/python backend/scripts/test_simulation_imports.py."""
import subprocess
import sys
import tempfile
from pathlib import Path

scripts = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as directory:
    for platform in ("parallel", "twitter", "reddit"):
        result = subprocess.run(
            [sys.executable, str(scripts / f"run_{platform}_simulation.py"), "--help"],
            cwd=directory, capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "--config" in result.stdout
print("PASS: all three simulation entry points import successfully")
