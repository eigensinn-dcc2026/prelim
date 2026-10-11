"""Fast, data-free smoke test for the KcBERT label-correlation pilot."""
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "lab" / "m3-kcbert-corr-001.py"


class KcBertCorrelationSmokeTest(unittest.TestCase):
    def test_correlation_training_helpers(self):
        # Runs with no dataset, network calls, model checkpoint, or GPU.
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--smoke"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=90,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
        self.assertIn("smoke OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
