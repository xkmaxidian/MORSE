"""Run the full MORSE workflow for visium_breast; see configs/visium_breast.json."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from morse.cli import main

if __name__ == "__main__":
    main("visium_breast", ROOT)
