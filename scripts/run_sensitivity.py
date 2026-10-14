from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment_runner import main_sensitivity
if __name__ == "__main__": raise SystemExit(main_sensitivity())
