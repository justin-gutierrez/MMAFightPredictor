# scripts/train_sideband_calibrators.py
import pandas as pd
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ufcml.utils import add_vig_free_probs
from ufcml.calibration import SideBandCalibrator

def main():
    inp = Path("data/reports/holdout_predictions.csv")
    if not inp.exists():
        raise FileNotFoundError(f"Missing {inp}. Run export_cv_predictions.py first.")
    df = pd.read_csv(inp)

    # add vig-free fair probs
    df = add_vig_free_probs(df)

    cal = SideBandCalibrator().fit(df)
    out_dir = Path("data/reports/calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sideband_calibrator.joblib"
    cal.save(str(out_path))
    print("Saved side+band calibrator to:", out_path)

if __name__ == "__main__":
    main()
