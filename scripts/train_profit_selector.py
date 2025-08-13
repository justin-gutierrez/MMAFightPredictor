# scripts/train_profit_selector.py
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ufcml.utils import add_vig_free_probs

def american_to_decimal(odds):
    o = pd.to_numeric(odds, errors="coerce")
    return np.where(o > 0, (o / 100.0) + 1.0, (100.0 / np.abs(o)) + 1.0)

def build_side_table(df: pd.DataFrame, side: str) -> pd.DataFrame:
    assert side in ("Red","Blue")
    d = df.copy()
    d = add_vig_free_probs(d)
    if side == "Red":
        p = d["ModelProb"]
        fair = d["RedFairP"]
        odds = d["RedOdds"]
        won = (d["Winner"].str.lower() == "red").astype(int)
    else:
        p = 1.0 - d["ModelProb"]
        fair = d["BlueFairP"]
        odds = d["BlueOdds"]
        won = (d["Winner"].str.lower() == "blue").astype(int)

    dec = american_to_decimal(odds)
    b = dec - 1.0
    # realized profit for $1 stake
    pl = won * b - (1 - won) * 1.0
    label = (pl > 0).astype(int)

    feat = pd.DataFrame({
        "Prob": p,
        "FairP": fair,
        "Edge": p - fair,
        "DecOdds": dec,
        "B": b,
        "FairP_Bin": pd.qcut(fair, q=10, duplicates="drop").cat.codes
    })
    feat["Label"] = label
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
    return feat

def main():
    inp = Path("data/reports/holdout_predictions.csv")
    out_dir = Path("data/reports/selector")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)

    models = {}
    for side in ("Red","Blue"):
        tbl = build_side_table(df, side)
        if tbl["Label"].nunique() < 2 or len(tbl) < 200:
            print(f"Not enough data for selector {side}. Skipping.")
            continue
        X = tbl[["Prob","FairP","Edge","DecOdds","B","FairP_Bin"]].values
        y = tbl["Label"].values
        clf = LogisticRegression(max_iter=200)
        clf.fit(X, y)
        pth = out_dir / f"{side}_selector.joblib"
        joblib.dump(clf, pth)
        models[side] = str(pth)
        print(f"Saved {side} selector to {pth}")

    print("Done. Selectors:", models)

if __name__ == "__main__":
    main()
