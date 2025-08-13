import glob, json
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ufcml.train import get_oof_skill_predictions
from ufcml.utils import add_market_features

def load_splits(pattern="data/processed/splits/time_based_cv_fold_*.json"):
    files = sorted(glob.glob(pattern))
    splits = []
    for fp in files:
        with open(fp, "r") as f:
            d = json.load(f)
        splits.append((np.array(d["train_indices"]), np.array(d["valid_indices"])))
    if not splits:
        raise FileNotFoundError("No CV split files found.")
    return splits

def main():
    out_path = Path("data/reports/holdout_predictions.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X = pd.read_parquet("data/processed/X.parquet")
    y = pd.read_parquet("data/processed/y.parquet").iloc[:,0]
    meta = pd.read_parquet("data/processed/meta.parquet")

    splits = load_splits()

    base_params_skill = dict(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.5,
        gamma=0.1,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
    )

    base_params_market = dict(
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.5,
        gamma=0.1,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
    )

    # Stage A: build OOF skill predictions + keep per-fold skill models
    oof_skill, skill_models = get_oof_skill_predictions(
        X=X, y=y, splits=splits, base_params=base_params_skill, drop_odds=True
    )

    rows = []

    for k, (tr, va) in enumerate(splits, start=1):
        # Build market features for train/valid
        X_tr = X.iloc[tr].copy()
        X_va = X.iloc[va].copy()
        y_tr = y.iloc[tr].values.ravel()
        y_va = y.iloc[va].values.ravel()

        # Add SkillPred (OOF on train; fold-specific on valid)
        X_tr["SkillPred"] = oof_skill[tr]
        skill_model_k = skill_models[k-1]
        X_va_nodds = X_va.drop(columns=[c for c in ["RedOdds","BlueOdds"] if c in X_va.columns], errors="ignore")
        X_va["SkillPred"] = skill_model_k.predict_proba(X_va_nodds)[:,1]

        # Add market features (incl. vig-free)
        X_tr = add_market_features(X_tr, use_vig_free=True)
        X_va = add_market_features(X_va, use_vig_free=True)

        # Minimal feature set (to match training)
        keep = [c for c in ["SkillPred","RedOdds","BlueOdds","RedImpP","BlueImpP","ImpP_Diff","Odds_Diff","RedFairP","BlueFairP","FairP_Diff"] if c in X_tr.columns]
        X_tr_m = X_tr[keep].copy()
        X_va_m = X_va[keep].copy()

        # Train market model
        base_rate = float(y_tr.mean())
        params_m = base_params_market.copy()
        params_m["base_score"] = base_rate

        market_model = XGBClassifier(**params_m)
        market_model.fit(X_tr_m, y_tr, verbose=False)

        p_va = market_model.predict_proba(X_va_m)[:,1]

        # Build output rows for this fold
        meta_va = meta.iloc[va]
        # y is 1 if Red won
        winners = np.where(y.iloc[va].values.ravel() == 1, "Red", "Blue")

        part = pd.DataFrame({
            "Date": meta_va["Date"].values,
            "RedFighter": meta_va["RedFighter"].values,
            "BlueFighter": meta_va["BlueFighter"].values,
            "Winner": winners,
            "RedOdds": X.iloc[va]["RedOdds"].values if "RedOdds" in X.columns else np.nan,
            "BlueOdds": X.iloc[va]["BlueOdds"].values if "BlueOdds" in X.columns else np.nan,
            "ModelProb": p_va,
            "Fold": k,
        })
        rows.append(part)

    out_df = pd.concat(rows, ignore_index=True)
    # Ensure Date is string for CSV friendliness
    out_df["Date"] = pd.to_datetime(out_df["Date"]).dt.strftime("%Y-%m-%d")
    out_df.to_csv(out_path, index=False)
    print(f"Saved CV validation predictions to: {out_path} with shape {out_df.shape}")

if __name__ == "__main__":
    main()
