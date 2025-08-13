#!/usr/bin/env python3
"""
Market-based model training script for UFC ML predictor.
"""

# This script will train models based on betting market data
# including odds, line movements, and market sentiment

import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import argparse

# Ensure src is on path for script execution
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ufcml.train import train_market_stacked


def main():
	parser = argparse.ArgumentParser(description="Train market-aware stacked model")
	parser.add_argument("--n-folds", type=int, default=3, help="Number of folds to use (default: 3)")
	parser.add_argument("--include-other-features", action="store_true", help="Include all features in Stage B (default: False)")
	args = parser.parse_args()

	X_path = Path("data/processed/X.parquet")
	y_path = Path("data/processed/y.parquet")
	splits_dir = Path("data/processed/splits")
	reports_dir = Path("data/reports")
	reports_dir.mkdir(parents=True, exist_ok=True)

	X = pd.read_parquet(X_path)
	y = pd.read_parquet(y_path).iloc[:, 0]

	split_files = sorted(glob.glob(str(splits_dir / "time_based_cv_fold_*.json")))
	splits = []
	for fp in split_files:
		with open(fp, "r") as f:
			d = json.load(f)
		splits.append((np.array(d["train_indices"]), np.array(d["valid_indices"])) )

	if args.n_folds < len(splits):
		splits = splits[:args.n_folds]
	print(f"Using {len(splits)} folds for stacking")

	base_params_skill = dict(
		n_estimators=2000, learning_rate=0.03, max_depth=4, min_child_weight=2,
		subsample=0.9, colsample_bytree=0.9, reg_lambda=1.5, gamma=0.1,
		eval_metric="logloss", n_jobs=-1, random_state=42, tree_method="hist",
	)
	base_params_market = dict(
		n_estimators=1500, learning_rate=0.05, max_depth=4, min_child_weight=2,
		subsample=0.9, colsample_bytree=0.9, reg_lambda=1.5, gamma=0.1,
		eval_metric="logloss", n_jobs=-1, random_state=42, tree_method="hist",
	)

	results = train_market_stacked(
		X, y, splits,
		base_params_skill=base_params_skill,
		base_params_market=base_params_market,
		include_other_features=args.include_other_features,
	)

	out_path = reports_dir / "market_cv_metrics.json"
	with open(out_path, "w") as f:
		json.dump(results, f, indent=2)
	print(f"Market-aware stacking completed. Metrics saved to: {out_path}")


if __name__ == "__main__":
	main()
