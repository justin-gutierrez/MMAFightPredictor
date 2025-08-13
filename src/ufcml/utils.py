"""
Utility functions for UFC ML predictor.
"""

import numpy as np
import pandas as pd


def american_to_implied_prob(odds: pd.Series) -> pd.Series:
	"""
	Convert American odds to implied probability.
	
	Args:
		odds: Series of American odds (e.g., -150, +120)
	
	Returns:
		Series of implied probabilities in [0, 1]
	"""
	o = pd.to_numeric(odds, errors="coerce")
	pos = o > 0
	neg = o < 0
	p = pd.Series(np.nan, index=o.index)
	p[pos] = 100.0 / (o[pos] + 100.0)
	p[neg] = (-o[neg]) / ((-o[neg]) + 100.0)
	return p


def add_odds_implied_probs(X: pd.DataFrame) -> pd.DataFrame:
	"""
	Add implied probability and simple odds features if odds are present.
	
	Adds columns: RedImpP, BlueImpP, ImpP_Diff, Odds_Diff
	"""
	X = X.copy()
	if "RedOdds" in X.columns and "BlueOdds" in X.columns:
		X["RedImpP"] = american_to_implied_prob(X["RedOdds"])
		X["BlueImpP"] = american_to_implied_prob(X["BlueOdds"])
		X["ImpP_Diff"] = X["RedImpP"] - X["BlueImpP"]
		X["Odds_Diff"] = pd.to_numeric(X["RedOdds"], errors="coerce") - pd.to_numeric(X["BlueOdds"], errors="coerce")
	return X
