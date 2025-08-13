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


def implied_probs_from_american(red_odds: pd.Series, blue_odds: pd.Series) -> pd.DataFrame:
	"""Convert American odds for Red/Blue into implied probabilities (no vig removed)."""
	def to_imp(o: pd.Series) -> pd.Series:
		o = pd.to_numeric(o, errors="coerce")
		pos = o > 0
		neg = o < 0
		p = pd.Series(np.nan, index=o.index)
		p[pos] = 100.0 / (o[pos] + 100.0)
		p[neg] = (-o[neg]) / ((-o[neg]) + 100.0)
		return p

	red_p = to_imp(red_odds)
	blue_p = to_imp(blue_odds)
	return pd.DataFrame({"RedImpP": red_p, "BlueImpP": blue_p}, index=red_odds.index)


def compute_overround(red_imp: pd.Series, blue_imp: pd.Series) -> pd.Series:
	"""Overround (vig) = RedImpP + BlueImpP for a 2-way market."""
	return pd.to_numeric(red_imp, errors="coerce") + pd.to_numeric(blue_imp, errors="coerce")


def remove_vig_proportional(red_imp: pd.Series, blue_imp: pd.Series) -> pd.DataFrame:
	"""
	Remove vig proportionally:
		fair_red = red_imp / (red_imp + blue_imp)
		fair_blue = blue_imp / (red_imp + blue_imp)
	Returns NaN if inputs are invalid or sum <= 0.
	"""
	r = pd.to_numeric(red_imp, errors="coerce")
	b = pd.to_numeric(blue_imp, errors="coerce")
	s = r + b

	fair_red = r / s
	fair_blue = b / s

	mask_bad = ~np.isfinite(s) | (s <= 0)
	fair_red[mask_bad] = np.nan
	fair_blue[mask_bad] = np.nan

	return pd.DataFrame({"RedFairP": fair_red, "BlueFairP": fair_blue}, index=r.index)


def add_vig_free_probs(df: pd.DataFrame,
					   red_odds_col: str = "RedOdds",
					   blue_odds_col: str = "BlueOdds",
					   prefix: str = "") -> pd.DataFrame:
	"""
	Add implied and vig-free probabilities to df.
	Columns added:
	  {prefix}RedImpP, {prefix}BlueImpP, {prefix}Overround,
	  {prefix}RedFairP, {prefix}BlueFairP, {prefix}FairP_Diff
	"""
	df = df.copy()
	if red_odds_col not in df.columns or blue_odds_col not in df.columns:
		return df

	imp = implied_probs_from_american(df[red_odds_col], df[blue_odds_col])
	df[f"{prefix}RedImpP"] = imp["RedImpP"]
	df[f"{prefix}BlueImpP"] = imp["BlueImpP"]

	df[f"{prefix}Overround"] = compute_overround(df[f"{prefix}RedImpP"], df[f"{prefix}BlueImpP"])

	fair = remove_vig_proportional(df[f"{prefix}RedImpP"], df[f"{prefix}BlueImpP"])
	df[f"{prefix}RedFairP"] = fair["RedFairP"]
	df[f"{prefix}BlueFairP"] = fair["BlueFairP"]
	df[f"{prefix}FairP_Diff"] = df[f"{prefix}RedFairP"] - df[f"{prefix}BlueFairP"]

	return df


def add_market_features(df: pd.DataFrame,
						   red_odds_col: str = "RedOdds",
						   blue_odds_col: str = "BlueOdds",
						   use_vig_free: bool = True) -> pd.DataFrame:
	"""
	Add raw implied probs, odds diffs, and optionally vig-free (fair) probs to df.
	Keeps raw implied features for diagnostics.
	"""
	df = df.copy()
	if red_odds_col in df.columns and blue_odds_col in df.columns:
		df["RedImpP"] = american_to_implied_prob(df[red_odds_col])
		df["BlueImpP"] = american_to_implied_prob(df[blue_odds_col])
		df["ImpP_Diff"] = df["RedImpP"] - df["BlueImpP"]
		df["Odds_Diff"] = pd.to_numeric(df[red_odds_col], errors="coerce") - pd.to_numeric(df[blue_odds_col], errors="coerce")

	if use_vig_free and "RedImpP" in df.columns and "BlueImpP" in df.columns:
		fair = remove_vig_proportional(df["RedImpP"], df["BlueImpP"])
		df["RedFairP"] = fair["RedFairP"]
		df["BlueFairP"] = fair["BlueFairP"]
		df["FairP_Diff"] = df["RedFairP"] - df["BlueFairP"]

	return df


if __name__ == "__main__":
	demo = pd.DataFrame({"RedOdds":[-150, 200, -110, np.nan],
						  "BlueOdds":[130, -240, -110, 120]})
	out = add_vig_free_probs(demo)
	print(out[["RedImpP","BlueImpP","Overround","RedFairP","BlueFairP","FairP_Diff"]])
	print("Fair sum ~1:", out["RedFairP"] + out["BlueFairP"]) 
