import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ufcml.utils import add_vig_free_probs
from ufcml.calibration import SideBandCalibrator

def american_to_decimal(odds_series: pd.Series) -> pd.Series:
    o = pd.to_numeric(odds_series, errors="coerce")
    dec = np.where(o > 0, (o / 100.0) + 1.0, (100.0 / np.abs(o)) + 1.0)
    return pd.Series(dec, index=odds_series.index)

def kelly_fraction(p: pd.Series, b: pd.Series):
    """
    Kelly fraction for a +b net decimal odds (decimal_odds = 1 + b).
    p: model win probability for the side being bet.
    b: profit per $1 stake in decimal-odds space (decimal_odds - 1).
    """
    q = 1.0 - p
    edge = (p * b) - q
    frac = edge / b
    frac = frac.where(frac > 0, 0.0)  # no negative bets
    frac = frac.fillna(0.0)
    return frac

def eval_side(df: pd.DataFrame, side: str, threshold: float, flat_stake: float,
              use_kelly: bool, kelly_mult: float, kelly_cap: float,
              side_filter="both", max_edge=0.30):
    """
    Evaluate one side (Red/Blue) given a threshold and staking policy.
    Returns dict of metrics.
    """
    assert side in ("Red", "Blue")
    
    # Respect side filter
    if side_filter != "both" and side_filter.lower() != side.lower():
        return {
            "Side": side, "Threshold": threshold, "Kelly": use_kelly,
            "KellyMult": kelly_mult, "Bets": 0, "HitRate": 0.0,
            "AvgEdge": 0.0, "TotalStaked": 0.0, "TotalPL": 0.0, "ROI": 0.0
        }
    
    side_win = df["Winner"].str.lower() == side.lower()

    if side == "Red":
        model_p = df["ModelProb"]
        fair_p = df["RedFairP"]
        odds_col = "RedOdds"
    else:
        model_p = 1.0 - df["ModelProb"]
        fair_p = df["BlueFairP"]
        odds_col = "BlueOdds"

    edge = model_p - fair_p
    # discard overly large edges which are often overconfident errors
    edge = edge.clip(upper=max_edge)
    bet_mask = edge >= threshold

    if bet_mask.sum() == 0:
        return {
            "Side": side, "Threshold": threshold, "Kelly": use_kelly,
            "KellyMult": kelly_mult, "Bets": 0, "HitRate": 0.0,
            "AvgEdge": float(edge.mean() if np.isfinite(edge.mean()) else 0.0),
            "TotalStaked": 0.0, "TotalPL": 0.0, "ROI": 0.0
        }

    dec_odds = american_to_decimal(df.loc[bet_mask, odds_col])
    b = dec_odds - 1.0

    if use_kelly:
        frac = kelly_fraction(model_p.loc[bet_mask], b) * float(kelly_mult)
        if kelly_cap is not None and kelly_cap > 0:
            frac = frac.clip(upper=float(kelly_cap))
        stake_amt = frac
    else:
        stake_amt = pd.Series(flat_stake, index=dec_odds.index, dtype=float)

    wins = side_win.loc[bet_mask]
    losses = ~wins

    pl = wins.astype(float) * (stake_amt * b) - losses.astype(float) * stake_amt

    total_staked = float(stake_amt.sum())
    total_pl = float(pl.sum())
    roi = float(total_pl / total_staked) if total_staked > 0 else 0.0

    return {
        "Side": side,
        "Threshold": threshold,
        "Kelly": use_kelly,
        "KellyMult": kelly_mult,
        "Bets": int(bet_mask.sum()),
        "HitRate": float(wins.mean()),
        "AvgEdge": float(edge.loc[bet_mask].mean()),
        "TotalStaked": total_staked,
        "TotalPL": total_pl,
        "ROI": roi,
    }

def sweep(preds_df: pd.DataFrame, thresholds, stake, use_kelly, kelly_multipliers, kelly_cap, 
          side_filter="both", max_edge=0.30):
    """
    For each threshold, compute metrics for Red, Blue, and Overall
    under the specified staking policies.
    Returns two DataFrames: per_side_df, overall_df.
    """
    preds = preds_df.copy()
    preds = add_vig_free_probs(preds)  # adds RedFairP/BlueFairP
    per_side_rows = []
    overall_rows = []

    for th in thresholds:
        if use_kelly:
            kms = kelly_multipliers
        else:
            kms = [1.0]  # dummy

        for km in kms:
            red = eval_side(preds, "Red", th, stake, use_kelly, km, kelly_cap,
                           side_filter=side_filter, max_edge=max_edge)
            blue = eval_side(preds, "Blue", th, stake, use_kelly, km, kelly_cap,
                            side_filter=side_filter, max_edge=max_edge)

            per_side_rows.extend([red, blue])

            # Overall aggregation (sum PL, sum stake, combine bets)
            total_bets = red["Bets"] + blue["Bets"]
            total_staked = red["TotalStaked"] + blue["TotalStaked"]
            total_pl = red["TotalPL"] + blue["TotalPL"]
            roi = (total_pl / total_staked) if total_staked > 0 else 0.0

            overall_rows.append({
                "Threshold": th,
                "Kelly": use_kelly,
                "KellyMult": km if use_kelly else 0.0,
                "Bets": int(total_bets),
                "TotalStaked": float(total_staked),
                "TotalPL": float(total_pl),
                "ROI": float(roi),
                # Weighted hit-rate by number of bets on each side (approx):
                "RedBets": red["Bets"], "RedROI": red["ROI"],
                "BlueBets": blue["Bets"], "BlueROI": blue["ROI"],
            })

    per_side_df = pd.DataFrame(per_side_rows)
    overall_df = pd.DataFrame(overall_rows)
    return per_side_df, overall_df

def parse_thresholds(thresh_str: str):
    # e.g. "0.01,0.02,0.03,0.05,0.07"
    return [float(x.strip()) for x in thresh_str.split(",") if x.strip()]

def parse_kelly_mults(mult_str: str):
    # e.g. "1.0,0.5,0.25"
    return [float(x.strip()) for x in mult_str.split(",") if x.strip()]

def main():
    parser = argparse.ArgumentParser(description="Multi-threshold EV backtest sweep.")
    parser.add_argument("--preds", type=str, default="data/reports/holdout_predictions.csv",
                        help="CSV with Date, RedFighter, BlueFighter, Winner, RedOdds, BlueOdds, ModelProb")
    parser.add_argument("--thresholds", type=str, default="0.01,0.02,0.03,0.05,0.07",
                        help="Comma-separated edge thresholds to test (e.g., '0.02,0.03,0.05')")
    parser.add_argument("--stake", type=float, default=1.0, help="Flat stake per bet (if not using Kelly)")
    parser.add_argument("--kelly", action="store_true", help="Use Kelly staking")
    parser.add_argument("--kelly_multipliers", type=str, default="1.0,0.5",
                        help="Comma-separated Kelly multipliers (only used with --kelly)")
    parser.add_argument("--kelly_cap", type=float, default=0.25,
                        help="Maximum Kelly fraction per bet (e.g., 0.25). Set 0 to disable.")
    parser.add_argument("--side_filter", type=str, default="both",
                        choices=["both","red","blue"],
                        help="Only bet this side")
    parser.add_argument("--min_fairp_red", type=float, default=0.0)
    parser.add_argument("--max_fairp_red", type=float, default=1.0)
    parser.add_argument("--min_fairp_blue", type=float, default=0.0)
    parser.add_argument("--max_fairp_blue", type=float, default=1.0)
    parser.add_argument("--max_edge", type=float, default=0.30,
                        help="Ignore bets with absolute edge above this (cap overconfident cases)")
    parser.add_argument("--calibrator", type=str, default="",
                        help="Path to side+band calibrator.joblib to recalibrate probs before computing edges")
    args = parser.parse_args()

    preds_path = Path(args.preds)
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {preds_path}")

    preds_df = pd.read_csv(preds_path)

    # Basic validation
    req_cols = {"RedOdds", "BlueOdds", "Winner", "ModelProb"}
    missing = req_cols - set(preds_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {preds_path}: {missing}")

    thresholds = parse_thresholds(args.thresholds)
    kelly_multipliers = parse_kelly_mults(args.kelly_multipliers)

    # Add vig-free probabilities
    preds_df = add_vig_free_probs(preds_df)

    # Optionally recalibrate p_red per side×band
    if args.calibrator:
        cal = SideBandCalibrator.load(args.calibrator)
        p_red = preds_df["ModelProb"].astype(float).values
        p_blue = 1.0 - p_red
        p_red_cal = cal.predict_red(p_red, preds_df["RedFairP"].values)
        p_blue_cal = cal.predict_blue(p_blue, preds_df["BlueFairP"].values)
        # enforce symmetry
        preds_df["ModelProb"] = np.clip(p_red_cal, 1e-6, 1-1e-6)

    # Probability band filters
    r_lo, r_hi = args.min_fairp_red, args.max_fairp_red
    b_lo, b_hi = args.min_fairp_blue, args.max_fairp_blue
    mask_red_band = (preds_df["RedFairP"] >= r_lo) & (preds_df["RedFairP"] <= r_hi)
    mask_blue_band = (preds_df["BlueFairP"] >= b_lo) & (preds_df["BlueFairP"] <= b_hi)
    # Keep rows if they fall in either side's allowed bands (so either side can be considered)
    preds_df = preds_df[mask_red_band | mask_blue_band].reset_index(drop=True)

    out_dir = Path("data/reports/backtests/sweeps")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run sweep for flat stake
    per_side_flat, overall_flat = sweep(
        preds_df, thresholds=thresholds, stake=args.stake,
        use_kelly=False, kelly_multipliers=[1.0], kelly_cap=args.kelly_cap,
        side_filter=args.side_filter, max_edge=args.max_edge
    )
    per_side_flat.to_csv(out_dir / "per_side_flat.csv", index=False)
    overall_flat.to_csv(out_dir / "overall_flat.csv", index=False)

    # Run sweep for Kelly (if requested)
    if args.kelly:
        per_side_kelly, overall_kelly = sweep(
            preds_df, thresholds=thresholds, stake=args.stake,
            use_kelly=True, kelly_multipliers=kelly_multipliers, kelly_cap=args.kelly_cap,
            side_filter=args.side_filter, max_edge=args.max_edge
        )
        per_side_kelly.to_csv(out_dir / "per_side_kelly.csv", index=False)
        overall_kelly.to_csv(out_dir / "overall_kelly.csv", index=False)
    else:
        per_side_kelly, overall_kelly = pd.DataFrame(), pd.DataFrame()

    # JSON summary
    summary = {
        "config": {
            "thresholds": thresholds,
            "stake": args.stake,
            "kelly": args.kelly,
            "kelly_multipliers": kelly_multipliers,
                    "kelly_cap": args.kelly_cap,
        "side_filter": args.side_filter,
        "min_fairp_red": args.min_fairp_red,
        "max_fairp_red": args.max_fairp_red,
        "min_fairp_blue": args.min_fairp_blue,
        "max_fairp_blue": args.max_fairp_blue,
        "max_edge": args.max_edge,
        "calibrator": args.calibrator,
        "preds": str(preds_path),
        },
        "flat": {
            "per_side_path": str(out_dir / "per_side_flat.csv"),
            "overall_path": str(out_dir / "overall_flat.csv"),
            "overall": overall_flat.to_dict(orient="records"),
        },
        "kelly": {
            "per_side_path": str(out_dir / "per_side_kelly.csv") if args.kelly else None,
            "overall_path": str(out_dir / "overall_kelly.csv") if args.kelly else None,
            "overall": overall_kelly.to_dict(orient="records") if args.kelly else [],
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Sweep complete. Outputs in:", out_dir)

if __name__ == "__main__":
    main()
