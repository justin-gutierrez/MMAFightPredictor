import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ufcml.utils import add_vig_free_probs

def kelly_fraction(p, q, b):
    """
    Kelly formula for fraction of bankroll to bet.
    p = model's win probability
    q = 1 - p
    b = decimal odds - 1 (profit per $1 stake)
    Returns: fraction of bankroll to stake.
    If edge <= 0, returns 0.
    """
    edge = (p * b) - q
    frac = edge / b
    frac[frac < 0] = 0
    return frac

def backtest_ev(preds_df, edge_threshold=0.03, stake=1.0, kelly=False):
    """
    Run EV backtest.
    preds_df must contain:
      - ModelProb: model's calibrated prob for Red win
      - RedOdds, BlueOdds: American odds
      - Winner: 'Red' or 'Blue'

    edge_threshold: minimum model vs market vig-free prob diff to bet
    stake: flat stake amount if kelly=False
    kelly: if True, use Kelly fractions instead of flat stake
    """
    df = preds_df.copy()
    df = add_vig_free_probs(df)  # adds RedFairP, BlueFairP

    results = []
    for side in ["Red", "Blue"]:
        fair_col = f"{side}FairP"
        if side == "Red":
            model_col = df["ModelProb"].values
        else:
            model_col = (1 - df["ModelProb"]).values
        odds_col = f"{side}Odds"

        # decimal odds
        dec_odds = np.where(df[odds_col] > 0,
                            (df[odds_col] / 100) + 1,
                            (100 / abs(df[odds_col])) + 1)

        edge = model_col - df[fair_col]
        df[f"{side}Edge"] = edge

        bet_mask = edge >= edge_threshold
        bet_count = bet_mask.sum()

        if bet_count == 0:
            continue

        if kelly:
            frac = kelly_fraction(model_col, 1 - model_col, dec_odds - 1)
            stake_amt = frac
        else:
            stake_amt = stake

        # P/L
        wins = ((df["Winner"].str.lower() == side.lower()) & bet_mask)
        losses = ((df["Winner"].str.lower() != side.lower()) & bet_mask)
        pl = (wins * (stake_amt * (dec_odds - 1))) - (losses * stake_amt)

        results.append({
            "Side": side,
            "Bets": int(bet_count),
            "HitRate": float(wins.sum() / bet_count),
            "AvgEdge": float(edge[bet_mask].mean()),
            "TotalStaked": float(stake_amt[bet_mask].sum() if kelly else bet_count * stake),
            "TotalPL": float(pl.sum()),
            "ROI": float(pl.sum() / (stake_amt[bet_mask].sum() if kelly else bet_count * stake))
        })

    return pd.DataFrame(results)

def main():
    # Load your predictions CSV with columns: ModelProb, RedOdds, BlueOdds, Winner
    preds_path = Path("data/reports/holdout_predictions.csv")
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing {preds_path}, export predictions first.")

    preds_df = pd.read_csv(preds_path)

    # Example: flat stake, 3% min edge
    flat_df = backtest_ev(preds_df, edge_threshold=0.03, stake=1.0, kelly=False)
    kelly_df = backtest_ev(preds_df, edge_threshold=0.03, kelly=True)

    out_dir = Path("data/reports/backtests")
    out_dir.mkdir(parents=True, exist_ok=True)

    flat_df.to_csv(out_dir / "flat_edge03.csv", index=False)
    kelly_df.to_csv(out_dir / "kelly_edge03.csv", index=False)

    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "flat_edge03": flat_df.to_dict(orient="records"),
            "kelly_edge03": kelly_df.to_dict(orient="records")
        }, f, indent=2)

    print("Backtest complete. Results saved to", out_dir)

if __name__ == "__main__":
    main()
