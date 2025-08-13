# src/ufcml/calibration.py
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import joblib

@dataclass
class Band:
    lo: float
    hi: float

DEFAULT_BANDS = [
    Band(0.00, 0.20),
    Band(0.20, 0.40),
    Band(0.40, 0.60),
    Band(0.60, 0.80),
    Band(0.80, 1.01),  # include 1.0
]

class SideBandCalibrator:
    """
    Calibrates model probabilities per side ('Red','Blue') within market
    fair-probability bands (vig-free), using IsotonicRegression.
    - For RED: fit mapping f_red_band(p_red) -> P(Red wins)
    - For BLUE: fit mapping f_blue_band(p_blue) -> P(Blue wins)
    """
    def __init__(self, bands: Optional[List[Band]] = None):
        self.bands = bands or DEFAULT_BANDS
        self.red_cals: Dict[Tuple[float,float], IsotonicRegression] = {}
        self.blue_cals: Dict[Tuple[float,float], IsotonicRegression] = {}

    def _band_key(self, p: float) -> Tuple[float,float]:
        for b in self.bands:
            if (p >= b.lo) and (p < b.hi):
                return (b.lo, b.hi)
        # edge case p==1.0
        return (self.bands[-1].lo, self.bands[-1].hi)

    def fit(self, df: pd.DataFrame) -> "SideBandCalibrator":
        """
        df must include:
          - ModelProb (prob Red wins)
          - RedFairP, BlueFairP (vig-free market probs)
          - Winner ('Red'/'Blue')
        """
        need = {"ModelProb","RedFairP","BlueFairP","Winner"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for calibration: {missing}")

        y_red = (df["Winner"].str.lower() == "red").astype(int).values
        y_blue = (df["Winner"].str.lower() == "blue").astype(int).values
        p_red = df["ModelProb"].astype(float).values
        p_blue = (1.0 - df["ModelProb"].astype(float).values)

        red_fair = df["RedFairP"].astype(float).values
        blue_fair = df["BlueFairP"].astype(float).values

        # Fit per band for RED
        for b in self.bands:
            msk = (red_fair >= b.lo) & (red_fair < b.hi)
            if msk.sum() >= 20:  # minimum for a stable isotonic
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(p_red[msk], y_red[msk])
                self.red_cals[(b.lo,b.hi)] = iso

        # Fit per band for BLUE
        for b in self.bands:
            msk = (blue_fair >= b.lo) & (blue_fair < b.hi)
            if msk.sum() >= 20:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(p_blue[msk], y_blue[msk])
                self.blue_cals[(b.lo,b.hi)] = iso

        return self

    def predict_red(self, p_red: np.ndarray, red_fair: np.ndarray) -> np.ndarray:
        out = np.asarray(p_red, dtype=float).copy()
        for i in range(len(out)):
            key = self._band_key(float(red_fair[i]))
            iso = self.red_cals.get(key, None)
            if iso is not None:
                out[i] = float(iso.predict([out[i]])[0])
        return out

    def predict_blue(self, p_blue: np.ndarray, blue_fair: np.ndarray) -> np.ndarray:
        out = np.asarray(p_blue, dtype=float).copy()
        for i in range(len(out)):
            key = self._band_key(float(blue_fair[i]))
            iso = self.blue_cals.get(key, None)
            if iso is not None:
                out[i] = float(iso.predict([out[i]])[0])
        return out

    def save(self, path: str) -> None:
        joblib.dump({
            "bands": self.bands,
            "red": self.red_cals,
            "blue": self.blue_cals
        }, path)

    @classmethod
    def load(cls, path: str) -> "SideBandCalibrator":
        d = joblib.load(path)
        obj = cls(bands=d["bands"])
        obj.red_cals = d["red"]
        obj.blue_cals = d["blue"]
        return obj
