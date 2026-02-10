import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def compute_effect_score(coef: float, pval: float) -> float:
    """
    Map a standardized logistic coefficient and its p-value to [0, 1],
    representing strength of evidence that the predictor influences win
    probability (direction-agnostic).
    """
    # Magnitude component: cap very large effects
    mag = min(1.0, abs(coef) / 2.0)

    # Significance component: p=0 -> 1, p>=0.5 -> 0, linear in between
    sig = 1.0 - min(1.0, max(0.0, pval) / 0.5)

    return mag * sig


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group wins, 0 otherwise
    y = df["feature4"].astype(float).to_numpy()

    # Relative group size: difference in total individuals
    size_diff = (df["feature7"] - df["feature8"]).to_numpy()

    # Contest location advantage: how much closer the focal group is to
    # its own home range center than the opponent is to its own center.
    # Positive values indicate a focal "home-range" advantage.
    loc_adv = (df["feature6"] - df["feature5"]).to_numpy()

    X_raw = np.column_stack([size_diff, loc_adv])

    # Standardize predictors to make coefficients comparable
    means = X_raw.mean(axis=0)
    stds = X_raw.std(axis=0, ddof=0)
    # Avoid division by zero
    stds[stds == 0] = 1.0
    X_std = (X_raw - means) / stds

    X = sm.add_constant(X_std, has_constant="add")

    overall_strength: float

    try:
        model = sm.Logit(y, X)
        result = model.fit(disp=False)

        coef_size = float(result.params[1])
        coef_loc = float(result.params[2])
        p_size = float(result.pvalues[1])
        p_loc = float(result.pvalues[2])

        s_size = compute_effect_score(coef_size, p_size)
        s_loc = compute_effect_score(coef_loc, p_loc)

        overall_strength = (s_size + s_loc) / 2.0
    except Exception:
        # Fallback: use absolute correlations if logistic regression fails
        def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
            if a.size == 0 or b.size == 0:
                return 0.0
            if np.all(a == a[0]) or np.all(b == b[0]):
                return 0.0
            corr = np.corrcoef(a, b)[0, 1]
            if np.isnan(corr):
                return 0.0
            return float(abs(corr))

        corr_size = safe_corr(size_diff, y)
        corr_loc = safe_corr(loc_adv, y)
        overall_strength = (corr_size + corr_loc) / 2.0

    # Ensure the strength is in [0, 1]
    overall_strength = max(0.0, min(1.0, overall_strength))

    # Map overall evidence strength in [0,1] to Likert [-100, 100]
    # 0   -> -100 (very strong "No, no influence")
    # 0.5 ->   0 (neutral / inconclusive)
    # 1   -> 100 (very strong "Yes, clear influence")
    scalar = int(round(200.0 * (overall_strength - 0.5)))
    scalar = max(-100, min(100, scalar))

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(str(scalar), encoding="utf-8")


if __name__ == "__main__":
    main()

