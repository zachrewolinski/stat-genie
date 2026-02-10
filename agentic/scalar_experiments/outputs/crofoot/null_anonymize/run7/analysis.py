import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base_path = Path(__file__).parent

    # Load metadata (not strictly needed for modeling, but for clarity/context)
    info_path = base_path / "info.json"
    with info_path.open() as f:
        info = json.load(f)

    # Load data
    data_path = base_path / "crofoot.csv"
    df = pd.read_csv(data_path)

    # Outcome: 1 if focal won, 0 otherwise
    y = df["feature4"]

    # Key predictors based on research question:
    # - Relative group size: difference in group size (focal - other)
    # - Contest location: relative home-range proximity (focal closer than other)
    df = df.copy()
    df["rel_size"] = df["feature7"] - df["feature8"]
    df["focal_closer_home"] = (df["feature5"] < df["feature6"]).astype(int)

    # Simple logistic regression with both predictors
    X = df[["rel_size", "focal_closer_home"]]
    X = sm.add_constant(X)

    model = sm.Logit(y, X)
    try:
        result = model.fit(disp=False)
    except Exception:
        # In case of complete separation or small-sample issues, fall back to
        # a penalized fit for robustness.
        result = model.fit_regularized(disp=False)

    params = result.params
    pvalues = result.pvalues if hasattr(result, "pvalues") else None

    # Heuristic evidence score combining effect sizes and (when available) p-values.
    # Larger positive coefficients and small p-values imply stronger support.
    size_beta = float(params.get("rel_size", 0.0))
    loc_beta = float(params.get("focal_closer_home", 0.0))

    if pvalues is not None:
        size_p = float(pvalues.get("rel_size", 1.0))
        loc_p = float(pvalues.get("focal_closer_home", 1.0))
    else:
        size_p = 1.0
        loc_p = 1.0

    # Map each effect to an evidence score between -1 and 1.
    def effect_score(beta: float, p: float) -> float:
        # Sign from beta, magnitude from |beta| and p-value.
        # Cap |beta| at 2 for stability.
        mag = min(abs(beta), 2.0) / 2.0  # 0..1
        # Convert p-value to confidence (0..1), emphasizing small p.
        conf = 1.0 - min(max(p, 0.0), 1.0)
        raw = mag * conf
        return raw if beta >= 0 else -raw

    size_score = effect_score(size_beta, size_p)
    loc_score = effect_score(loc_beta, loc_p)

    # Combine the two, giving equal weight.
    combined_score = (size_score + loc_score) / 2.0

    # Treat negative combined scores as evidence against the joint hypothesis
    # that relative size and contest location meaningfully increase win odds.
    # Map to Likert -100..100.
    scalar = int(round(combined_score * 100))

    # Safety clamp to ensure within [-100, 100]
    scalar = max(-100, min(100, scalar))

    conclusion_path = base_path / "conclusion.txt"
    with conclusion_path.open("w") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

