import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_metadata(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Compute student-teacher ratio
    df["stratio"] = df["students"] / df["teachers"]
    return df


def run_regression(df: pd.DataFrame):
    # Outcome: average of reading and math scores as overall performance
    df = df.copy()
    df["score"] = df[["read", "math"]].mean(axis=1)

    # Main regressor: student-teacher ratio
    X = df[["stratio", "income", "lunch", "english", "calworks"]].copy()
    X = sm.add_constant(X)
    y = df["score"]

    model = sm.OLS(y, X, missing="drop").fit()
    return model


def map_effect_to_scalar(beta: float, se: float) -> int:
    """Map estimated effect and uncertainty to Likert-scale [-100, 100].

    Positive beta (lower ratio -> higher performance) should yield positive score.
    Note: beta here is effect of stratio, so negative beta implies
    lower ratio (smaller classes) associated with higher scores.
    """

    if np.isnan(beta) or np.isnan(se) or se <= 0:
        return 0

    # We want effect of a DECREASE in stratio on score.
    # If y = a + b * stratio, then dy/d(-stratio) = -b.
    effect = -beta

    # Compute t-statistic magnitude for that implied effect
    t_value = abs(beta / se)

    # Base strength from t-stat (cap at 10)
    strength = min(t_value / 3.0, 1.0)  # 0 ~ no evidence, 1 ~ very strong

    # Normalize effect size using a plausible range: say 5-point improvement per 5-student change
    # so 1 point per 1-student change is already big. Scale beta accordingly.
    scaled_effect = np.tanh(effect / 1.0)  # squash to [-1, 1]

    raw_score = scaled_effect * strength * 100.0

    # Clip to [-100, 100] and round to nearest integer
    score = int(np.clip(np.round(raw_score), -100, 100))
    return score


def main():
    base = Path(__file__).resolve().parent
    meta = load_metadata(base / "info.json")
    df = load_data(base / "caschools.csv")

    model = run_regression(df)
    beta = model.params["stratio"]
    se = model.bse["stratio"]

    scalar = map_effect_to_scalar(beta, se)

    # Write scalar conclusion only
    conclusion_path = base / "conclusion.txt"
    conclusion_path.write_text(str(scalar))


if __name__ == "__main__":
    main()
