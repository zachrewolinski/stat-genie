import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

BASE_DIR = Path(__file__).parent


def main() -> None:
    info = json.loads((BASE_DIR / "info.json").read_text())
    question = info["research_questions"][0]

    df = pd.read_csv(BASE_DIR / "caschools.csv")

    # student-teacher ratio: total enrollment / number of teachers
    df = df.copy()
    df["stratio"] = df["feature6"] / df["feature7"]

    # academic performance: average of reading and math scores
    df["avg_score"] = (df["feature14"] + df["feature15"]) / 2.0

    # drop any problematic rows
    df = df.replace([pd.NA, float("inf"), -float("inf")], pd.NA).dropna(subset=["stratio", "avg_score"])

    X = sm.add_constant(df[["stratio"]])
    y = df["avg_score"]

    model = sm.OLS(y, X).fit()

    coef = model.params["stratio"]
    p_value = model.pvalues["stratio"]
    r2 = model.rsquared

    # Map evidence strength to Likert-style integer in [-100, 100].
    # Negative coefficient means lower ratio (smaller classes) -> higher scores.
    if p_value < 0.001 and abs(coef) > 1.0:
        base = 90
    elif p_value < 0.01 and abs(coef) > 0.5:
        base = 70
    elif p_value < 0.05 and abs(coef) > 0.25:
        base = 50
    elif p_value < 0.1 and abs(coef) > 0.1:
        base = 30
    else:
        base = 0

    scalar = -base if coef > 0 else base

    scalar = max(-100, min(100, int(round(scalar))))

    (BASE_DIR / "conclusion.txt").write_text(str(scalar))


if __name__ == "__main__":
    main()
