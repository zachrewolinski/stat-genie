import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base = Path(__file__).parent
    info_path = base / "info.json"
    data_path = base / "caschools.csv"

    with info_path.open() as f:
        info = json.load(f)

    df = pd.read_csv(data_path)

    # student-teacher ratio: enrollment / teachers
    df["stratio"] = df["feature6"] / df["feature7"]

    # academic performance: mean of reading and math scores
    df["testscr"] = df[["feature14", "feature15"]].mean(axis=1)

    # drop rows with missing or infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["stratio", "testscr"])

    # simple linear regression: testscr ~ stratio
    X = sm.add_constant(df["stratio"])
    y = df["testscr"]
    model = sm.OLS(y, X).fit()

    coef = model.params["stratio"]
    r2 = model.rsquared

    # We expect a *negative* association: higher ratio -> lower scores.
    # Map effect size and fit into a Likert score in [-100, 100].
    # Use standardized coefficient magnitude as a measure of strength.
    stratio_std = df["stratio"].std()
    testscr_std = df["testscr"].std()
    beta_std = coef * stratio_std / testscr_std

    # Combine standardized beta and R^2 into a single strength metric.
    strength = abs(beta_std) * (0.5 + 0.5 * r2)

    # Cap strength to avoid extreme outliers, then rescale.
    strength_cap = min(strength, 1.0)

    # Direction: negative coefficient => evidence for the research hypothesis.
    if coef < 0:
        scalar = 20 + int(round(strength_cap * 80))
    else:
        scalar = -20 - int(round(strength_cap * 80))

    # Ensure within [-100, 100]
    scalar = int(max(-100, min(100, scalar)))

    # Write scalar conclusion
    conclusion_path = base / "conclusion.txt"
    with conclusion_path.open("w") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

