import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("crofoot.csv")
    df = pd.read_csv(data_path)

    # Outcome: 1 if focal group won, 0 otherwise
    df["focal_win"] = df["feature4"].astype(int)

    # Relative group size: focal / other and difference
    df["rel_group_size_ratio"] = df["feature7"] / df["feature8"]
    df["rel_group_size_diff"] = df["feature7"] - df["feature8"]

    # Contest location: relative home-range advantage.
    # Distances (m) from each group's home-range center: smaller = closer to home.
    # Positive value means focal group is closer to its home-range center than the other group is to its own.
    df["rel_location_advantage"] = df["feature6"] - df["feature5"]

    # Standardize predictors for numerical stability and interpretability
    predictors = ["rel_group_size_ratio", "rel_location_advantage"]
    X = df[predictors].copy()
    X = (X - X.mean()) / X.std(ddof=0)
    X = sm.add_constant(X)

    y = df["focal_win"]

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    print("Logistic regression: focal_win ~ rel_group_size_ratio + rel_location_advantage")
    print(result.summary())

    # Also fit an alternative model using simple differences as a robustness check.
    predictors_alt = ["rel_group_size_diff", "rel_location_advantage"]
    X_alt = df[predictors_alt].copy()
    X_alt = (X_alt - X_alt.mean()) / X_alt.std(ddof=0)
    X_alt = sm.add_constant(X_alt)

    logit_model_alt = sm.Logit(y, X_alt)
    result_alt = logit_model_alt.fit(disp=False)

    print("\nLogistic regression (alternative): focal_win ~ rel_group_size_diff + rel_location_advantage")
    print(result_alt.summary())


if __name__ == "__main__":
    main()

