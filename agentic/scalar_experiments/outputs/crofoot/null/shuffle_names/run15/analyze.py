import json

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Map columns to their semantic meaning based on info.json descriptions
    win_focal = df["m_focal"]  # 1 if focal group won, 0 otherwise

    # Group sizes
    focal_size = df["f_other"]  # "Number of individuals in focal group"
    other_size = df["win"]  # "Number of individuals in other group"

    # Distances from home range centers
    dist_focal_center = df["m_other"]  # distance of focal group from its home-range center
    dist_other_center = df["n_focal"]  # distance of other group from its home-range center

    # Derived predictors
    df["log_rel_size"] = np.log(focal_size / other_size)
    # Positive values: opponent is farther from its center than focal is from its center
    df["rel_location"] = dist_other_center - dist_focal_center

    X = df[["log_rel_size", "rel_location"]]
    X = sm.add_constant(X)

    model = sm.Logit(win_focal, X).fit(disp=False)

    print("Logistic regression: focal win ~ relative group size + relative location")
    print(model.summary())
    print("\nCoefficients:")
    print(model.params)
    print("\nP-values:")
    print(model.pvalues)

    # Simple descriptive checks
    df["focal_bigger"] = (focal_size > other_size).astype(int)
    df["focal_home_advantage"] = (dist_focal_center < dist_other_center).astype(int)

    print("\nWin rate by focal larger vs smaller/equal:")
    print(df.groupby("focal_bigger")["m_focal"].mean())

    print("\nWin rate by focal home-range advantage (closer to center):")
    print(df.groupby("focal_home_advantage")["m_focal"].mean())


if __name__ == "__main__":
    main()

