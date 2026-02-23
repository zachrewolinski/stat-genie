import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Construct key predictors related to the research question
    df["size_ratio"] = df["n_focal"] / df["n_other"]
    # Home-field indicator: 1 if focal group is closer to its home-range center
    df["home_adv"] = (df["dist_focal"] < df["dist_other"]).astype(int)
    df["dist_diff"] = df["dist_other"] - df["dist_focal"]

    y = df["win"]

    # Full model with size and two location-related terms
    X_full = df[["size_ratio", "home_adv", "dist_diff"]]
    X_full = sm.add_constant(X_full)
    model_full = sm.Logit(y, X_full).fit(disp=False)

    print("Logistic regression results for win ~ size_ratio + home_adv + dist_diff")
    print(model_full.summary2())

    # Size-only model
    X_size = sm.add_constant(df[["size_ratio"]])
    model_size = sm.Logit(y, X_size).fit(disp=False)
    print("\nLogistic regression results for win ~ size_ratio")
    print(model_size.summary2())

    # Location-only models
    X_dist = sm.add_constant(df[["dist_diff"]])
    model_dist = sm.Logit(y, X_dist).fit(disp=False)
    print("\nLogistic regression results for win ~ dist_diff")
    print(model_dist.summary2())

    X_home = sm.add_constant(df[["home_adv"]])
    model_home = sm.Logit(y, X_home).fit(disp=False)
    print("\nLogistic regression results for win ~ home_adv")
    print(model_home.summary2())

    # Predicted probabilities under contrasting scenarios for interpretation
    size_q25, size_q75 = df["size_ratio"].quantile([0.25, 0.75])
    dist_q25, dist_q75 = df["dist_diff"].quantile([0.25, 0.75])

    def predict_prob(size_ratio: float, home_adv: int, dist_diff: float) -> float:
        row = pd.DataFrame(
            {
                "const": [1.0],
                "size_ratio": [size_ratio],
                "home_adv": [home_adv],
                "dist_diff": [dist_diff],
            }
        )
        return float(model_full.predict(row)[0])

    scenarios = {
        "small_group_no_home_adv": predict_prob(size_q25, 0, dist_q75),
        "small_group_home_adv": predict_prob(size_q25, 1, dist_q25),
        "large_group_no_home_adv": predict_prob(size_q75, 0, dist_q75),
        "large_group_home_adv": predict_prob(size_q75, 1, dist_q25),
    }

    print("\nScenario predicted win probabilities:")
    for name, prob in scenarios.items():
        print(f"{name}: {prob:.3f}")

    print("\nMean win rate:", y.mean())
    print("N contests:", len(df))


if __name__ == "__main__":
    main()
