import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Derived variables
    df["social_choice"] = df["y"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)
    df["demonstrated_choice"] = df["y"].isin([2, 3])

    # Basic descriptive summaries
    n = len(df)
    majority_rate = df["majority_choice"].mean()
    social_rate = df["social_choice"].mean()

    print(f"Total N: {n}")
    print("Outcome distribution (y: 1=undemo, 2=majority, 3=minority):")
    print(df["y"].value_counts(normalize=True).sort_index())
    print("\nOverall social information use (2 or 3 vs 1):")
    print(f"Proportion social_choice: {social_rate:.3f}")
    print("Overall majority choice (2 vs others):")
    print(f"Proportion majority_choice: {majority_rate:.3f}")

    # By culture
    print("\nMajority choice rate by culture:")
    print(df.groupby("culture")["majority_choice"].mean())

    print("\nSocial choice rate by culture:")
    print(df.groupby("culture")["social_choice"].mean())

    # Age summaries
    print("\nMajority choice rate by age (years):")
    print(df.groupby("age")["majority_choice"].mean())

    print("\nSocial choice rate by age (years):")
    print(df.groupby("age")["social_choice"].mean())

    # Logistic regression models
    # Model 1: Social information use ~ age + culture + gender + majority_first
    model_social = smf.logit(
        "social_choice ~ age + C(culture) + gender + majority_first", data=df
    ).fit(disp=False)
    print("\nLogit model: social_choice ~ age + C(culture) + gender + majority_first")
    print(model_social.summary())

    # Model 2: Majority preference vs all others ~ age + culture + gender + majority_first
    model_majority = smf.logit(
        "majority_choice ~ age + C(culture) + gender + majority_first", data=df
    ).fit(disp=False)
    print(
        "\nLogit model: majority_choice ~ age + C(culture) + gender + majority_first"
    )
    print(model_majority.summary())

    # Restricted sample: only children who chose a demonstrated option, majority vs minority
    df_demo = df[df["demonstrated_choice"]].copy()
    df_demo["majority_vs_minority"] = (df_demo["y"] == 2).astype(int)

    model_majority_demo = smf.logit(
        "majority_vs_minority ~ age + C(culture) + gender + majority_first",
        data=df_demo,
    ).fit(disp=False)
    print(
        "\nLogit model (among demonstrated choosers): "
        "majority_vs_minority ~ age + C(culture) + gender + majority_first"
    )
    print(model_majority_demo.summary())

    # Collect key statistics to help manual interpretation downstream if needed
    results = {
        "overall": {
            "n": int(n),
            "majority_rate": float(majority_rate),
            "social_rate": float(social_rate),
        },
        "by_culture": {
            "majority_rate": df.groupby("culture")["majority_choice"].mean()
            .to_dict(),
            "social_rate": df.groupby("culture")["social_choice"].mean()
            .to_dict(),
        },
        "by_age": {
            "majority_rate": df.groupby("age")["majority_choice"].mean()
            .to_dict(),
            "social_rate": df.groupby("age")["social_choice"].mean()
            .to_dict(),
        },
        "models": {
            "social_choice": {
                "params": model_social.params.to_dict(),
                "pvalues": model_social.pvalues.to_dict(),
            },
            "majority_choice": {
                "params": model_majority.params.to_dict(),
                "pvalues": model_majority.pvalues.to_dict(),
            },
            "majority_vs_minority": {
                "params": model_majority_demo.params.to_dict(),
                "pvalues": model_majority_demo.pvalues.to_dict(),
            },
        },
    }

    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

