import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("boxes.csv")
    if not data_path.exists():
        raise FileNotFoundError("boxes.csv not found in current directory.")

    df = pd.read_csv(data_path)

    # Basic derived variables
    df["social"] = df["y"].isin([2, 3]).astype(int)
    df_social = df[df["social"] == 1].copy()
    df_social["majority_choice"] = (df_social["y"] == 2).astype(int)

    print("=== Descriptive statistics ===")
    print("Total N:", len(df))
    print("\nOutcome counts (y):")
    print(df["y"].value_counts().sort_index())

    print("\nSocial vs asocial choices:")
    print(df["social"].value_counts())

    print("\nSocial choice rate by culture:")
    social_by_culture = df.groupby("culture")["social"].mean()
    print(social_by_culture)

    print("\nMajority choice rate (among social choosers) by culture:")
    majority_by_culture = df_social.groupby("culture")["majority_choice"].mean()
    print(majority_by_culture)

    print("\n=== Logistic regression: any social choice (y in {2,3}), with age x culture interactions ===")
    formula_social = "social ~ age + C(culture) + age:C(culture) + gender + majority_first"
    model_social = smf.glm(
        formula_social, data=df, family=sm.families.Binomial()
    ).fit()
    print(model_social.summary())

    print("\nKey p-values (social choice model):")
    print(model_social.pvalues)

    print("\n=== Logistic regression: majority vs minority (among social choosers), with age x culture interactions ===")
    formula_majority = (
        "majority_choice ~ age + C(culture) + age:C(culture) + gender + majority_first"
    )
    model_majority = smf.glm(
        formula_majority, data=df_social, family=sm.families.Binomial()
    ).fit()
    print(model_majority.summary())

    print("\nKey p-values (majority vs minority model):")
    print(model_majority.pvalues)

    # Simpler models without interaction terms for robustness
    print(
        "\n=== Logistic regression: any social choice (y in {2,3}), main effects only ==="
    )
    formula_social_main = "social ~ age + C(culture) + gender + majority_first"
    model_social_main = smf.glm(
        formula_social_main, data=df, family=sm.families.Binomial()
    ).fit()
    print(model_social_main.summary())
    print("\nKey p-values (social choice main-effects model):")
    print(model_social_main.pvalues)

    print(
        "\n=== Logistic regression: majority vs minority (among social choosers), main effects only ==="
    )
    formula_majority_main = (
        "majority_choice ~ age + C(culture) + gender + majority_first"
    )
    model_majority_main = smf.glm(
        formula_majority_main, data=df_social, family=sm.families.Binomial()
    ).fit()
    print(model_majority_main.summary())
    print("\nKey p-values (majority vs minority main-effects model):")
    print(model_majority_main.pvalues)

    # Predicted probabilities at representative ages for interpretability
    ages = np.array([5, 8, 11, 14])
    cultures = sorted(df["culture"].unique())

    print("\n=== Predicted probabilities: social choice by age and culture ===")
    for c in cultures:
        for a in ages:
            row = {
                "age": a,
                "culture": c,
                "gender": 1,
                "majority_first": 1,
            }
            pred = model_social.predict(pd.DataFrame([row]))[0]
            print(f"culture={c}, age={a}: P(social)={pred:.3f}")

    print("\n=== Predicted probabilities: majority choice (among social) by age and culture ===")
    for c in cultures:
        for a in ages:
            row = {
                "age": a,
                "culture": c,
                "gender": 1,
                "majority_first": 1,
            }
            pred = model_majority.predict(pd.DataFrame([row]))[0]
            print(f"culture={c}, age={a}: P(majority|social)={pred:.3f}")


if __name__ == "__main__":
    main()
