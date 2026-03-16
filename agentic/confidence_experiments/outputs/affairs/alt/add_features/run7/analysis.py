from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Basic cleaning / feature construction
    df = df.copy()
    df["any_affair"] = (df["affairs"] > 0).astype(int)
    df["children_yes"] = (df["children"].str.lower() == "yes").astype(int)

    # Descriptive statistics: affair rates by children
    summary = (
        df.groupby("children")["any_affair"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "affair_rate"})
    )
    print("Affair rate by children:")
    print(summary)
    print()

    # Simple logistic regression: any_affair ~ children_yes
    X_simple = sm.add_constant(df["children_yes"])
    model_simple = sm.Logit(df["any_affair"], X_simple).fit(disp=False)
    print("Simple logistic regression: any_affair ~ children_yes")
    print(model_simple.summary())
    print()

    # Multiple logistic regression controlling for key covariates
    # Use numeric encodings of ordered categorical variables as given
    covariates = [
        "children_yes",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]
    X_full = sm.add_constant(df[covariates])
    model_full = sm.Logit(df["any_affair"], X_full).fit(disp=False)
    print("Full logistic regression with controls")
    print(model_full.summary())
    print()

    # Extract key results for children effect
    child_coef_simple = model_simple.params["children_yes"]
    child_p_simple = model_simple.pvalues["children_yes"]

    child_coef_full = model_full.params["children_yes"]
    child_p_full = model_full.pvalues["children_yes"]

    # Compute odds ratios for interpretability
    or_simple = float(np.exp(child_coef_simple))
    or_full = float(np.exp(child_coef_full))

    print("Children effect (simple model):")
    print(f"  coef = {child_coef_simple:.3f}, p = {child_p_simple:.3f}, OR = {or_simple:.3f}")
    print("Children effect (full model):")
    print(f"  coef = {child_coef_full:.3f}, p = {child_p_full:.3f}, OR = {or_full:.3f}")


if __name__ == "__main__":
    main()
