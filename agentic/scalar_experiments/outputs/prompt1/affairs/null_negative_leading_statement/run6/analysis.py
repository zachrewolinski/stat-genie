import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Create binary indicator for any affair
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Map children to binary indicator (1 = yes, 0 = no)
    df["children_yes"] = (df["children"] == "yes").astype(int)

    # Descriptive statistics by children status
    desc = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            median_affairs=("affairs", "median"),
            prop_with_affair=("has_affair", "mean"),
            count=("affairs", "size"),
        )
        .reset_index()
    )

    print("Descriptive statistics by children status:")
    print(desc.to_string(index=False))
    print()

    # Logistic regression: any affair ~ children + controls
    # Encode gender as binary (1 = male, 0 = female) for simplicity
    df["male"] = (df["gender"] == "male").astype(int)

    predictors = [
        "children_yes",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
        "male",
    ]

    X = df[predictors]
    X = sm.add_constant(X)
    y = df["has_affair"]

    logit_model = sm.Logit(y, X)
    logit_result = logit_model.fit(disp=False)

    print("Logistic regression results (dependent variable: has_affair):")
    print(logit_result.summary())
    print()

    children_coef = logit_result.params["children_yes"]
    children_pvalue = logit_result.pvalues["children_yes"]

    print(f"Coefficient for children_yes: {children_coef:.4f}")
    print(f"P-value for children_yes: {children_pvalue:.4g}")


if __name__ == "__main__":
    main()

