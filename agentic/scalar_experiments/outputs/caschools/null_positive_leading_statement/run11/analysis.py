import pandas as pd
import statsmodels.api as sm
from pathlib import Path


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct student–teacher ratio and average test score
    df["stu_teach_ratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in variables we use
    vars_of_interest = [
        "avg_score",
        "stu_teach_ratio",
        "income",
        "english",
        "lunch",
        "calworks",
        "expenditure",
        "computer",
    ]
    df_model = df[vars_of_interest].dropna()

    # Simple correlation between ratio and performance
    corr = df_model["stu_teach_ratio"].corr(df_model["avg_score"])

    # Regression of performance on ratio and key controls
    X = df_model[
        [
            "stu_teach_ratio",
            "income",
            "english",
            "lunch",
            "calworks",
            "expenditure",
            "computer",
        ]
    ]
    X = sm.add_constant(X)
    y = df_model["avg_score"]
    model = sm.OLS(y, X).fit()

    coef_ratio = model.params["stu_teach_ratio"]
    pval_ratio = model.pvalues["stu_teach_ratio"]
    r2 = model.rsquared

    # Map evidence to a Likert-style scalar from -100 to 100.
    # Lower student–teacher ratio is "better" performance if coef is negative
    # (because ratio = students per teacher).
    if pval_ratio < 0.001 and coef_ratio < 0:
        base = 90
    elif pval_ratio < 0.01 and coef_ratio < 0:
        base = 75
    elif pval_ratio < 0.05 and coef_ratio < 0:
        base = 60
    elif pval_ratio < 0.1 and coef_ratio < 0:
        base = 40
    elif coef_ratio < 0:
        base = 20
    elif pval_ratio < 0.05 and coef_ratio > 0:
        base = -40
    elif coef_ratio > 0:
        base = -20
    else:
        base = 0

    # Adjust slightly for effect size (scaled R^2)
    adjustment = int(10 * r2)
    if coef_ratio < 0:
        scalar = base + adjustment
    else:
        scalar = base - adjustment

    # Clip to [-100, 100] and cast to int
    scalar = max(min(int(round(scalar)), 100), -100)

    # Write scalar conclusion to file, as required
    Path("conclusion.txt").write_text(str(scalar), encoding="utf-8")

    # Also print a concise summary to stdout for human inspection
    print("Correlation (ratio vs avg_score):", corr)
    print("Coefficient on ratio:", coef_ratio)
    print("p-value for ratio:", pval_ratio)
    print("R-squared:", r2)
    print("Likert scalar written to conclusion.txt:", scalar)


if __name__ == "__main__":
    main()

