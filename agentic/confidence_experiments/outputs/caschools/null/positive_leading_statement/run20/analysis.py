import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in variables used
    cols = [
        "testscr",
        "stratio",
        "income",
        "english",
        "lunch",
        "calworks",
        "expenditure",
        "computer",
        "students",
    ]
    data = df[cols].dropna()

    # Correlation between student-teacher ratio and test scores
    r, pval = stats.pearsonr(data["stratio"], data["testscr"])

    print("Bivariate correlation between STR and test score:")
    print(f"  r = {r:.3f}, p-value = {pval:.4g}")

    # Simple bivariate regression
    X_simple = sm.add_constant(data["stratio"])
    model_simple = sm.OLS(data["testscr"], X_simple).fit()
    print("\nSimple OLS: testscr ~ stratio")
    print(model_simple.summary())

    # Multiple regression with key covariates
    X_cov = data[
        [
            "stratio",
            "income",
            "english",
            "lunch",
            "calworks",
            "expenditure",
            "computer",
            "students",
        ]
    ]
    X_cov = sm.add_constant(X_cov)
    model_cov = sm.OLS(data["testscr"], X_cov).fit()
    print("\nMultiple OLS with covariates:")
    print(model_cov.summary())

    # Also look at mean scores by STR quartile
    data["str_quartile"] = pd.qcut(data["stratio"], 4, labels=False)
    group_means = data.groupby("str_quartile")["testscr"].agg(["mean", "count"])
    print("\nMean test scores by STR quartile (0 = lowest STR, 3 = highest STR):")
    print(group_means.to_string())


if __name__ == "__main__":
    main()

