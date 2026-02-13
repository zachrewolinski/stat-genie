import json

import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    # Load metadata (not strictly needed for analysis but kept for context)
    with open("info.json", "r") as f:
        info = json.load(f)

    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio: students per teacher
    df = df.copy()
    df["students"] = df["feature6"]
    df["teachers"] = df["feature7"]
    df["stratio"] = df["students"] / df["teachers"]

    # Academic performance measures
    df["read"] = df["feature14"]
    df["math"] = df["feature15"]
    df["score_mean"] = df[["read", "math"]].mean(axis=1)

    # Basic descriptives for ratio and scores
    print("N =", len(df))
    print(
        "Student-teacher ratio (students per teacher): "
        f"mean={df['stratio'].mean():.2f}, "
        f"sd={df['stratio'].std():.2f}, "
        f"min={df['stratio'].min():.2f}, "
        f"max={df['stratio'].max():.2f}"
    )

    # Correlations (bivariate associations)
    for label, col in [("reading", "read"), ("math", "math"), ("mean_score", "score_mean")]:
        r, p = stats.pearsonr(df["stratio"], df[col])
        print(f"corr(stratio, {label}) = {r:.3f}, p = {p:.3g}")

    # Simple linear regression: mean score on student-teacher ratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["score_mean"], X_simple).fit()
    coef_simple = model_simple.params["stratio"]
    p_simple = model_simple.pvalues["stratio"]
    r2_simple = model_simple.rsquared
    print(
        "OLS simple: score_mean ~ stratio | "
        f"coef_stratio = {coef_simple:.3f}, p = {p_simple:.3g}, R2 = {r2_simple:.3f}"
    )

    # Adjusted regression controlling for key demographic and resource variables
    covariates = ["feature8", "feature9", "feature12", "feature13", "students"]
    X_adj = sm.add_constant(df[["stratio"] + covariates])
    model_adj = sm.OLS(df["score_mean"], X_adj).fit()
    coef_adj = model_adj.params["stratio"]
    p_adj = model_adj.pvalues["stratio"]
    r2_adj = model_adj.rsquared
    print(
        "OLS adjusted: score_mean ~ stratio + covariates | "
        f"coef_stratio = {coef_adj:.3f}, p = {p_adj:.3g}, R2 = {r2_adj:.3f}"
    )


if __name__ == "__main__":
    main()

