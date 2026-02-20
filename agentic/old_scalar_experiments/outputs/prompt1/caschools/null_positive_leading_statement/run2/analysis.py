import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio (students per teacher)
    df["stratio"] = df["students"] / df["teachers"]

    # Overall academic performance: average of reading and math scores
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with any missing values in key variables, if any
    key_cols = ["stratio", "testscr", "income", "english", "lunch", "calworks", "computer", "expenditure"]
    df_model = df.dropna(subset=key_cols).copy()

    # Basic correlation between student-teacher ratio and test scores
    corr = df_model["stratio"].corr(df_model["testscr"])

    # Simple linear regression: testscr ~ stratio
    X_simple = sm.add_constant(df_model["stratio"])
    model_simple = sm.OLS(df_model["testscr"], X_simple).fit()
    coef_simple = model_simple.params["stratio"]
    pval_simple = model_simple.pvalues["stratio"]

    # Multivariable regression controlling for observable confounders
    predictors = ["stratio", "income", "english", "lunch", "calworks", "computer", "expenditure"]
    X_full = sm.add_constant(df_model[predictors])
    model_full = sm.OLS(df_model["testscr"], X_full).fit()
    coef_full = model_full.params["stratio"]
    pval_full = model_full.pvalues["stratio"]

    # Descriptive comparison: mean test scores by quartile of student-teacher ratio
    df_model["stratio_quartile"] = pd.qcut(df_model["stratio"], 4, labels=False)
    group_means = df_model.groupby("stratio_quartile")["testscr"].mean()

    print("Number of districts used:", len(df_model))
    print("Student-teacher ratio summary:")
    print(df_model["stratio"].describe())
    print("\nTest score summary (testscr = (read+math)/2):")
    print(df_model["testscr"].describe())

    print("\nCorrelation between student-teacher ratio and test scores:")
    print(f"  corr(stratio, testscr) = {corr:.3f}")

    print("\nSimple OLS: testscr ~ stratio")
    print(f"  Coefficient on stratio: {coef_simple:.3f}")
    print(f"  p-value: {pval_simple:.3g}")

    print("\nMultivariable OLS: testscr ~ stratio + income + english + lunch + calworks + computer + expenditure")
    print(f"  Coefficient on stratio (adjusted): {coef_full:.3f}")
    print(f"  p-value: {pval_full:.3g}")

    print("\nMean test scores by student-teacher ratio quartile (0 = lowest ratio, 3 = highest ratio):")
    for q, mean_val in group_means.items():
        print(f"  Quartile {int(q)}: mean testscr = {mean_val:.2f}")

    # Robustness check: restrict to a plausible range of student-teacher ratios
    # Here we focus on districts with ratios between 10 and 30 students per teacher.
    df_trim = df_model[(df_model["stratio"] >= 10) & (df_model["stratio"] <= 30)].copy()
    if len(df_trim) > 0:
        corr_trim = df_trim["stratio"].corr(df_trim["testscr"])
        X_trim = sm.add_constant(df_trim["stratio"])
        model_trim = sm.OLS(df_trim["testscr"], X_trim).fit()

        print("\nRobustness (10 <= stratio <= 30):")
        print(f"  N (trimmed) = {len(df_trim)}")
        print(f"  corr(stratio, testscr) = {corr_trim:.3f}")
        print(f"  OLS coef on stratio: {model_trim.params['stratio']:.3f}")
        print(f"  p-value: {model_trim.pvalues['stratio']:.3g}")
    else:
        print("\nRobustness (10 <= stratio <= 30): no observations in this range.")


if __name__ == "__main__":
    main()
