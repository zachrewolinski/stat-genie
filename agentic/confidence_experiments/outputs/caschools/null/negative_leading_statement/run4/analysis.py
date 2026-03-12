import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["testscr"] = (df["read"] + df["math"]) / 2.0
    df["stratio"] = df["students"] / df["teachers"]

    # Drop rows with missing values in variables of interest
    vars_basic = ["testscr", "read", "math", "stratio"]
    df_basic = df.dropna(subset=vars_basic).copy()

    print("Number of observations (non-missing):", len(df_basic))
    print()

    # Descriptive statistics for student-teacher ratio and performance
    print("Descriptive statistics:")
    print(df_basic[["stratio", "testscr", "read", "math"]].describe())
    print()

    # Correlations between ratio and scores
    print("Pearson correlations between student-teacher ratio (students/teacher)")
    print("and academic performance measures:")
    for outcome in ["testscr", "read", "math"]:
        r, p = stats.pearsonr(df_basic["stratio"], df_basic[outcome])
        print(f"  stratio vs {outcome}: r = {r:.3f}, p = {p:.4g}")
    print()

    # Simple OLS regression: testscr on student-teacher ratio
    X1 = sm.add_constant(df_basic["stratio"])
    model1 = sm.OLS(df_basic["testscr"], X1).fit()
    coef1 = model1.params["stratio"]
    pval1 = model1.pvalues["stratio"]
    ci1_low, ci1_high = model1.conf_int().loc["stratio"]

    print("OLS regression: testscr ~ stratio")
    print(f"  Coefficient on stratio: {coef1:.3f}")
    print(f"  95% CI: [{ci1_low:.3f}, {ci1_high:.3f}]")
    print(f"  p-value: {pval1:.4g}")
    print()

    # Multiple regression controlling for observable confounders
    controls = ["income", "calworks", "lunch", "english", "computer", "expenditure"]
    vars_full = ["testscr", "stratio"] + controls
    df_full = df.dropna(subset=vars_full).copy()

    X2 = sm.add_constant(df_full[["stratio"] + controls])
    model2 = sm.OLS(df_full["testscr"], X2).fit()
    coef2 = model2.params["stratio"]
    pval2 = model2.pvalues["stratio"]
    ci2_low, ci2_high = model2.conf_int().loc["stratio"]

    print("OLS regression with controls:")
    print("  Specification: testscr ~ stratio + income + calworks + lunch")
    print("                  + english + computer + expenditure")
    print(f"  Coefficient on stratio: {coef2:.3f}")
    print(f"  95% CI: [{ci2_low:.3f}, {ci2_high:.3f}]")
    print(f"  p-value: {pval2:.4g}")
    print()

    # Non-parametric comparison: mean scores by quartiles of student-teacher ratio
    df_basic["stratio_q"] = pd.qcut(df_basic["stratio"], 4, labels=False)
    group_means = df_basic.groupby("stratio_q")[["stratio", "testscr"]].agg(
        {"stratio": "mean", "testscr": "mean"}
    )
    print("Mean test scores by student-teacher ratio quartile")
    print("(Quartile 0 = lowest ratio, Quartile 3 = highest ratio):")
    print(group_means)


if __name__ == "__main__":
    main()

