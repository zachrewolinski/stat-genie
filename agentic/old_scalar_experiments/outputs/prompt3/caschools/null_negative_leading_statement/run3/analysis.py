import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing values in variables used
    cols_basic = ["testscr", "stratio"]
    cols_controls = cols_basic + [
        "income",
        "english",
        "lunch",
        "calworks",
        "computer",
        "expenditure",
    ]

    df_basic = df[cols_basic].dropna()
    df_ctrl = df[cols_controls].dropna()

    # Correlation between student-teacher ratio and test scores
    corr = df_basic["testscr"].corr(df_basic["stratio"])

    # Mean test scores by quartiles of student-teacher ratio
    df_basic["str_quartile"] = pd.qcut(df_basic["stratio"], 4, labels=False)
    quartile_means = df_basic.groupby("str_quartile")["testscr"].mean()

    # Simple bivariate regression: testscr ~ stratio
    X_basic = sm.add_constant(df_basic["stratio"])
    model_basic = sm.OLS(df_basic["testscr"], X_basic).fit(cov_type="HC1")
    beta_basic = model_basic.params["stratio"]
    se_basic = model_basic.bse["stratio"]
    t_basic = model_basic.tvalues["stratio"]
    p_basic = model_basic.pvalues["stratio"]

    # Multiple regression with controls
    X_ctrl = sm.add_constant(df_ctrl[["stratio", "income", "english", "lunch", "calworks", "computer", "expenditure"]])
    model_ctrl = sm.OLS(df_ctrl["testscr"], X_ctrl).fit(cov_type="HC1")
    beta_ctrl = model_ctrl.params["stratio"]
    se_ctrl = model_ctrl.bse["stratio"]
    t_ctrl = model_ctrl.tvalues["stratio"]
    p_ctrl = model_ctrl.pvalues["stratio"]

    print("Correlation testscr vs student-teacher ratio:", corr)
    print("\nMean testscr by student-teacher ratio quartile (0=lowest ratio):")
    for q, mean_val in quartile_means.items():
        print(f"  Quartile {int(q)}: mean testscr = {mean_val:.2f}")
    print("\nBivariate OLS: testscr ~ stratio (HC1 SE)")
    print(f"  beta_stratio = {beta_basic:.3f}, se = {se_basic:.3f}, t = {t_basic:.2f}, p = {p_basic:.4f}")
    print("\nMultivariate OLS: testscr ~ stratio + controls (HC1 SE)")
    print(f"  beta_stratio = {beta_ctrl:.3f}, se = {se_ctrl:.3f}, t = {t_ctrl:.2f}, p = {p_ctrl:.4f}")


if __name__ == "__main__":
    main()
