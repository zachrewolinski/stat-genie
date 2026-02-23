import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Map shuffled column names to their semantic meanings based on info.json.
    df = df.copy()
    df["enrollment"] = pd.to_numeric(df["english"], errors="coerce")
    df["teachers_fte"] = pd.to_numeric(df["students"], errors="coerce")
    df["avg_income"] = pd.to_numeric(df["income"], errors="coerce")
    df["ell_pct"] = pd.to_numeric(df["rownames"], errors="coerce")
    df["calworks_pct"] = pd.to_numeric(df["school"], errors="coerce")
    df["lunch_pct"] = pd.to_numeric(df["computer"], errors="coerce")
    df["expn_stu"] = pd.to_numeric(df["grades"], errors="coerce")
    df["read_scr"] = pd.to_numeric(df["district"], errors="coerce")
    df["math_scr"] = pd.to_numeric(df["expenditure"], errors="coerce")

    # Academic performance: average of reading and math scores.
    df["testscr"] = (df["read_scr"] + df["math_scr"]) / 2.0

    # Student–teacher ratio: students per teacher.
    df["stratio"] = df["enrollment"] / df["teachers_fte"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["testscr", "stratio"])

    print("N used:", len(df))
    print("\nStudent–teacher ratio (stratio) summary:\n", df["stratio"].describe())
    print("\nTest score (testscr) summary:\n", df["testscr"].describe())

    corr = df[["stratio", "testscr"]].corr().loc["stratio", "testscr"]
    print("\nCorrelation between stratio and testscr:", corr)

    model_simple = smf.ols("testscr ~ stratio", data=df).fit()
    print("\nSimple OLS: testscr ~ stratio")
    print(model_simple.summary())
    print("\nCoefficient on stratio:", model_simple.params["stratio"])
    print("p-value for stratio:", model_simple.pvalues["stratio"])
    print("R-squared:", model_simple.rsquared)

    # Add key covariates to control for observable differences across districts.
    model_controls = smf.ols(
        "testscr ~ stratio + avg_income + ell_pct + calworks_pct + lunch_pct + expn_stu",
        data=df,
    ).fit()
    print("\nOLS with controls: testscr ~ stratio + controls")
    print(model_controls.summary())
    print("\nCoefficient on stratio (controls):", model_controls.params["stratio"])
    print("p-value for stratio (controls):", model_controls.pvalues["stratio"])
    print("R-squared (controls):", model_controls.rsquared)

    # Effect size over the interdecile range of stratio.
    q10, q50, q90 = df["stratio"].quantile([0.1, 0.5, 0.9])
    print("\nstratio quantiles (10%, 50%, 90%):", (q10, q50, q90))

    beta_simple = model_simple.params["stratio"]
    delta_testscr_simple = beta_simple * (q90 - q10)
    print(
        "Predicted testscr change (simple model) when stratio moves\n"
        f"from 10th to 90th percentile: {delta_testscr_simple:.3f} points"
    )

    beta_ctrl = model_controls.params["stratio"]
    delta_testscr_ctrl = beta_ctrl * (q90 - q10)
    print(
        "Predicted testscr change (controls model) when stratio moves\n"
        f"from 10th to 90th percentile: {delta_testscr_ctrl:.3f} points"
    )


if __name__ == "__main__":
    main()

