import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # According to info.json descriptions, "english" is total enrollment
    # and "students" is the number of teachers.
    df = df.copy()
    df["stratio"] = df["english"] / df["students"]

    # Academic performance: use both reading and math, plus their mean.
    df["testscr_read"] = df["district"]
    df["testscr_math"] = df["expenditure"]
    df["testscr_mean"] = (df["testscr_read"] + df["testscr_math"]) / 2.0

    # Basic descriptives
    print("N:", len(df))
    print("stratio summary:")
    print(df["stratio"].describe())
    print()

    for outcome in ["testscr_read", "testscr_math", "testscr_mean"]:
        print(f"=== Outcome: {outcome} ===")
        y = df[outcome]
        x = df["stratio"]
        corr = x.corr(y)
        print("Pearson corr(stratio, outcome):", corr)

        x_const = sm.add_constant(x)
        model = sm.OLS(y, x_const).fit()
        print("OLS coef on stratio:", model.params["stratio"])
        print("OLS p-value on stratio:", model.pvalues["stratio"])
        print("R-squared:", model.rsquared)
        print()

    # Multiple regression with key controls for the mean test score
    # Based on info.json descriptions:
    #   school   -> % CalWorks (income assistance)
    #   computer -> % reduced-price lunch
    #   grades   -> expenditure per student
    #   income   -> district average income (in $1,000)
    #   rownames -> % English learners
    controls = ["income", "school", "computer", "grades", "rownames"]
    cols = ["testscr_mean", "stratio"] + controls
    df_reg = df[cols].dropna()

    y = df_reg["testscr_mean"]
    X = df_reg[["stratio"] + controls]
    X = sm.add_constant(X)
    model_ctrl = sm.OLS(y, X).fit()

    print("=== Multiple regression: testscr_mean ~ stratio + controls ===")
    print(model_ctrl.summary())


if __name__ == "__main__":
    main()

