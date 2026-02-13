import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Map shuffled column names to their semantic meaning based on info.json.
    enroll = df["english"]  # total enrollment
    teachers_n = df["students"]  # number of teachers (FTE)
    read_scr = df["district"]  # average reading score
    math_scr = df["expenditure"]  # average math score

    # Construct key variables.
    df["stratio"] = enroll / teachers_n  # student-teacher ratio
    df["testscr"] = (read_scr + math_scr) / 2.0  # overall test score

    # Potential confounders.
    df["avginc"] = df["income"]  # district average income
    df["calw_pct"] = df["school"]  # % qualifying for CalWorks
    df["lunch_pct"] = df["computer"]  # % qualifying for reduced-price lunch
    df["ell_pct"] = df["rownames"]  # % English learners

    # Basic summaries.
    corr = df["stratio"].corr(df["testscr"])

    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()

    X_ctrl = df[["stratio", "avginc", "calw_pct", "lunch_pct", "ell_pct"]]
    X_ctrl = sm.add_constant(X_ctrl)
    model_ctrl = sm.OLS(df["testscr"], X_ctrl).fit()

    # Quartile comparison for effect size intuition.
    df["str_quartile"] = pd.qcut(df["stratio"], 4, labels=False)
    low_str = df[df["str_quartile"] == 0]["testscr"].mean()
    high_str = df[df["str_quartile"] == 3]["testscr"].mean()

    print("N:", len(df))
    print("stratio mean:", df["stratio"].mean())
    print("stratio sd:", df["stratio"].std())
    print("testscr mean:", df["testscr"].mean())
    print("testscr sd:", df["testscr"].std())
    print("corr(stratio, testscr):", corr)
    print("Simple OLS coef(stratio):", model_simple.params["stratio"])
    print("Simple OLS p-value(stratio):", model_simple.pvalues["stratio"])
    print("Simple OLS R^2:", model_simple.rsquared)
    print("Controls OLS coef(stratio):", model_ctrl.params["stratio"])
    print("Controls OLS p-value(stratio):", model_ctrl.pvalues["stratio"])
    print("Controls OLS R^2:", model_ctrl.rsquared)
    print("Mean testscr low STR (Q1):", low_str)
    print("Mean testscr high STR (Q4):", high_str)
    print("Difference (low - high):", low_str - high_str)


if __name__ == "__main__":
    main()

