import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    print("Number of districts:", len(df))
    print()

    print("Student-teacher ratio (stratio) summary:")
    print(df["stratio"].describe())
    print()

    print("Test score (testscr) summary:")
    print(df["testscr"].describe())
    print()

    corr = df["stratio"].corr(df["testscr"])
    print(f"Correlation between stratio and testscr: {corr:.3f}")
    print()

    # Simple bivariate regression: testscr ~ stratio
    X1 = sm.add_constant(df["stratio"])
    model1 = sm.OLS(df["testscr"], X1).fit(cov_type="HC1")
    print("Model 1: testscr ~ stratio")
    print(model1.summary())
    print()

    # Multiple regression with standard demographic and resource controls
    controls = ["income", "english", "lunch", "expenditure", "computer"]
    X2 = df[["stratio"] + controls]
    X2 = sm.add_constant(X2)
    model2 = sm.OLS(df["testscr"], X2).fit(cov_type="HC1")
    print("Model 2: testscr ~ stratio + controls")
    print(model2.summary())
    print()

    # Explore non-parametric relationship via quantiles
    df["str_bin"] = pd.qcut(df["stratio"], 5, labels=False)
    grouped = df.groupby("str_bin")["stratio", "testscr"].agg(
        {"stratio": "mean", "testscr": "mean"}
    )
    print("Average test scores by student-teacher ratio quintile:")
    print(grouped)


if __name__ == "__main__":
    main()

