import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["teachers_per_100_students"] = df["teachers"] / df["students"] * 100
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    print("Basic description")
    print(df[["stratio", "teachers_per_100_students", "avg_score"]].describe())
    print("\nCorrelations:")
    print(df[["stratio", "teachers_per_100_students", "avg_score"]].corr())

    # Simple bivariate regression
    X_simple = sm.add_constant(df["stratio"])
    y = df["avg_score"]
    model_simple = sm.OLS(y, X_simple, missing="drop").fit()
    print("\nSimple OLS: avg_score ~ stratio")
    print(model_simple.summary())

    X_simple_alt = sm.add_constant(df["teachers_per_100_students"])
    model_simple_alt = sm.OLS(y, X_simple_alt, missing="drop").fit()
    print("\nSimple OLS: avg_score ~ teachers_per_100_students")
    print(model_simple_alt.summary())

    # Multiple regression with key covariates
    covariates = [
        "income",
        "english",
        "lunch",
        "calworks",
        "expenditure",
        "computer",
    ]
    X_multi = sm.add_constant(df[["stratio"] + covariates])
    model_multi = sm.OLS(y, X_multi, missing="drop").fit()
    print("\nMultiple OLS: avg_score ~ stratio + controls")
    print(model_multi.summary())


if __name__ == "__main__":
    main()
