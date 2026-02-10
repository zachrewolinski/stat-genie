import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["score"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in the variables we use
    df_model = df[
        [
            "score",
            "stratio",
            "calworks",
            "lunch",
            "english",
            "income",
            "expenditure",
        ]
    ].dropna()

    print("Number of districts used:", len(df_model))
    print("Student-teacher ratio summary:")
    print(df_model["stratio"].describe())
    print("\nAcademic performance summary (average of read & math):")
    print(df_model["score"].describe())

    corr = df_model["stratio"].corr(df_model["score"])
    print("\nPearson correlation between student-teacher ratio and score:")
    print(corr)

    # Linear regression with controls
    y = df_model["score"]
    X = df_model[
        ["stratio", "calworks", "lunch", "english", "income", "expenditure"]
    ]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    print("\nOLS regression of score on student-teacher ratio and controls:")
    print(model.summary())

    print("\nCoefficient on student-teacher ratio:")
    print(model.params["stratio"])
    print("Standard error:", model.bse["stratio"])
    print("t-statistic:", model.tvalues["stratio"])
    print("p-value:", model.pvalues["stratio"])


if __name__ == "__main__":
    main()

