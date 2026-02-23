import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    df = df.rename(
        columns={
            "feature6": "enrollment",
            "feature7": "teachers",
            "feature8": "calworks_pct",
            "feature9": "lunch_pct",
            "feature10": "computers",
            "feature11": "expenditure",
            "feature12": "avg_income",
            "feature13": "english_pct",
            "feature14": "read_score",
            "feature15": "math_score",
        }
    )

    # Student-teacher ratio (students per teacher)
    df["stratio"] = df["enrollment"] / df["teachers"]
    df["avg_score"] = df[["read_score", "math_score"]].mean(axis=1)

    print("Descriptive statistics for student-teacher ratio and test scores")
    print(df[["stratio", "read_score", "math_score", "avg_score"]].describe())
    print()

    print("Correlation matrix: student-teacher ratio vs test scores")
    print(df[["stratio", "read_score", "math_score", "avg_score"]].corr())
    print()

    # Bivariate regression: average score on student-teacher ratio
    X1 = sm.add_constant(df["stratio"])
    y_avg = df["avg_score"]
    model1 = sm.OLS(y_avg, X1).fit()
    print("OLS: avg_score ~ stratio (bivariate)")
    print(model1.summary())
    print()

    # Multiple regression with key demographic and resource controls
    controls = ["avg_income", "calworks_pct", "lunch_pct", "english_pct", "expenditure"]
    X2 = sm.add_constant(df[["stratio"] + controls])
    model2 = sm.OLS(y_avg, X2).fit()
    print("OLS: avg_score ~ stratio + controls")
    print(model2.summary())
    print()

    # Separate regressions for reading and math scores with controls
    y_read = df["read_score"]
    y_math = df["math_score"]

    model_read = sm.OLS(y_read, X2).fit()
    print("OLS: read_score ~ stratio + controls")
    print(model_read.summary())
    print()

    model_math = sm.OLS(y_math, X2).fit()
    print("OLS: math_score ~ stratio + controls")
    print(model_math.summary())
    print()


if __name__ == "__main__":
    main()

