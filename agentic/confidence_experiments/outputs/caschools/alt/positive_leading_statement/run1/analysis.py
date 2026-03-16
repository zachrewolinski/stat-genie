import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio and overall test score
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = df[["read", "math"]].mean(axis=1)

    print("Head of data with derived columns:")
    print(df[["school", "students", "teachers", "stratio", "read", "math", "testscr"]].head())

    print("\nCorrelation matrix (key variables):")
    print(
        df[
            [
                "stratio",
                "testscr",
                "read",
                "math",
                "income",
                "english",
                "lunch",
                "calworks",
                "expenditure",
            ]
        ].corr()
    )

    # Simple bivariate regression: test score on student-teacher ratio
    X1 = sm.add_constant(df["stratio"])
    y = df["testscr"]
    model1 = sm.OLS(y, X1).fit()

    print("\nModel 1: testscr ~ stratio")
    print(model1.summary())

    # Multivariate regression controlling for key demographics and resources
    covariates = ["stratio", "income", "english", "lunch", "calworks", "expenditure"]
    X2 = sm.add_constant(df[covariates])
    model2 = sm.OLS(y, X2).fit()

    print("\nModel 2: testscr ~ stratio + controls")
    print(model2.summary())


if __name__ == "__main__":
    main()

