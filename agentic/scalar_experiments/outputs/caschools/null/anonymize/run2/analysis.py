import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    numeric_cols = [
        "feature6",
        "feature7",
        "feature8",
        "feature9",
        "feature10",
        "feature11",
        "feature12",
        "feature13",
        "feature14",
        "feature15",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["stratio"] = df["feature6"] / df["feature7"]
    df["testscr"] = (df["feature14"] + df["feature15"]) / 2.0

    print("Descriptive statistics for key variables:")
    print(df[["stratio", "testscr"]].describe())
    print()

    corr = df[["stratio", "testscr"]].corr().iloc[0, 1]
    print(f"Correlation between student-teacher ratio and test score: {corr:.4f}")
    print()

    # Bivariate regression: testscr ~ stratio
    X1 = sm.add_constant(df[["stratio"]])
    y = df["testscr"]
    model1 = sm.OLS(y, X1, missing="drop").fit()
    print("Bivariate OLS: testscr ~ stratio")
    print(model1.summary())
    print()

    # Multiple regression with demographic and resource controls
    covariates = [
        "stratio",
        "feature8",   # % CalWorks
        "feature9",   # % reduced-price lunch
        "feature10",  # number of computers
        "feature11",  # expenditure per student
        "feature12",  # district average income
        "feature13",  # % English learners
    ]
    X2 = sm.add_constant(df[covariates])
    model2 = sm.OLS(y, X2, missing="drop").fit()
    print("Multiple OLS: testscr ~ stratio + controls")
    print(model2.summary())


if __name__ == "__main__":
    main()

