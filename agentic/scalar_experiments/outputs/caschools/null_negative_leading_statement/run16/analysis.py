import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2

    # Drop any rows with missing values in variables used
    cols = [
        "testscr",
        "stratio",
        "income",
        "english",
        "lunch",
        "calworks",
        "computer",
        "expenditure",
    ]
    work = df[cols].dropna().copy()

    print("Number of observations:", len(work))
    print()

    # Simple bivariate association
    corr = work["testscr"].corr(work["stratio"])
    print("Correlation between test score and student-teacher ratio:", corr)
    print()

    work = sm.add_constant(work)

    # Simple OLS: testscr ~ stratio
    model_simple = sm.OLS(work["testscr"], work[["const", "stratio"]]).fit()
    print("Simple regression: testscr ~ stratio")
    print(model_simple.summary())
    print()

    # Multiple OLS with key covariates
    covariates = [
        "stratio",
        "income",
        "english",
        "lunch",
        "calworks",
        "computer",
        "expenditure",
    ]
    model_full = sm.OLS(work["testscr"], work[["const"] + covariates]).fit()
    print("Multiple regression: testscr ~ stratio + controls")
    print(model_full.summary())


if __name__ == "__main__":
    main()

