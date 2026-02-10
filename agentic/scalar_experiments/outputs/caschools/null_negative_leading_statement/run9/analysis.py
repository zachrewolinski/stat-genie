import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Simple correlation between student-teacher ratio and test scores
    corr = df["stratio"].corr(df["testscr"])

    # Regression controlling for observable covariates
    covariates = [
        "stratio",
        "income",
        "english",
        "lunch",
        "calworks",
        "computer",
        "expenditure",
    ]
    reg_df = df.dropna(subset=covariates + ["testscr"])
    X = sm.add_constant(reg_df[covariates])
    y = reg_df["testscr"]
    model = sm.OLS(y, X).fit()

    print("Correlation(stratio, testscr):", corr)
    print("\nOLS regression of testscr on stratio and controls")
    print(model.summary())


if __name__ == "__main__":
    main()

