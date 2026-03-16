import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Basic cleaning: drop rows missing key variables
    required_cols = ["students", "teachers", "read", "math"]
    df = df.dropna(subset=required_cols)

    # Construct student-teacher ratio and overall test score
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = df[["read", "math"]].mean(axis=1)

    # Simple correlation
    corr = df["stratio"].corr(df["testscr"])

    # Simple linear regression: testscr ~ stratio
    X_simple = sm.add_constant(df[["stratio"]])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()

    # Multiple regression with common covariates to check robustness
    covariates = ["income", "english", "lunch", "calworks", "computer", "expenditure"]
    covariates = [c for c in covariates if c in df.columns]
    X_full = sm.add_constant(df[["stratio"] + covariates])
    model_full = sm.OLS(df["testscr"], X_full).fit()

    print("Number of districts used:", len(df))
    print("\nCorrelation between student-teacher ratio and test score:")
    print(f"  corr(stratio, testscr) = {corr:.3f}")

    print("\nSimple regression: testscr ~ stratio")
    print(model_simple.summary())

    print("\nMultiple regression: testscr ~ stratio + controls")
    print(model_full.summary())


if __name__ == "__main__":
    main()

