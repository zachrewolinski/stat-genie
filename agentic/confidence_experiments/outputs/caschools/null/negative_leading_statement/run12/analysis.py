import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio and a combined test score
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing key variables (should be none, but be safe)
    df = df.dropna(subset=["stratio", "testscr"])

    print("Number of districts:", len(df))
    print("Student-teacher ratio summary:")
    print(df["stratio"].describe())
    print("\nTest score summary:")
    print(df["testscr"].describe())

    # Simple correlation
    corr = df["stratio"].corr(df["testscr"])
    print(f"\nPearson correlation between STR and testscr: {corr:.3f}")

    # Simple linear regression: testscr ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()
    print("\nSimple OLS: testscr ~ stratio")
    print(model_simple.summary())

    # Multiple regression with key covariates that may confound the relationship
    covariates = ["income", "english", "lunch", "calworks", "expenditure"]
    available_covariates = [c for c in covariates if c in df.columns]
    if available_covariates:
        X_multi = sm.add_constant(df[["stratio"] + available_covariates])
        model_multi = sm.OLS(df["testscr"], X_multi).fit()
        print("\nMultiple OLS: testscr ~ stratio + controls")
        print(model_multi.summary())
    else:
        print("\nNo covariates available for multiple regression.")


if __name__ == "__main__":
    main()

