import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio: students per teacher (higher = larger classes).
    df["stratio"] = df["students"] / df["teachers"]

    # Overall academic performance: average of reading and math scores.
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    print("Basic descriptives")
    print("------------------")
    print(df[["stratio", "testscr"]].describe())
    print()

    # Simple Pearson correlation between class size and test scores.
    corr = df["stratio"].corr(df["testscr"])
    print(f"Correlation between stratio and testscr: {corr:.4f}")
    print()

    # Simple linear regression: testscr ~ stratio.
    X1 = sm.add_constant(df["stratio"])
    model1 = sm.OLS(df["testscr"], X1).fit()
    print("Model 1: testscr ~ stratio")
    print(model1.summary())
    print()

    # Multiple regression with standard controls.
    controls = ["income", "english", "lunch", "calworks", "expenditure", "computer"]
    X2 = sm.add_constant(df[["stratio"] + controls])
    model2 = sm.OLS(df["testscr"], X2).fit(cov_type="HC1")
    print("Model 2: testscr ~ stratio + controls (HC1 robust SEs)")
    print(model2.summary())
    print()

    # Separate subject scores as a robustness check.
    for outcome in ["read", "math"]:
        print(f"Model 3 ({outcome}): {outcome} ~ stratio + controls (HC1 robust SEs)")
        y = df[outcome]
        X = sm.add_constant(df[["stratio"] + controls])
        model = sm.OLS(y, X).fit(cov_type="HC1")
        print(model.summary())
        print()


if __name__ == "__main__":
    main()

