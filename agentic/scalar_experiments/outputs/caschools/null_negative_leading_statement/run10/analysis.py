import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    print("Basic summary:")
    print(df[["stratio", "testscr"]].describe())
    print()

    corr = df["stratio"].corr(df["testscr"])
    print(f"Correlation between student-teacher ratio and test score: {corr:.4f}")
    print("(Negative means lower ratio is associated with higher scores.)")
    print()

    # Simple bivariate regression
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()
    print("Bivariate OLS: testscr ~ stratio")
    print(model_simple.summary())
    print()

    # Multiple regression controlling for key demographics and resources
    controls = [
        "income",
        "english",
        "lunch",
        "calworks",
        "expenditure",
        "computer",
    ]
    X_controls = sm.add_constant(df[["stratio"] + controls])
    model_controls = sm.OLS(df["testscr"], X_controls).fit()
    print("Multiple OLS with controls: testscr ~ stratio + controls")
    print(model_controls.summary())


if __name__ == "__main__":
    main()

