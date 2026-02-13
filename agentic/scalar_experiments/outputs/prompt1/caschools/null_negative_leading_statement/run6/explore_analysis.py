import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    print("Basic description:")
    print(df[["stratio", "testscr", "read", "math"]].describe())
    print("\nCorrelation matrix:")
    print(df[["stratio", "testscr", "read", "math"]].corr())

    # Check distribution of student-teacher ratios and restrict to a central range
    q01, q99 = df["stratio"].quantile([0.01, 0.99])
    print(f"\nstratio percentiles: 1% = {q01:.2f}, 99% = {q99:.2f}")
    df_central = df[(df["stratio"] >= q01) & (df["stratio"] <= q99)]
    print("Central 98% of stratio values description:")
    print(df_central["stratio"].describe())
    print("Correlation in central range (stratio vs testscr):")
    print(df_central[["stratio", "testscr"]].corr())

    # Simple bivariate regression
    print("\nOLS: testscr ~ stratio")
    model_simple = smf.ols("testscr ~ stratio", data=df).fit()
    print(model_simple.summary())

    # Multiple regression with key controls
    print("\nOLS: testscr ~ stratio + income + english + lunch + calworks + computer + expenditure")
    model_controls = smf.ols(
        "testscr ~ stratio + income + english + lunch + calworks + computer + expenditure",
        data=df,
    ).fit()
    print(model_controls.summary())


if __name__ == "__main__":
    main()
