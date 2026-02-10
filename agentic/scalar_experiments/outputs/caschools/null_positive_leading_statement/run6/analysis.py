import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio: more students per teacher = larger value
    df["stratio"] = df["students"] / df["teachers"]
    df["avgscore"] = (df["read"] + df["math"]) / 2.0

    # Simple correlations
    corr_read = df["stratio"].corr(df["read"])
    corr_math = df["stratio"].corr(df["math"])
    corr_avg = df["stratio"].corr(df["avgscore"])

    print("Correlation(stratio, read):", corr_read)
    print("Correlation(stratio, math):", corr_math)
    print("Correlation(stratio, avgscore):", corr_avg)

    # Simple linear regression of avgscore on stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["avgscore"], X_simple).fit()
    print("\nSimple regression avgscore ~ stratio")
    print(model_simple.summary())

    # Multiple regression controlling for key demographics and resources
    controls = ["income", "calworks", "lunch", "english", "expenditure", "computer"]
    X_controls = df[["stratio"] + controls].copy()
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(df["avgscore"], X_controls).fit()
    print("\nMultiple regression avgscore ~ stratio + controls")
    print(model_controls.summary())


if __name__ == "__main__":
    main()

